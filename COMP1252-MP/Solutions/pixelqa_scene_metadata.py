"""Read one Sentinel SAFE scene's pixel-QA prerequisites without raster I/O.

Only bounded XML/GML metadata and directory entries are read. A complete
metadata record is not evidence that a crop is clear: pixel and visual checks
must still follow. Paths are relative to ``root_dir`` when possible.
"""

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from annotation_scene_inventory import MAX_METADATA_BYTES, _scene_record


CLOUD_BANDS = ("B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12")
FEATURE_BANDS = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")
QUALITY_BANDS = tuple(dict.fromkeys((*CLOUD_BANDS, *FEATURE_BANDS)))
QUALITY_TYPES = ("NODATA", "SATURA", "DEFECT", "TECQUA")


def _local_name(tag):
    return tag.rsplit("}", 1)[-1]


def _path(path, root_dir):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(root_dir))
    except ValueError:
        return str(path)


def _angle(group, name, issues, maximum, inclusive_maximum=False):
    elements = [e for e in group.iter() if _local_name(e.tag) == name]
    label = "solar_zenith" if name == "ZENITH_ANGLE" else "solar_azimuth"
    if len(elements) != 1:
        issues.append(f"missing_or_ambiguous_{label}")
        return None
    element = elements[0]
    if element.attrib.get("unit", "deg").lower() not in ("deg", "degree", "degrees"):
        issues.append(f"unsupported_{label}_unit")
        return None
    try:
        value = float(element.text)
    except (TypeError, ValueError):
        issues.append(f"invalid_{label}")
        return None
    within_range = 0 <= value <= maximum if inclusive_maximum else 0 <= value < maximum
    if not math.isfinite(value) or not within_range:
        issues.append(f"invalid_{label}")
        return None
    return value


def _sun_angles(safe, issues):
    if safe is None:
        return None, None
    matches = sorted(safe.glob("GRANULE/*/MTD_TL.xml"))
    if len(matches) != 1:
        issues.append("missing_or_ambiguous_solar_metadata")
        return None, None
    path = matches[0]
    try:
        if path.stat().st_size > MAX_METADATA_BYTES:
            issues.append("solar_metadata_exceeds_5_mib_limit")
            return None, None
        root = ET.fromstring(path.read_bytes())
    except (OSError, ET.ParseError):
        issues.append("unreadable_or_malformed_solar_metadata")
        return None, None
    groups = [element for element in root.iter() if _local_name(element.tag) == "Mean_Sun_Angle"]
    if len(groups) != 1:
        issues.append("missing_or_ambiguous_mean_sun_angle")
        return None, None
    # Zenith == 90 is deliberately unsupported: its tangent cannot define a
    # finite shadow-projection distance. Azimuth follows degrees clockwise
    # from north; no image-space direction is inferred here.
    zenith = _angle(groups[0], "ZENITH_ANGLE", issues, maximum=90)
    azimuth = _angle(groups[0], "AZIMUTH_ANGLE", issues, maximum=360)
    return azimuth, zenith


def qa_scene_metadata(scene_path, root_dir=None):
    """Return JSON-compatible metadata for one SAFE or its enclosing product.

    ``metadata_usable`` requires the existing seven-band L1C prerequisites,
    all ten cloud-model bands, valid mean solar angles, and the expected QA
    file paths for the union of cloud/feature bands. A missing QA path is an
    explicit issue, never an implicit all-clear mask. ``metadata_uid`` hashes
    metadata content and the discovered logical file names; it does not hash
    or verify satellite raster contents.
    """
    path = Path(scene_path).resolve()
    root_dir = Path(root_dir if root_dir is not None else Path.cwd()).resolve()
    issues = []
    if path.is_dir() and path.name.endswith(".SAFE"):
        safe, product = path, path.parent
    elif path.is_dir():
        matches = sorted(p for p in path.glob("*.SAFE") if p.is_dir())
        safe, product = (matches[0] if len(matches) == 1 else None), path
        if len(matches) > 1:
            issues.append("ambiguous_safe_directory")
    else:
        safe, product = None, path
        issues.append("scene_path_not_directory")

    record = _scene_record(product, safe, root_dir)
    issues = list(record["issues"]) + issues
    cloud_paths = {band: None for band in CLOUD_BANDS}
    ambiguous_cloud_bands = []
    quality_band_paths = {kind: {band: None for band in QUALITY_BANDS} for kind in QUALITY_TYPES}
    missing_quality = {kind: [] for kind in QUALITY_TYPES}
    quality_files = []
    if safe is not None:
        # Each glob is bounded to a SAFE granule layout; no recursive dataset
        # scan or JP2 read is involved.
        for band in CLOUD_BANDS:
            matches = sorted(safe.glob(f"GRANULE/*/IMG_DATA/*_{band}.jp2"))
            if len(matches) == 1:
                cloud_paths[band] = _path(matches[0], root_dir)
            else:
                prefix = "missing" if not matches else "ambiguous"
                issues.append(f"{prefix}_cloud_band_{band}")
                if matches:
                    ambiguous_cloud_bands.append(band)
        for kind in QUALITY_TYPES:
            for band in QUALITY_BANDS:
                matches = sorted(safe.glob(f"GRANULE/*/QI_DATA/MSK_{kind}_{band}.gml"))
                if len(matches) == 1:
                    quality_band_paths[kind][band] = _path(matches[0], root_dir)
                    quality_files.append(matches[0])
                    try:
                        if matches[0].stat().st_size > MAX_METADATA_BYTES:
                            issues.append(f"quality_mask_exceeds_5_mib_limit:{kind}:{band}")
                    except OSError:
                        issues.append(f"unreadable_quality_mask:{kind}:{band}")
                else:
                    missing_quality[kind].append(band)
                    prefix = "missing" if not matches else "ambiguous"
                    issues.append(f"{prefix}_quality_mask_{kind}_{band}")
    else:
        missing_quality = {kind: list(QUALITY_BANDS) for kind in QUALITY_TYPES}
    azimuth, zenith = _sun_angles(safe, issues)
    missing_cloud_bands = [band for band in CLOUD_BANDS if cloud_paths[band] is None]
    paths = {
        "cloud": record["cloud_qa_path"],
        **{kind.lower(): [p for p in quality_band_paths[kind].values() if p is not None]
           for kind in QUALITY_TYPES},
    }
    record.update(
        cloud_bands=list(CLOUD_BANDS), cloud_band_paths=cloud_paths,
        feature_bands=list(FEATURE_BANDS), feature_band_paths=dict(record["band_paths"]),
        missing_cloud_bands=missing_cloud_bands,
        ambiguous_cloud_bands=ambiguous_cloud_bands,
        solar_azimuth_deg=azimuth, solar_zenith_deg=zenith,
        solar_angle_units="degrees", shadow_geometry_usable=(azimuth is not None and zenith is not None and record["grid_10m"] is not None),
        quality_paths=paths, quality_band_paths=quality_band_paths,
        missing_quality_masks=missing_quality,
    )
    record["issues"] = sorted(set(issues))
    record["metadata_usable"] = not record["issues"]
    record["cloud_model_usable"] = bool(
        not missing_cloud_bands and not ambiguous_cloud_bands
        and record["quantification_value"] is not None
        and not record["nonzero_offset"]
        and record["processing_level"] == "L1C"
    )
    fingerprint = {
        "scene_id": record["scene_id"], "metadata_sha256": record["metadata_sha256"],
        "cloud_band_files": {band: Path(p).name if p else None for band, p in cloud_paths.items()},
        "quality_files": sorted(p.name for p in quality_files),
        "solar_azimuth_deg": azimuth, "solar_zenith_deg": zenith,
    }
    record["metadata_uid"] = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    record["metadata_uid_scope"] = "Product/tile/cloud-GML content and logical discovered filenames; excludes raster content"
    return record
