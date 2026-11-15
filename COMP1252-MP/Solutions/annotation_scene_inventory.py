"""Inventory local Sentinel SAFE metadata without opening raster pixels.

``metadata_usable`` means that a scene has the metadata and files required by
the project's L1C crop workflow. It is not a claim that a particular location
is clear, unchanged, or visually suitable for temporal training.
"""

from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


REQUIRED_BANDS = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")
MAX_METADATA_BYTES = 5 * 1024 * 1024
PRODUCT_PATTERN = re.compile(
    r"^(?P<satellite>S2[ABCD])_MSI(?P<level>L1C|L2A)_"
    r"(?P<acquisition>\d{8}T\d{6})_(?P<baseline>N\d{4})_"
    r"R\d{3}_T(?P<tile>\d{2}[A-Z]{3})_\d{8}T\d{6}$"
)


def _local_name(tag):
    return tag.rsplit("}", 1)[-1]


def _path_text(path, root_dir):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(root_dir))
    except ValueError:
        return str(path)


def _load_xml(path, label, issues, hashes):
    """Read only bounded-size XML; record file failures instead of dropping scenes."""
    try:
        if path.stat().st_size > MAX_METADATA_BYTES:
            issues.append(f"{label}_exceeds_5_mib_limit")
            return None
        data = path.read_bytes()
    except FileNotFoundError:
        issues.append(f"missing_{label}")
        return None
    except OSError as error:
        issues.append(f"unreadable_{label}:{type(error).__name__}")
        return None
    hashes[label] = hashlib.sha256(data).hexdigest()
    try:
        return ET.fromstring(data)
    except ET.ParseError:
        issues.append(f"malformed_{label}")
        return None


def _texts(root, name):
    if root is None:
        return []
    return [(element.text or "").strip() for element in root.iter()
            if _local_name(element.tag) == name]


def _one_number(texts, label, issues, positive=False, percentage=False):
    if len(texts) != 1:
        issues.append(f"missing_or_ambiguous_{label}")
        return None
    try:
        number = float(texts[0])
    except (TypeError, ValueError):
        issues.append(f"invalid_{label}")
        return None
    if not math.isfinite(number) or (positive and number <= 0) or (percentage and not 0 <= number <= 100):
        issues.append(f"invalid_{label}")
        return None
    return number


def _one_path(matches, label, issues):
    matches = sorted(matches)
    if len(matches) == 1:
        return matches[0]
    issues.append(f"missing_{label}" if not matches else f"ambiguous_{label}")
    return None


def _grid(root, issues):
    """Read the 10 m grid from Tile_Geocoding using local tag names."""
    if root is None:
        return None, None
    crs_values = _texts(root, "HORIZONTAL_CS_CODE")
    crs = crs_values[0] if len(crs_values) == 1 else None
    if crs is None or not re.fullmatch(r"EPSG:\d+", crs):
        issues.append("missing_or_invalid_crs")
        crs = None
    sizes = [element for element in root.iter()
             if _local_name(element.tag) == "Size" and element.attrib.get("resolution") == "10"]
    positions = [element for element in root.iter()
                 if _local_name(element.tag) == "Geoposition" and element.attrib.get("resolution") == "10"]
    if len(sizes) != 1 or len(positions) != 1:
        issues.append("missing_or_ambiguous_10m_grid")
        return crs, None
    values = {}
    for name in ("ULX", "ULY", "XDIM", "YDIM"):
        values[name] = _one_number(_texts(positions[0], name), f"10m_{name.lower()}", issues)
    for name in ("NROWS", "NCOLS"):
        number = _one_number(_texts(sizes[0], name), f"10m_{name.lower()}", issues, positive=True)
        if number is not None and not number.is_integer():
            issues.append(f"invalid_10m_{name.lower()}")
            number = None
        values[name] = int(number) if number is not None else None
    if any(value is None for value in values.values()):
        return crs, None
    if values["XDIM"] != 10 or values["YDIM"] != -10:
        issues.append("unsupported_10m_grid_spacing_or_orientation")
        return crs, None
    ulx, uly = values["ULX"], values["ULY"]
    width, height = values["NCOLS"], values["NROWS"]
    return crs, {
        "ulx": ulx, "uly": uly, "xdim": values["XDIM"], "ydim": values["YDIM"],
        "width": width, "height": height,
        "transform": [10.0, 0.0, ulx, 0.0, -10.0, uly],
        "bounds": [ulx, uly - height * 10.0, ulx + width * 10.0, uly],
    }


def _scene_record(product, safe, root_dir):
    name = (safe.name if safe is not None else product.name).removesuffix(".SAFE")
    issues, hashes = [], {}
    record = {
        "scene_id": name, "product_name": product.name.removesuffix(".SAFE"),
        "satellite": None, "processing_level": None, "tile": None,
        "acquisition_date": None, "acquisition_datetime": None,
        "baseline_code": None, "processing_baseline": None,
        "path": _path_text(safe if safe is not None else product, root_dir),
        "product_path": _path_text(product, root_dir),
        "band_paths": {band: None for band in REQUIRED_BANDS},
        "missing_bands": list(REQUIRED_BANDS), "ambiguous_bands": [],
        "cloud_qa_path": None, "cloud_qa_missing": True,
        "cloud_qa_well_formed": False, "cloud_cover_percent": None,
        "cloud_cover_source": None, "product_cloud_cover_percent": None,
        "tile_cloud_cover_percent": None, "quantification_value": None,
        "radiometric_offsets": {}, "nonzero_offset": False,
        "crs": None, "grid_10m": None, "metadata_sha256": hashes,
        "metadata_usable": False, "issues": issues,
    }
    match = PRODUCT_PATTERN.fullmatch(name)
    if match:
        record.update(satellite=match["satellite"], processing_level=match["level"],
                      tile=match["tile"], baseline_code=match["baseline"])
        try:
            acquired = datetime.strptime(match["acquisition"], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            record["acquisition_date"] = acquired.date().isoformat()
            record["acquisition_datetime"] = acquired.isoformat().replace("+00:00", "Z")
        except ValueError:
            issues.append("invalid_acquisition_datetime")
        if match["level"] != "L1C":
            issues.append("unsupported_processing_level")
    else:
        issues.append("unrecognized_product_name")
    if safe is None:
        issues.append("missing_safe_directory")
        return record

    product_root = _load_xml(safe / "MTD_MSIL1C.xml", "product_metadata", issues, hashes)
    tile_path = _one_path(safe.glob("GRANULE/*/MTD_TL.xml"), "tile_metadata", issues)
    tile_root = _load_xml(tile_path, "tile_metadata", issues, hashes) if tile_path else None
    record["crs"], record["grid_10m"] = _grid(tile_root, issues)
    if product_root is not None:
        baseline = _texts(product_root, "PROCESSING_BASELINE")
        if len(baseline) == 1:
            record["processing_baseline"] = baseline[0]
        else:
            issues.append("missing_or_ambiguous_processing_baseline")
        record["quantification_value"] = _one_number(
            _texts(product_root, "QUANTIFICATION_VALUE"), "quantification_value", issues, positive=True,
        )
        for element in product_root.iter():
            if _local_name(element.tag) == "RADIO_ADD_OFFSET":
                band_id = element.attrib.get("band_id", element.attrib.get("bandId", "unknown"))
                value = _one_number([element.text], f"radiometric_offset_{band_id}", issues)
                if band_id in record["radiometric_offsets"]:
                    issues.append(f"duplicate_radiometric_offset_{band_id}")
                record["radiometric_offsets"][band_id] = value
        record["nonzero_offset"] = any(value not in (None, 0) for value in record["radiometric_offsets"].values())
        if record["nonzero_offset"]:
            issues.append("nonzero_radiometric_offset_unsupported")

    # Prefer tile-level coverage. Neither percentage establishes crop quality.
    product_coverage = _texts(product_root, "Cloud_Coverage_Assessment")
    tile_coverage = _texts(tile_root, "CLOUDY_PIXEL_PERCENTAGE")
    if product_coverage:
        record["product_cloud_cover_percent"] = _one_number(product_coverage, "product_cloud_cover", issues, percentage=True)
    if tile_coverage:
        record["tile_cloud_cover_percent"] = _one_number(tile_coverage, "tile_cloud_cover", issues, percentage=True)
    if record["tile_cloud_cover_percent"] is not None:
        record["cloud_cover_percent"] = record["tile_cloud_cover_percent"]
        record["cloud_cover_source"] = "tile_metadata"
    elif record["product_cloud_cover_percent"] is not None:
        record["cloud_cover_percent"] = record["product_cloud_cover_percent"]
        record["cloud_cover_source"] = "product_metadata"
    else:
        issues.append("missing_cloud_cover_percent")

    for band in REQUIRED_BANDS:
        matches = sorted(safe.glob(f"GRANULE/*/IMG_DATA/*_{band}.jp2"))
        if len(matches) > 1:
            record["ambiguous_bands"].append(band)
        path = _one_path(matches, f"band_{band}", issues)
        if path is not None:
            record["band_paths"][band] = _path_text(path, root_dir)
    record["missing_bands"] = [band for band in REQUIRED_BANDS if record["band_paths"][band] is None]
    cloud_path = _one_path(safe.glob("GRANULE/*/QI_DATA/MSK_CLOUDS_B00.gml"), "cloud_qa", issues)
    record["cloud_qa_missing"] = cloud_path is None
    if cloud_path is not None:
        record["cloud_qa_path"] = _path_text(cloud_path, root_dir)
        cloud_root = _load_xml(cloud_path, "cloud_qa", issues, hashes)
        if cloud_root is not None:
            if _local_name(cloud_root.tag) != "Mask":
                issues.append("unrecognized_cloud_qa_root")
            else:
                record["cloud_qa_well_formed"] = True
    record["metadata_usable"] = not issues
    return record


def inventory_scenes(sentinel_dir, root_dir=None):
    """Return one JSON-serializable metadata record per local SAFE scene.

    Directory discovery is bounded to the usual ``product/*.SAFE`` layout or
    direct ``*.SAFE`` entries. Missing/ambiguous/malformed files produce flagged
    records, preserving their scene identity for an audit. No raster library is
    imported, and no JP2 contents are opened. Paths are relative to ``root_dir``
    when possible (the current working directory by default).
    """
    sentinel_dir = Path(sentinel_dir)
    root_dir = Path(root_dir if root_dir is not None else Path.cwd()).resolve()
    if not sentinel_dir.is_dir():
        raise FileNotFoundError(f"Sentinel directory does not exist: {sentinel_dir}")
    records = []
    for product in sorted(sentinel_dir.iterdir()):
        if not product.is_dir():
            continue
        if product.name.endswith(".SAFE"):
            records.append(_scene_record(product, product, root_dir))
            continue
        safes = sorted(path for path in product.glob("*.SAFE") if path.is_dir())
        if not safes:
            if product.name.startswith("S2"):
                records.append(_scene_record(product, None, root_dir))
            continue
        for safe in safes:
            record = _scene_record(product, safe, root_dir)
            if len(safes) > 1:
                record["issues"].append("multiple_safe_directories_in_product")
                record["metadata_usable"] = False
            records.append(record)
    return sorted(records, key=lambda record: (record["tile"] or "", record["acquisition_datetime"] or "", record["scene_id"], record["path"]))
