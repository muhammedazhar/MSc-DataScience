#!/usr/bin/env python3
"""Re-render the original dated Landsat views for the new fixed site-09 square."""

import argparse
import os
from pathlib import Path
import resource
import sys
from tempfile import TemporaryDirectory

from inventory_unseen_validation_sites import ROOT, digest, load, require, write
from preflight_fresh_validation_sources import checked_artifacts, evidence_hashes
from prepare_stage1_candidate_review import build_packet


REPORT = ROOT/'Docs/Reviews/ValidationStage1SearchV35/candidates/site-09'
ORIGINAL = ROOT/'Datasets/ValidationPilot/v35/candidates/site-09/packet'
LOCAL = ROOT/'Datasets/ValidationPilot/v36/site09-alternate-optical'
PROTOCOL = REPORT/'alternate-optical-v36/site-09.json'


def prepare(output):
    retained = load(REPORT/'packet.json')
    protocol = load(PROTOCOL)
    require(protocol['sources'] == load(REPORT.parent.parent/'candidate_protocols/site-09.json')['sources']
            and protocol['control'] == load(REPORT/'radar-alternate-v36.json')['selected_control'],
            'The alternate board changes the dates or selected geometry')
    for receipt in retained['receipts']:
        original = ORIGINAL/receipt['filename']
        require(original.is_file() and original.stat().st_size == receipt['bytes']
                and digest(original) == receipt['sha256'], 'Retained Landsat source bytes changed')
    build_packet(output, PROTOCOL, ORIGINAL)
    return evidence_hashes(output)


def worker(args, output):
    checksum = digest(PROTOCOL)
    if args.packet:
        prepared = args.packet.resolve()
        require(prepared.is_relative_to(LOCAL), 'Review replay must use the local V36 packet')
        prior = load(prepared/'metadata.json')
        require(prior['execution_status'] == 'completed' and prior['protocol_sha256'] == checksum,
                'Invalid alternate optical review packet')
        checked_artifacts(prepared, prior['artifact_sha256'])
        with TemporaryDirectory(dir=output) as temporary:
            hashes = prepare(Path(temporary))
        require(hashes == prior['artifact_sha256'], 'Alternate optical review replay differs')
        result = {'verified_artifact_sha256': hashes, 'replay_verified': True}
    else:
        result = {'artifact_sha256': prepare(output)}
    require(digest(PROTOCOL) == checksum and 'torch' not in sys.modules, 'Protocol changed or model runtime imported')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {'schema_version': 36, 'execution_status': 'completed',
          'protocol_sha256': checksum, 'image_requests': 0, 'model_executions': 0, 'labels_approved': 0,
          'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2), **result})


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=512, min_available_gib=1.5, max_seconds=120)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(LOCAL):
        parser.error('Use the local V36 optical review output area')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='site09-alternate-optical',
                    gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
