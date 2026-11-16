#!/usr/bin/env python3
"""Run a CPU import probe or tests under the unchanged Mac resource guard."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def configure_cpu(seed=42):
    """Called in a fresh guarded worker before any tensor/model operations."""
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return torch


def worker(args, output):
    os.nice(10)
    torch = configure_cpu()
    probe = torch.zeros(1, device='cpu')
    metadata = {
        'mode': args.mode, 'device': str(probe.device),
        'torch_version': str(torch.__version__),
        'torch_intraop_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'native_threads': 1, 'dataloader_workers': 0, 'seed': 42,
        'gpu_used': False,
        'environment_versions': {name: importlib.metadata.version(name)
                                 for name in ('torch', 'numpy', 'rasterio')},
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if args.mode == 'tests':
        import unittest

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            suite = unittest.defaultTestLoader.discover(str(ROOT/'Tests'), pattern=args.pattern)
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        log = stream.getvalue()
        (output/'test_output.txt').write_text(log)
        print(log, flush=True)
        metadata.update(pattern=args.pattern, tests_run=result.testsRun,
                        failures=len(result.failures), errors=len(result.errors),
                        skipped=len(result.skipped), successful=result.wasSuccessful())
        if not result.wasSuccessful():
            raise RuntimeError('Tests failed; guarded output will not be published')
    elif args.mode == 'audit':
        from pilot_dataset import ReviewedPilotDataset

        metadata['dataset_audit'] = ReviewedPilotDataset(args.dataset).audit()
    else:
        metadata['probe_passed'] = (str(probe.device) == 'cpu'
                                    and torch.get_num_threads() == torch.get_num_interop_threads() == 1)
        if not metadata['probe_passed']:
            raise RuntimeError('CPU/thread probe did not pass')
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metadata['worker_high_water_rss_mib'] = round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2)
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('probe', 'tests', 'audit'), default='probe')
    parser.add_argument('--pattern', default='test_*.py')
    parser.add_argument('--dataset', type=Path, default=ROOT/'Datasets/CuratedPilot/v1/dataset')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=120)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Use 128–512 MiB RSS, >=1.5 GiB reserve and <=300 seconds')
    if '/' in args.pattern or '\\' in args.pattern:
        parser.error('The test pattern must be a filename pattern within Tests')
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    if args.worker_output:
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='pilot-check', gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
