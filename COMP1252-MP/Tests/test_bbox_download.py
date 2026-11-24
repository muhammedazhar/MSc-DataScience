"""Bounding-box download stages against archived metadata and local payloads."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Solutions'))
import download
from bbox_sentinel_preflight import PROVIDER, catalogue_source
from inventory_earlier_validation_sources import ENDPOINT, digest, write
from preflight_earlier_validation_metadata import BANDS, BASE, KINDS, listing_url


ITEM_ID = 'S2B_36UYA_20190303_0_L1C'
PLOT = {'code': 'PLOT-A', 'status': 'Converted', 'bbox': [36.13, 49.70, 36.14, 49.71]}


class Response(BytesIO):
    status = 200

    def __init__(self, url, body, headers=None):
        super().__init__(body)
        self.url = url
        self.headers = headers or {}

    def geturl(self):
        return self.url


class BoundingBoxDownloadTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.workspace = Path(directory.name)
        disk = patch('bbox_sentinel_transfer.shutil.disk_usage', return_value=SimpleNamespace(free=8 * 2**30))
        disk.start()
        self.addCleanup(disk.stop)
        reviews = ROOT/'Docs/Reviews'
        self.page = (reviews/'Validation2019SourcesV25/sentinel/march/raw/page-01.json').read_bytes()
        self.feature = next(feature for feature in json.loads(self.page)['features'] if feature['id'] == ITEM_ID)
        self.source = catalogue_source(self.feature)
        xml = reviews/'Validation2019PreflightV26/preflight/raw'/ITEM_ID
        prefix = self.source['tile_prefix']
        self.tile_key = prefix+'metadata.xml'
        self.product_key = self.source['product_key']
        self.payloads = {self.product_key: (xml/'product.xml').read_bytes(),
                         self.tile_key: (xml/'tile.xml').read_bytes(),
                         prefix+'qi/MSK_CLOUDS_B00.gml': b'<Mask/>'}
        for band in BANDS:
            self.payloads[prefix+band+'.jp2'] = band.encode()
            for kind in KINDS:
                self.payloads[f'{prefix}qi/MSK_{kind}_{band}.gml'] = b'<Mask/>'
        self.pins = {key: self.pin(body) for key, body in self.payloads.items()}
        self.plan_dir = self.workspace/'plan'
        self.preflight_dir = self.workspace/'preflight'
        self.destination = self.workspace/'Sentinel-2'

    @staticmethod
    def pin(body):
        return {'ETag': f'"{hashlib.md5(body).hexdigest()}"', 'Content-Length': str(len(body))}

    def cli(self, *arguments):
        # The test suite already runs under the guard. Exercise argument parsing
        # and the real stage in-process so the HTTP stubs stay in scope.
        def guarded(args, **_options):
            args.output.mkdir()
            download.worker(args, args.output)

        with patch.object(sys, 'argv', ['download.py', *map(str, arguments)]), patch.dict(os.environ), \
                patch.object(download, 'run_guarded', side_effect=guarded), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return download.main()

    def plan(self):
        west, south, east, north = PLOT['bbox']
        ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
        geometry = {'type': 'Polygon', 'coordinates': [ring]}
        bbox_file, search_file = (self.workspace/name for name in ('bbox.geojson', 'search.json'))
        write(bbox_file, {'type': 'FeatureCollection', 'features': [
            {'type': 'Feature', 'properties': {'code': code, 'status': PLOT['status']}, 'geometry': geometry}
            for code in ('PLOT-A', 'PLOT-SKIP')]})
        write(search_file, {'collection': 'sentinel-2-l1c', 'start': '2019-03-03', 'end': '2019-03-03',
                            'count': 1, 'cloud_cover': 20})

        def serve(request, timeout):
            self.assertEqual((request.get_method(), request.full_url), ('POST', ENDPOINT))
            self.assertEqual(json.loads(request.data)['bbox'], PLOT['bbox'])
            return Response(ENDPOINT, self.page)

        with patch('bbox_sentinel_catalogue.build_opener') as opener:
            opener.return_value.open.side_effect = serve
            status = self.cli('plan', '--plot', PLOT['code'], '--bbox-file', bbox_file, '--search-file',
                              search_file, '--output', self.plan_dir)
            self.assertEqual(status, 0)
            opener.return_value.open.assert_called_once()
        return download.load(self.plan_dir/'plan.json')

    def listing(self):
        root = ET.Element('ListBucketResult')
        prefix = self.source['tile_prefix']
        ET.SubElement(root, 'IsTruncated').text = 'false'
        ET.SubElement(root, 'Prefix').text = prefix
        ET.SubElement(root, 'KeyCount').text = str(len(self.payloads)-1)
        for key, pin in self.pins.items():
            if key.startswith(prefix):
                entry = ET.SubElement(root, 'Contents')
                ET.SubElement(entry, 'Key').text = key
                ET.SubElement(entry, 'ETag').text = pin['ETag']
                ET.SubElement(entry, 'Size').text = pin['Content-Length']
        return ET.tostring(root)

    def preflight(self):
        self.plan()
        requests = []

        def serve(request, timeout):
            method, url = request.get_method(), request.full_url
            requests.append((method, url))
            if url == listing_url(self.source, PROVIDER):
                self.assertEqual(method, 'GET')
                return Response(url, self.listing())
            key = url.removeprefix(BASE)
            self.assertIn(key, (self.product_key, self.tile_key))
            if method == 'HEAD':
                self.assertEqual(key, self.product_key)
                return Response(url, b'', self.pins[key])
            return self.serve_payload(request, timeout)

        with patch('preflight_earlier_validation_metadata.build_opener') as opener:
            opener.return_value.open.side_effect = serve
            status = self.cli('preflight', '--plan', self.plan_dir, '--scene', ITEM_ID,
                              '--output', self.preflight_dir)
            report = download.load(self.preflight_dir/'preflight.json')
            self.assertEqual(status, 0, report['issues'])
        self.assertEqual(len(requests), 4)
        self.assertEqual(report['plan_sha256'], digest(self.plan_dir/'plan.json'))
        self.assertEqual(report['plots'], [PLOT])
        self.assertEqual(len(report['assets']), 58)
        self.assertEqual(report['metadata']['plot_coverage'][PLOT['code']],
                         {'catalogue_footprint': True, 'native_grid': True})
        self.assertNotEqual(report['metadata']['catalogue_minus_native_seconds'], 0)
        return report

    def serve_payload(self, request, timeout):
        url = request.full_url
        self.assertEqual(request.get_method(), 'GET')
        self.assertTrue(url.startswith(BASE))
        key = url.removeprefix(BASE)
        request_headers = {name.lower(): value for name, value in request.header_items()}
        self.assertEqual(request_headers['if-match'], self.pins[key]['ETag'])
        return Response(url, self.payloads[key], self.pins[key])

    def test_plan_pins_the_named_plot_and_exact_scene(self):
        plan = self.plan()
        self.assertEqual([plot['code'] for plot in plan['plots']], ['PLOT-A'])
        self.assertEqual(download.find_scene(plan, ITEM_ID), (self.feature, [PLOT]))
        self.assertEqual(download.find_scene(plan, self.source['scene_id']), (self.feature, [PLOT]))
        with self.assertRaisesRegex(ValueError, 'not found'):
            download.find_scene(plan, '36UYA')

    def test_download_publishes_verified_scene_and_repeats_without_network(self):
        preflight = self.preflight()
        output = self.workspace/'download'
        with patch('acquire_earlier_validation_sources.build_opener') as opener:
            opener.return_value.open.side_effect = self.serve_payload
            self.assertEqual(self.cli('download', '--preflight', self.preflight_dir, '--destination',
                                      self.destination, '--output', output), 0)
            self.assertEqual(opener.return_value.open.call_count, 58)
        report = download.load(output/'download.json')
        self.assertEqual(report['new_downloads'], 58)
        self.assertEqual(report['transaction_state'], 'complete')
        self.assertTrue(report['final_metadata']['metadata_usable'])
        final = self.destination/self.source['scene_id']
        for asset in preflight['assets']:
            relative = asset['target_relpath']
            body = self.payloads[asset['key']]
            self.assertEqual((final/relative).read_bytes(), body)
            self.assertEqual(report['source_receipts'][relative]['sha256'], hashlib.sha256(body).hexdigest())

        repeat = self.workspace/'repeat'
        with patch('acquire_earlier_validation_sources.build_opener') as opener:
            self.assertEqual(self.cli('download', '--preflight', self.preflight_dir, '--destination',
                                      self.destination, '--output', repeat), 0)
            opener.assert_not_called()
        self.assertEqual(download.load(repeat/'download.json')['new_downloads'], 0)

    def test_interrupted_transfer_resumes_with_verified_files(self):
        preflight = self.preflight()
        fail_key = preflight['assets'][3]['key']
        attempts = []

        def interrupt(request, timeout):
            key = request.full_url.removeprefix(BASE)
            attempts.append(key)
            if key == fail_key:
                raise OSError('interrupted fixture transfer')
            return self.serve_payload(request, timeout)

        with patch('acquire_earlier_validation_sources.build_opener') as opener:
            opener.return_value.open.side_effect = interrupt
            with self.assertRaises(SystemExit) as stopped:
                self.cli('download', '--preflight', self.preflight_dir, '--destination', self.destination,
                         '--output', self.workspace/'interrupted')
            self.assertEqual(stopped.exception.code, 1)
        self.assertEqual(attempts, [asset['key'] for asset in preflight['assets'][:4]])
        self.assertFalse((self.destination/self.source['scene_id']).exists())

        output = self.workspace/'resumed'
        with patch('acquire_earlier_validation_sources.build_opener') as opener:
            opener.return_value.open.side_effect = self.serve_payload
            self.assertEqual(self.cli('download', '--preflight', self.preflight_dir, '--destination',
                                      self.destination, '--output', output), 0)
            self.assertEqual(opener.return_value.open.call_count, 55)
        self.assertEqual(download.load(output/'download.json')['new_downloads'], 55)


if __name__ == '__main__':
    unittest.main()
