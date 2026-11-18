"""Source metadata/transfer regressions without satellite content or network."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
import preflight_earlier_validation_metadata as preflight


class Response(io.BytesIO):
    def __init__(self, url, raw, headers):
        super().__init__(raw); self.url=url; self.headers=headers; self.status=200

    def geturl(self):
        return self.url


class EarlierMetadataPreflightTests(unittest.TestCase):
    def setUp(self):
        self.protocol=json.loads((preflight.ROOT/'Docs/Reviews/ValidationEarlierMetadataV13/protocol.json').read_text())
        self.roster=self.protocol['selection']['roster']; self.selected=self.roster[0]
        self.requirements=json.loads((preflight.ROOT/'Docs/Reviews/ValidationCoverageV6/source_requirements.json').read_text())
        self.folder='L1C_T36UXB_A016077_20180721T084234'
        self.tile_id='S2A_OPER_MSI_L1C_TL_MTI__20180721T105541_A016077_T36UXB_N02.06'
        self.feature={'properties':{'s2:granule_id':self.tile_id,'s2:datastrip_id':'datastrip',
                                   'datetime':'2018-07-21T08:42:34.710Z'},'assets':{}}
        images=''.join(f'<IMAGE_FILE>GRANULE/{self.folder}/IMG_DATA/T36UXB_20180721T083601_{b}</IMAGE_FILE>' for b in preflight.BANDS)
        self.product=f'''<Level-1C_User_Product><PRODUCT_URI>{self.selected['scene_id']}.SAFE</PRODUCT_URI>
        <PROCESSING_LEVEL>Level-1C</PROCESSING_LEVEL><PROCESSING_BASELINE>02.06</PROCESSING_BASELINE>
        <SPACECRAFT_NAME>Sentinel-2A</SPACECRAFT_NAME><SENSING_ORBIT_NUMBER>64</SENSING_ORBIT_NUMBER>
        <QUANTIFICATION_VALUE>10000</QUANTIFICATION_VALUE><PRODUCT_START_TIME>2018-07-21T08:36:01.024Z</PRODUCT_START_TIME>
        <Granule granuleIdentifier="{self.tile_id}" datastripIdentifier="datastrip">{images}</Granule></Level-1C_User_Product>'''.encode()
        self.tile=f'''<Level-1C_Tile_ID><TILE_ID>{self.tile_id}</TILE_ID><DATASTRIP_ID>datastrip</DATASTRIP_ID>
        <SENSING_TIME>2018-07-21T08:42:34.710Z</SENSING_TIME><HORIZONTAL_CS_CODE>EPSG:32636</HORIZONTAL_CS_CODE>
        <Size resolution="10"><NROWS>10980</NROWS><NCOLS>10980</NCOLS></Size>
        <Geoposition resolution="10"><ULX>600000</ULX><ULY>5700000</ULY><XDIM>10</XDIM><YDIM>-10</YDIM></Geoposition>
        <Mean_Sun_Angle><ZENITH_ANGLE unit="deg">32.6</ZENITH_ANGLE><AZIMUTH_ANGLE unit="deg">153.5</AZIMUTH_ANGLE></Mean_Sun_Angle>
        </Level-1C_Tile_ID>'''.encode()

    def parse(self, product=None, tile=None, feature=None):
        return preflight.parse_metadata(product or self.product,tile or self.tile,self.selected,
                                        feature or self.feature,self.requirements)

    def test_actual_xml_identity_grid_and_solar_metadata_pass_without_assets(self):
        result=self.parse()
        self.assertEqual(result['issues'],[])
        self.assertEqual(result['quantification_value'],10000)
        self.assertEqual(result['radio_add_offsets'],{})
        self.assertFalse(result['xml_offset_elements_present'])
        self.assertTrue(all(all(c.values()) for c in result['fixed_event_coverage'].values()))
        self.assertEqual(result['granule_folder'],self.folder)

    def test_xml_identity_offsets_and_granule_conflicts_cannot_pass(self):
        changes=[(b'Level-1C</PROCESSING_LEVEL>',b'Level-2A</PROCESSING_LEVEL>','product_identity_conflict:PROCESSING_LEVEL'),
                 (b'<SENSING_ORBIT_NUMBER>64',b'<SENSING_ORBIT_NUMBER>107','orbit_or_quantification_incompatible'),
                 (b'02.06</PROCESSING_BASELINE>',b'05.00</PROCESSING_BASELINE>','product_identity_conflict:PROCESSING_BASELINE'),
                 (b'20180721T083601_B02',b'20180721T084234_B02','image_file_acquisition_name_conflict'),
                 (self.folder.encode(),b'L1C_T36UXB_A999999_20180721T084234','granule_folder_tile_identity_conflict')]
        for before,after,issue in changes:
            with self.subTest(issue=issue): self.assertIn(issue,self.parse(product=self.product.replace(before,after))['issues'])
        for value in ('-1000','NaN'):
            product=self.product.replace(b'</Level-1C_User_Product>',f'<RADIO_ADD_OFFSET band_id="0">{value}</RADIO_ADD_OFFSET></Level-1C_User_Product>'.encode())
            result=self.parse(product=product)
            self.assertIn('invalid_duplicate_or_nonzero_xml_offset',result['issues'])
            json.dumps(result,allow_nan=False)
        feature=deepcopy(self.feature);feature['properties']['s2:granule_id']='other'
        self.assertIn('product_tile_catalogue_granule_or_datastrip_conflict',self.parse(feature=feature)['issues'])

    def test_actual_grid_context_solar_and_sensing_failures_are_explicit(self):
        tile=self.tile.replace(b'<ULY>5700000',b'<ULY>5600000')
        self.assertIn('actual_grid_missing_full_qa_context',self.parse(tile=tile)['issues'])
        tile=self.tile.replace(b'>32.6<',b'>90<')
        self.assertIn('invalid_solar_zenith',self.parse(tile=tile)['issues'])
        tile=self.tile.replace(b'2018-07-21T08:42:34.710Z',b'2018-07-21T08:42:35.710Z')
        self.assertIn('tile_catalogue_sensing_time_conflict',self.parse(tile=tile)['issues'])

    def complete_listing(self):
        metadata=self.parse(); prefix=self.selected['tile_prefix']; result={}
        names=['metadata.xml','qi/MSK_CLOUDS_B00.gml']+[b+'.jp2' for b in preflight.BANDS]
        names += [f'qi/MSK_{kind}_{band}.gml' for band in preflight.BANDS for kind in preflight.KINDS]
        for name in names:
            key=prefix+name;result[key]={'key':key,'url':preflight.BASE+key,'size':123,'etag':'"'+'1'*32+'"'}
        return metadata,result

    def test_all_58_assets_are_pinned_and_missing_quality_is_not_synthesized(self):
        metadata,listed=self.complete_listing(); product_pin={'size':456,'etag':'"'+'2'*32+'"'}
        assets,issues=preflight.asset_manifest(self.selected,metadata,listed,product_pin)
        self.assertEqual((len(assets),issues),(58,[]))
        self.assertEqual(sum(a['role']=='jp2' for a in assets),11)
        self.assertEqual(sum(a['role']=='quality_gml' for a in assets),44)
        key=self.selected['tile_prefix']+'qi/MSK_TECQUA_B09.gml';del listed[key]
        assets,issues=preflight.asset_manifest(self.selected,metadata,listed,product_pin)
        self.assertEqual(len(assets),57);self.assertEqual(issues,['missing_required_object:'+key])
        listed[key]={'key':key,'url':preflight.BASE+key,'size':123,'etag':'"multipart-2"'}
        self.assertTrue(preflight.asset_manifest(self.selected,metadata,listed,product_pin)[1])

    def test_truncated_duplicate_or_wrong_prefix_listing_is_refused(self):
        prefix=self.selected['tile_prefix'];key=prefix+'metadata.xml'
        entry=f'<Contents><Key>{key}</Key><Size>123</Size><ETag>"'+('1'*32)+'"</ETag></Contents>'
        raw=f'<ListBucketResult><Prefix>{prefix}</Prefix><IsTruncated>false</IsTruncated><KeyCount>1</KeyCount>{entry}</ListBucketResult>'.encode()
        self.assertEqual(len(preflight.parse_listing(raw,prefix,300)),1)
        for changed in (raw.replace(b'>false<',b'>true<'),raw.replace(prefix.encode(),b'other/',1),
                        raw.replace(b'<KeyCount>1',b'<KeyCount>2').replace(b'</ListBucketResult>',entry.encode()+b'</ListBucketResult>')):
            with self.subTest(raw=changed),self.assertRaises(ValueError):preflight.parse_listing(changed,prefix,300)

    def test_network_allows_only_metadata_and_verifies_exact_xml_transfer(self):
        url=preflight.BASE+self.selected['product_key'];payload=b'<actual_xml/>'
        asset={'size':len(payload),'etag':'"'+hashlib.md5(payload).hexdigest()+'"'}
        calls=[]
        def opener(request,**kwargs):
            calls.append(request);return Response(request.full_url,payload,{'Content-Length':str(len(payload)),'ETag':asset['etag']})
        with tempfile.TemporaryDirectory() as directory:
            net=preflight.MetadataNetwork(self.protocol['provider'],self.roster,Path(directory),opener)
            head=net.call('HEAD',url,None);self.assertEqual(head,asset)
            self.assertEqual(net.call('GET',url,'product.xml',asset),payload)
            self.assertEqual(calls[-1].get_header('If-match'),asset['etag'])
            self.assertTrue(net.requests[-1]['verified'])
            for forbidden in (preflight.BASE+self.selected['tile_prefix']+'B02.jp2',
                              preflight.BASE+self.selected['tile_prefix']+'qi/MSK_CLOUDS_B00.gml',
                              'https://sentinel-s2-l2a.s3.amazonaws.com/'+self.selected['product_key']):
                with self.assertRaisesRegex(ValueError,'allowlist'):net.call('GET',forbidden,'forbidden')

    def test_xml_md5_etag_length_and_network_budgets_fail_closed(self):
        url=preflight.BASE+self.selected['product_key'];payload=b'xml'
        asset={'size':3,'etag':'"'+hashlib.md5(payload).hexdigest()+'"'}
        cases=[(b'bad',{'Content-Length':'3','ETag':asset['etag']}),
               (payload,{'Content-Length':'3','ETag':'"'+'0'*32+'"'}),
               (payload,{'Content-Length':'2','ETag':asset['etag']})]
        for raw,headers in cases:
            with self.subTest(headers=headers),tempfile.TemporaryDirectory() as directory:
                net=preflight.MetadataNetwork(self.protocol['provider'],self.roster,Path(directory),lambda *_a,**_k:Response(url,raw,headers))
                with self.assertRaisesRegex(ValueError,'mismatch'):net.call('GET',url,'bad.xml',asset)
                self.assertIn('error',net.requests[0])
        with tempfile.TemporaryDirectory() as directory:
            provider={**self.protocol['provider'],'maximum_total_response_bytes':2}
            net=preflight.MetadataNetwork(provider,self.roster,Path(directory),lambda *_a,**_k:Response(url,payload,{}))
            with self.assertRaisesRegex(ValueError,'byte cap'):net.call('GET',url,'too_big.xml',asset)
            net.provider={**provider,'maximum_requests':1}
            with self.assertRaisesRegex(ValueError,'count bound'):net.call('HEAD',url,None)

    def test_conservative_disk_budget_never_credits_incomplete_sources(self):
        scenes=[{'assets':[{'size':10} for _ in range(58)]} for _ in range(2)]
        expected=2*1160+10+100;reserve=2*2**30
        result=preflight.disk_budget(scenes,expected+reserve,100)
        self.assertEqual(result['complete_payload_bytes'],1160)
        self.assertEqual(result['future_two_copy_staging_plus_temporary_bytes'],expected)
        self.assertTrue(result['sufficient_for_conservative_future_acquisition'])
        self.assertFalse(preflight.disk_budget(scenes,expected+reserve-1,100)['sufficient_for_conservative_future_acquisition'])
        scenes[0]['assets'].pop()
        self.assertIsNone(preflight.disk_budget(scenes,10**12,100)['complete_payload_bytes'])

    def test_first_source_timeout_retains_both_fixed_failed_outcomes_without_fallback(self):
        raw={r['item_id']:{'assets':{}} for r in self.roster}
        fake=SimpleNamespace(requests=[],bytes=0,call=Mock(side_effect=[TimeoutError('first timed out'),URLError('second denied')]))
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);args=SimpleNamespace(protocol=preflight.ROOT/'Docs/Reviews/ValidationEarlierMetadataV13/protocol.json')
            with patch.object(preflight,'inputs',return_value=(self.roster,raw,self.requirements,{})),\
                 patch.object(preflight,'MetadataNetwork',return_value=fake),\
                 patch.object(preflight,'sys',SimpleNamespace(platform=sys.platform,modules={})):
                preflight.worker(args,output)
            result=json.loads((output/'preflight.json').read_text())
            self.assertEqual([r['acquisition_date'] for r in result['results']],list(preflight.DATES))
            self.assertIn('TimeoutError',result['results'][0]['issues'][0])
            self.assertIn('URLError',result['results'][1]['issues'][0])
            self.assertEqual(fake.call.call_count,2)
            self.assertEqual(result['status'],'metadata_preflight_blocked_no_acquisition')


if __name__=='__main__':unittest.main()
