import os
import logging
from sentinelhub import SHConfig, BBox, CRS, MimeType, SentinelHubRequest, DataCollection, bbox_to_dimensions
from extractor import *

# Logging setup
logging.basicConfig(level=logging.INFO)

# Load the bounding box from the GeoJSON
geojson_path = '../Datasets/bbox.geojson'
bbox_coordinates, code, name = extract_bbox(geojson_path)

logging.info(f"Bounding box extracted: {bbox_coordinates}, Code: {code}, Name: {name}")

# Sentinel Hub configuration (set your instance ID, client ID, and secret)
config = SHConfig()

config.instance_id = 'your-instance-id'
config.sh_client_id = 'your-client-id'
config.sh_client_secret = 'your-client-secret'

if not config.sh_client_id or not config.sh_client_secret:
    raise ValueError("Missing Sentinel Hub credentials!")

# Create the bounding box object
bbox = BBox(bbox=bbox_coordinates, crs=CRS.WGS84)

# Define the resolution and size of the image
resolution = 10  # meters per pixel
width, height = bbox_to_dimensions(bbox, resolution=resolution)

# Define the evalscript for Sentinel-2 imagery (RGB bands)
evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: ["B04", "B03", "B02"],
            output: { bands: 3 }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""

# Create a SentinelHubRequest for the satellite imagery
request = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=('2008-01-01', '2024-01-01'),
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.TIFF)  # or use MimeType.JPEG for JPEG format
    ],
    bbox=bbox,
    size=(width, height),
    config=config
)

# Fetch the data
response = request.get_data(save_data=True, data_folder='downloaded_data')

logging.info("Data downloaded successfully!")