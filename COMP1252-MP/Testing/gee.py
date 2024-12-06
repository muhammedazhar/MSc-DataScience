# Code for downloading satellite images from Google Earth Engine (GEE) using the Python API.

import os
from dotenv import load_dotenv
import ee
import requests
import zipfile
import io

# Load environment variables from the .env file
load_dotenv('../Solutions/.env')

# Set up authentication using the service account credentials
service_account_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
credentials = ee.ServiceAccountCredentials(None, service_account_file)

# Initialize the Earth Engine API
ee.Initialize(credentials)

# Define a region of interest
roi = ee.Geometry.Rectangle([75.0, 8.5, 76.0, 9.5])  # Smaller region

# Select a Landsat 8 image collection (Collection 2) and reduce to necessary bands
collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
               .filterDate('2022-01-01', '2022-12-31') \
               .filterBounds(roi) \
               .select([3, 2, 1])  # Selecting bands by index (3 = SR_B4, 2 = SR_B3, 1 = SR_B2)

# Filter the collection to ensure there are images available
filtered_collection = collection.filterBounds(roi)
if filtered_collection.size().gt(0):
    image = collection.first()
    
    # Get the download URL for the image
    url = image.getDownloadURL({
        'scale': 100,  # Reduced resolution for smaller file size
        'region': roi
    })
    print(f"Fetched download URL from Google Earth Engine.")
    
    # Download the ZIP file
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall('../Datasets/GoogleEarthEngine/')  # Extracts files to the 'output' directory
        print("Download and extraction complete.")
    else:
        print("Failed to download the file.")
else:
    print("No images found in the filtered collection.")