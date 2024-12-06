import os
import logging
from turtle import st
import earthaccess

# Local function imports
from helper import *
from extractor import *

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the current directory of this script
path = os.path.dirname(os.path.realpath(__file__))

dataset = "../Datasets"
system = "NASA-Earthdata"
directory = os.path.join(dataset, system)

# Path to the bbox.geojson file
bbox_file_path = os.path.join(path, dataset, "bbox.geojson")
logging.info(f"Path to GeoJSON file: {bbox_file_path}")
search_file_path = os.path.join(path, dataset, "search.json")
logging.info(f"Path to search.json file: {search_file_path}")

# Extract the bounding box and "code" property from the GeoJSON file
bbox, plotcode, plotname = extract_bbox(bbox_file_path)
logging.info(f"Extracted bounding box from GeoJSON: {bbox}")
logging.info(f"Extracted code from GeoJSON: {plotcode}")
logging.info(f"Downloading data for {plotname}")

# Path to the directory where the downloaded data will be stored
# Dynamic directory based on the "code" property in the GeoJSON file
directory = os.path.join(path, f"{directory}/{plotcode}")

# Ensure the dynamic download directory exists
if not os.path.exists(directory):
    os.makedirs(directory)
    logging.info(f"Directory '{plotcode}' created.")
else:
    logging.info(f"Directory '{plotcode}' already exists.")

# Check and get NASA credentials from environment variables
username = env_check('EARTHDATA_USERNAME', "<your_username>")
password = env_check('EARTHDATA_PASSWORD', "<your_password>")

# Setting parameters for the search
# TODO: Change the `short_name` after research.
search = extract_search(search_file_path)

short_name = search.get('short_name')
start = search.get('start')
end = search.get('end')
temporal = (f"{start}", f"{end}")
count = search.get('count')
cloud_cover = search.get('cloud_cover')

# Print the extracted values
logging.info(f"Cloud cover: {cloud_cover}")
logging.info(f"Short name: {short_name}")
logging.info(f"Start date: {start}")
logging.info(f"End date: {end}")
logging.info(f"Temporal: {temporal}")
logging.info(f"Count: {count}")

try:
    # Step 1: Login to NASA Earthdata
    auth = earthaccess.login(strategy="environment", persist=True)
    logging.info(f"Auth object: {auth}")
    
    if auth is None:
        raise ValueError("Login failed. Auth object is None.")
    logging.info("Login successful!")

    # Step 2: Search for data using the bounding box from the GeoJSON
    results = earthaccess.search_data(
        short_name=short_name,
        # cloud_hosted=True,
        bounding_box=bbox,  # Use the extracted bounding box
        temporal=temporal,
        count=count,
        cloud_cover=cloud_cover,
    )
    # results = format(results)
    logging.info(f"Search results:\n{results}")

    # Step 3: Download the data
    files = earthaccess.download(results, directory)
    logging.info(f"Downloaded files: {files}")

# Handle specific exceptions that may occur during the data retrieval process
except ValueError as e:
    logging.error(f"Value Error: {str(e)}")
except Exception as e:
    # Catch any other unexpected exceptions
    logging.error(f"An unexpected error occurred: {str(e)}")
    logging.error("Please check your NASA credentials and ensure they are correctly set in your environment variables.")

# Log the completion of the script execution
logging.info("Script execution completed.")