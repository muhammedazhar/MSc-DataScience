#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Downnloader Script
------------------
This script downloads data from NASA Earthdata based on bounding boxes and
search parameters. It extracts bounding boxes and search parameters from
GeoJSON and JSON files, respectively, and downloads data for each bounding box.

Author: Azhar Muhammed
Date: July 2024
"""


import earthaccess

# Local function imports
from helper import *
from extractor import *

# Set up logging configuration
file_logger = setup_logging()

# Get full path of current script
current_file = __file__
# Extract just the filename
filename = os.path.basename(current_file)
# Filename without extension
filename_no_ext = os.path.splitext(filename)[0]
logging.info(f"Running {filename_no_ext} script...")

# Get the current directory of this script
path = os.path.dirname(os.path.realpath(__file__))

dataset = "../Datasets"
input_path = "../Datasets/BoundingBox"
system = "NASA-Earthdata"
directory = os.path.join(dataset, system)

# Path to the bbox.geojson file
bbox_file_path = os.path.join(path, input_path, "bbox.geojson")
logging.info(f"Loaded GeoJSON file: {os.path.basename(bbox_file_path)}")
search_file_path = os.path.join(path, input_path, "search.json")
logging.info(f"Loaded search parameter file: {os.path.basename(search_file_path)}")

# Extract the bounding box and "code" property from the GeoJSON file
bboxes_info = extract_bboxes(bbox_file_path)

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
logging.info(f"Cloud cover : {cloud_cover}")
logging.info(f"Short name  : {short_name}")
logging.info(f"Start date  : {start}")
logging.info(f"End date    : {end}")
logging.info(f"Temporal    : {temporal}")
logging.info(f"Count       : {count}")

# Step 1: Login to NASA Earthdata
try:
    auth = earthaccess.login(strategy="environment", persist=True)
    logging.info(f"Auth object: {auth}")

    if auth is None:
        raise ValueError("Login failed. Auth object is None.")
    logging.info("Login successful!")
except Exception as e:
    logging.error(f"An unexpected error occurred during login: {str(e)}")
    logging.error("Please check your NASA credentials and ensure they are correctly set in your environment variables.")
    sys.exit(1)

# Iterate over each polygon
for idx, (bbox, plotcode, plotstatus) in enumerate(bboxes_info, start=1):
    logging.info(f"Processing polygon {idx}/{len(bboxes_info)}: {plotcode} - {plotstatus}")
    logging.info(f"Extracted bounding box from GeoJSON: {bbox}")
    logging.info(f"Extracted code from GeoJSON: {plotcode}")
    logging.info(f"Downloading data for {plotstatus}")

    # Directory for the current polygon
    output_dir = os.path.join(directory, plotcode)

    # Ensure the dynamic download directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Directory '{plotcode}' created.")
    else:
        logging.info(f"Directory '{plotcode}' already exists.")

    try:
        # Step 2: Search for data using the bounding box from the GeoJSON
        results = earthaccess.search_data(
            short_name=short_name,
            bounding_box=bbox,  # Use the current bounding box
            temporal=temporal,
            count=count,
            cloud_cover=cloud_cover,
        )
        file_logger.info(f"Search results for {plotcode}:\n{results}")

        # Step 3: Download the data
        files = earthaccess.download(results, output_dir)
        file_logger.info(f"Downloaded files for {plotcode}: {files}")
        logging.info(f"Downloaded files for {plotcode} - {plotstatus}")

    except ValueError as e:
        logging.error(f"Value Error for {plotcode}: {str(e)}")
    except Exception as e:
        # Catch any other unexpected exceptions
        logging.error(f"An unexpected error occurred for {plotcode}: {str(e)}")
        logging.error("Please check your NASA credentials and ensure they are correctly set in your environment variables.")

# Log the completion of the script execution
logging.info(f"The {filename_no_ext} script execution completed.")
