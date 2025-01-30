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

# -----------------------------------------------------------------------------
# Essential Imports
# -----------------------------------------------------------------------------
import earthaccess

# -----------------------------------------------------------------------------
# Local Imports
# -----------------------------------------------------------------------------
from helper import *
from extractor import *

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
filename = os.path.splitext(os.path.basename(__file__))[0]
# Set up logging using the imported function
logger, file_logger = setup_logging(file=filename)

# -----------------------------------------------------------------------------
# Path and Directory Setup
# -----------------------------------------------------------------------------
path = os.path.dirname(os.path.realpath(__file__))
dataset = "../Datasets"
input_path = "../Datasets/BoundingBox"
system = "NASA-Earthdata"
directory = os.path.join(dataset, system)

bbox_file_path = os.path.join(path, input_path, "bbox.geojson")
search_file_path = os.path.join(path, input_path, "search.json")

# -----------------------------------------------------------------------------
# Configuration and Parameter Setup
# -----------------------------------------------------------------------------
bboxes_info = extract_bboxes(bbox_file_path)
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
logger.info(f"Cloud cover : {cloud_cover}")
logger.info(f"Short name  : {short_name}")
logger.info(f"Start date  : {start}")
logger.info(f"End date    : {end}")
logger.info(f"Temporal    : {temporal}")
logger.info(f"Count       : {count}")

# -----------------------------------------------------------------------------
# NASA Earthdata Authentication
# -----------------------------------------------------------------------------
try:
    auth = earthaccess.login(strategy="environment", persist=True)
    logger.info(f"Auth object: {auth}")

    if auth is None:
        raise ValueError("Login failed. Auth object is None.")
    logger.info("Login successful!")
except Exception as e:
    logger.error(f"An unexpected error occurred during login: {str(e)}")
    logger.error("Please check your NASA credentials and ensure they are correctly set in your environment variables.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Data Download Pipeline
# -----------------------------------------------------------------------------
for idx, (bbox, plotcode, plotstatus) in enumerate(bboxes_info, start=1):
    logger.info(f"Processing polygon {idx}/{len(bboxes_info)}: {plotcode} - {plotstatus}")
    logger.info(f"Extracted bounding box from GeoJSON: {bbox}")
    logger.info(f"Extracted code from GeoJSON: {plotcode}")
    logger.info(f"Downloading data for {plotstatus}")

    # Directory for the current polygon
    output_dir = os.path.join(directory, plotcode)

    # Ensure the dynamic download directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Directory '{plotcode}' created.")
    else:
        logger.info(f"Directory '{plotcode}' already exists.")

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
        logger.info(f"Downloaded files for {plotcode} - {plotstatus}")

    except ValueError as e:
        logger.error(f"Value Error for {plotcode}: {str(e)}")
    except Exception as e:
        # Catch any other unexpected exceptions
        logger.error(f"An unexpected error occurred for {plotcode}: {str(e)}")
        logger.error("Please check your NASA credentials and ensure they are correctly set in your environment variables.")

# -----------------------------------------------------------------------------
# Script Completion
# -----------------------------------------------------------------------------
logger.info(f"The {filename} script execution completed.")
