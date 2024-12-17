#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helper Functions
----------------
This script contains helper functions for the Earth Data API. It includes
functions for setting up logging, checking environment variables, and
formatting search results.

Author: Azhar Muhammed
Date: July 2024
"""

import os
import sys
import logging
from dotenv import load_dotenv

def setup_logging():
    # Main logger with both handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../Docs/Logs/processing.log'),
            logging.StreamHandler()
        ]
    )

    # File-only logger
    file_logger = logging.getLogger('file_only')
    file_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('../Docs/Logs/processing.log')
    file_logger.addHandler(file_handler)

    # Prevent messages from propagating to the root logger
    file_logger.propagate = False

    return file_logger

# Load the environment variables from the .env file
load_dotenv("../Keys/.env")

def env_check(var_name, placeholder):
    """
    Check if the environment variable is correctly set.
    
    Args:
        var_name (str): Name of the environment variable.
        placeholder (str): Default placeholder value to check against.
    
    Returns:
        str: Value of the environment variable.
    
    Raises:
        SystemExit: If the variable is not set or set to the default placeholder.
    """
    value = os.getenv(var_name)
    if not value or value == placeholder:
        logging.error(f"Error: {var_name} is not set or is set to the default placeholder. Please update your .env file.")
        sys.exit(1)
    return value

def format(results):
    formatted_results = []
    
    for idx, result in enumerate(results, start=1):
        # Use .get() to safely access dictionary keys
        collection = result.get('Collection', {})
        collection_title = collection.get('EntryTitle', 'No Title')

        # Safely handle spatial coverage
        spatial_coverage = result.get('Spatial coverage', {}).get('HorizontalSpatialDomain', {}).get('Geometry', {}).get('GPolygons', [])
        if spatial_coverage:
            boundary_points = spatial_coverage[0].get('Boundary', {}).get('Points', [])
        else:
            boundary_points = 'No spatial coverage available'

        # Safely handle temporal coverage
        temporal_coverage = result.get('Temporal coverage', {}).get('SingleDateTime', 'No Temporal Data')

        # Handle size
        size = result.get('Size(MB)', 'Unknown Size')

        # Handle data URLs
        data_urls = result.get('Data', [])

        # Formatting each search result
        formatted_results.append(f"Result {idx}:\n"
                                 f"Title: {collection_title}\n"
                                 f"Spatial Coverage (Boundary Points): {boundary_points}\n"
                                 f"Temporal Coverage: {temporal_coverage}\n"
                                 f"Size: {size} MB\n"
                                 f"Data URLs:\n" + "\n".join(f" - {url}" for url in data_urls) + "\n")

    return "\n".join(formatted_results)