#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extractor Functions
-------------------
This script contains functions to extract bounding boxes and search parameters
from files. It includes functions to extract bounding boxes from GeoJSON files
and search parameters from JSON files.

Author: Azhar Muhammed
Date: July 2024
"""

import sys
import logging
import json
import geojson

# GeoJSON extractor function
def extract_bboxes(geojson_path):
    """
    Extract bounding boxes and properties from a GeoJSON file containing multiple polygons.
    
    Args:
        geojson_path (str): Path to the GeoJSON file.
    
    Returns:
        list: A list of tuples, each containing (bbox, code, status)
    """
    try:
        with open(geojson_path, 'r') as geojson_file:
            data = geojson.load(geojson_file)
            
            bboxes = []

            for feature in data['features']:
                geometry = feature['geometry']
                if geometry['type'] != 'Polygon':
                    continue  # Skip if not a polygon
                coordinates = geometry['coordinates'][0]
                code = feature['properties'].get('code', '')
                status = feature['properties'].get('status', '')
                
                # Extracting all longitude and latitude values
                lons = [coord[0] for coord in coordinates]
                lats = [coord[1] for coord in coordinates]
                
                # Calculate the bounding box
                min_lon = min(lons)
                max_lon = max(lons)
                min_lat = min(lats)
                max_lat = max(lats)
                
                bbox = (min_lon, min_lat, max_lon, max_lat)
                bboxes.append((bbox, code, status))
            
            return bboxes
    except (FileNotFoundError, KeyError, IndexError) as e:
        logging.error(f"Error reading or parsing GeoJSON file: {str(e)}")
        sys.exit(1)

# JSON extractor function
def extract_search(json_path):
    # Open and read the JSON file
    with open(json_path, 'r') as file:
        search = json.load(file)

    # Return values as a dictionary
    return search
