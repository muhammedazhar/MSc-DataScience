import sys
import logging
import json
import geojson

# GeoJSON extractor function
def extract_bbox(geojson_path):
    """
    Extract the bounding box and "code" property from a GeoJSON file.
    
    Args:
        geojson_path (str): Path to the GeoJSON file.
    
    Returns:
        tuple: Bounding box as (min_lon, min_lat, max_lon, max_lat), and the "code" property.
    """
    try:
        with open(geojson_path, 'r') as geojson_file:
            data = geojson.load(geojson_file)
            
            # Assuming that the first feature contains the Polygon we are interested in
            feature = data['features'][0]
            coordinates = feature['geometry']['coordinates'][0]
            code = feature['properties']['code']
            name = feature['properties']['name']
            
            # Extracting all longitude and latitude values
            lons = [coord[0] for coord in coordinates]
            lats = [coord[1] for coord in coordinates]
            
            # Calculate the bounding box
            min_lon = min(lons)
            max_lon = max(lons)
            min_lat = min(lats)
            max_lat = max(lats)
            
            return (min_lon, min_lat, max_lon, max_lat), code, name
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