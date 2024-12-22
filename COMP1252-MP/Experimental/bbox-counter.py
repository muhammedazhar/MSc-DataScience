import xml.etree.ElementTree as ET
import glob
import sys

# Find all .kml files in the current directory
file_list = glob.glob('*.kml')

if not file_list:
    print("No KML files found in the current directory.")
    sys.exit()

# Display the list of KML files
print("Select a KML file to process:")
for idx, file_name in enumerate(file_list, 1):
    print(f"{idx}. {file_name}")

# Prompt the user to select a file
try:
    choice = int(input("Enter the number of the KML file to process: "))
    if 1 <= choice <= len(file_list):
        selected_file = file_list[choice - 1]
    else:
        print("Invalid selection.")
        sys.exit()
except ValueError:
    print("Invalid input. Please enter a number.")
    sys.exit()

# Load the selected KML file
try:
    tree = ET.parse(selected_file)
    root = tree.getroot()
    print(f"Loaded dataset: {selected_file}")
except ET.ParseError:
    print("Error parsing the KML file.")
    sys.exit()

# Define the KML namespace
namespace = {'kml': 'http://www.opengis.net/kml/2.2'}

# Count the number of bounding boxes
bounding_box_count = 0
for placemark in root.findall(".//kml:Placemark", namespaces=namespace):
    if (
        placemark.find(".//kml:Polygon", namespaces=namespace) is not None or
        placemark.find(".//kml:Point", namespaces=namespace) is not None
    ):
        bounding_box_count += 1

print(f"Number of bounding boxes found: {bounding_box_count}")