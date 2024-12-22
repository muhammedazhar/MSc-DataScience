import xml.etree.ElementTree as ET
import glob
import os

def rename_and_combine_kml(input_files, output_file):
    # Define the KML namespace
    kml_ns = 'http://www.opengis.net/kml/2.2'
    namespaces = {'kml': kml_ns}

    # Register namespace to preserve it in the output
    ET.register_namespace('', kml_ns)

    # Create the root element for the combined KML
    combined_kml = ET.Element('{%s}kml' % kml_ns)
    combined_doc = ET.SubElement(combined_kml, 'Document')

    plot_num = 1

    for file in input_files:
        tree = ET.parse(file)
        root = tree.getroot()

        # Find all Placemark elements
        for placemark in root.findall('.//kml:Placemark', namespaces):
            name_tag = placemark.find('kml:name', namespaces)
            if name_tag is not None:
                actual_value = name_tag.text if name_tag.text else ''
                new_value = f'PLOT{plot_num}-{actual_value}'
                name_tag.text = new_value
                plot_num += 1

            # Append the Placemark to the combined Document
            combined_doc.append(placemark)

    # Write the combined KML to the output file
    tree = ET.ElementTree(combined_kml)
    tree.write(output_file, encoding='UTF-8', xml_declaration=True)

def main():
    # Find all .kml files in the current directory
    kml_files = glob.glob('*.kml')

    if len(kml_files) < 2:
        print("Not enough .kml files found in the current directory.")
        return

    # Display the found .kml files
    print("Found the following .kml files:")
    for idx, file in enumerate(kml_files):
        print(f"{idx + 1}. {file}")

    # Assume the first two files
    file1 = kml_files[0]
    file2 = kml_files[1]

    # Prompt the user to confirm the files
    print(f"\nThe first two .kml files are:\n1. {file1}\n2. {file2}")
    confirm = input("Do you want to use these files? (y/n): ")

    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return

    # Specify the output KML file
    output_file = 'Combined.kml'

    # Call the function to rename and combine KML files
    rename_and_combine_kml([file1, file2], output_file)
    print(f"Combined KML file has been created: {output_file}")

if __name__ == "__main__":
    main()