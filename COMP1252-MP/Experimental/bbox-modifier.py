import json
from collections import OrderedDict

# Load the GeoJSON data
with open('bbox.geojson', 'r') as f:
    data = json.load(f)

# Prepare a list to hold the updated features
updated_features = []

# Iterate over each feature
for feature in data['features']:
    properties = feature.get('properties', {})
    name = properties.get('name', '')
    if name:
        # Split the name to extract code and status
        parts = name.split('-', 1)
        if len(parts) == 2:
            code, status = parts
        else:
            code = name
            status = ''
        # Update properties
        properties['code'] = code
        properties['status'] = status
        del properties['name']
    else:
        # Rename 'name' to 'status'
        properties['status'] = properties.pop('name', '')
    # Remove 'styleUrl' property if it exists
    properties.pop('styleUrl', None)
    # Create an ordered feature with 'properties' first
    ordered_feature = OrderedDict([
        ('type', feature.get('type')),
        ('properties', properties),
        ('id', feature.get('id')),
        ('geometry', feature.get('geometry')),
    ])
    updated_features.append(ordered_feature)

# Create the updated data dictionary
updated_data = OrderedDict([
    ('type', data.get('type')),
    ('features', updated_features)
])

# Write the updated data with each feature on a single line
with open('bbox.geojson', 'w') as f:
    f.write('{"type":"FeatureCollection","features":[\n')
    for i, feature in enumerate(updated_data['features']):
        json_str = json.dumps(feature, separators=(',', ':'))
        f.write(json_str)
        if i != len(updated_data['features']) - 1:
            f.write(',\n')
    f.write('\n]}')