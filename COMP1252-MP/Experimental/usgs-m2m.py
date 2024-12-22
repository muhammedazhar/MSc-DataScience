from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer
from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
load_dotenv('../Keys/.env')

# Replace with your USGS credentials
username = os.getenv('USGS_USERNAME')
password = os.getenv('USGS_PASSWORD')

# Search parameters
bbox = (75.0, 8.5, 76.0, 9.5)  # Smaller region
start_date = "2022-01-01"
end_date = "2022-12-31"

# Initialize API with username and password
api = API(username, password)

# Search for Landsat 8 scenes
scenes = api.search(
    dataset='landsat_ot_c2_l2',
    bbox=bbox,
    start_date=start_date,
    end_date=end_date,
    max_cloud_cover=10
)

print(f"{len(scenes)} scenes found.")

# Initialize EarthExplorer
ee = EarthExplorer(username, password)

# Try to download a scene, retrying with the next one if the first fails
downloaded = False
for scene in scenes:
    try:
        print(f"Downloading scene {scene['display_id']}...")
        ee.download(scene['display_id'], output_dir='../Datasets/USGS-EarthExplorer/')
        print("Download complete.")
        downloaded = True
        break  # Exit loop if download is successful
    except Exception as e:
        print(f"Failed to download scene {scene['display_id']}: {e}")

if not downloaded:
    print("No scenes could be downloaded successfully.")

# Logout
ee.logout()
api.logout()
