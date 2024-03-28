import pandas as pd
import requests
import os

# The student's ID
student_id = '001364857'  # Replaces with the actual student ID

# The base URL without the ID and specific file name
base_url = "https://tinyurl.com/ChrisCoDV/{}/{}"

# The list of dataset names as they appear in the URLs
datasets = [
    "CinemaWeeklyVisitors.csv",
    "CinemaAge.csv",
    "CinemaCapacity.csv",
    "CinemaMarketing.csv",
    "CinemaOverheads.csv",
    "CinemaSpend.csv",
]

def download_dataset(file_name):
    """
    Constructs the URL, downloads the dataset, and saves it as a CSV file.
    """
    # Constructs the URL by formatting with the student ID and the dataset name
    url = base_url.format(student_id, file_name)
    
    # Makes a request to the URL
    response = requests.get(url)
    
    # Checks if the request was successful
    if response.status_code == 200:
        # Reads the content of the request into a pandas DataFrame
        data = pd.read_csv(url)
        
        # Defines the directory where the file will be saved
        directory = '../Datasets'
        
        # Checks if the directory exists
        if not os.path.exists(directory):
            # If the directory does not exist, creates it
            os.makedirs(directory)
        
        # Saves the DataFrame to a CSV file in the specified directory
        data.to_csv(f'{directory}/{file_name}', index=False)
        print(f"Downloaded and saved {file_name} successfully.")
    else:
        print(f"Failed to download {file_name}. Status code: {response.status_code}")

# Downloads all datasets
# Loops over each dataset and calls the download function
for dataset in datasets:
    download_dataset(dataset)