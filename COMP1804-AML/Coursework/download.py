import requests
import gzip

# Download URL for the model (choose a reliable source)
url = 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g'

# Download the model file
response = requests.get(url, stream=True)

# Save the model to the current directory
with open('GoogleNews-vectors-negative300.bin.gz', 'wb') as f:
    for chunk in response.iter_content(chunk_size=1024): 
        if chunk:  # filter out keep-alive new chunks
            f.write(chunk)

with gzip.open('GoogleNews-vectors-negative300.bin.gz', 'rb') as f_in:
    with open('GoogleNews-vectors-negative300.bin', 'wb') as f_out:
        f_out.write(f_in.read())

print("File downloaded and extracted successfully!")