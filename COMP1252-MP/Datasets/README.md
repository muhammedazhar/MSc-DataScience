# Dataset folder for the Falcon Project

To access the dataset `cd` to `Solutions/` direcory and run the [`download.py`](../Solutions/download.py) Python 3 script. Then there will be a folder `NASA-Earthaccess` within this folder along with configured data in it.

## Reference Bounding Box

The reference bounding box is the bounding box of the coordinates that is used to download the datasets from satellite sources. Use the [`bbox.geojson`](./bbox.geojson) file to get the reference bounding box in the [`download.py`](../Solutions/download.py) script. For updating the reference bounding box, update the `bbox.geojson` file using [GeoJSON.io](https://geojson.io/#map=2/11.42/72.58). After that, run the [`download.py`](../Solutions/download.py) script.
