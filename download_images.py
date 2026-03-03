import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import pandas as pd
import requests
from google.cloud import storage
from tqdm import tqdm

train = pd.read_csv("train.csv")
val = pd.read_csv("val.csv")

images = train["im"].tolist() + val["im"].tolist()

os.makedirs("images", exist_ok=True)


def download_and_save(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        filename = image_url.split("/")[4] + ".png"
        storage_client = storage.Client()
        bucket = storage_client.bucket("clip-asos")
        blob = bucket.blob(f"images/{filename}")
        blob.upload_from_string(response.content, content_type="image/png")

        return True
    except Exception as e:
        print(f"Failed: {image_url} | {e}")
        return False


# Adjust max_workers depending on your bandwidth / server limits
max_workers = 4

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(download_and_save, img) for img in images]

    for _ in tqdm(as_completed(futures), total=len(futures)):
        pass

        # Check which images are missing from the bucket
        storage_client = storage.Client()
        bucket = storage_client.bucket("clip-asos")
        blobs = {blob.name.split("/")[-1] for blob in bucket.list_blobs(prefix="images/")}
