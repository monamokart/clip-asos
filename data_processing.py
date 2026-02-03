import os
from io import BytesIO

import pandas as pd
import requests
import torch
from google.cloud import storage
from PIL import Image
from torch.utils.data import Dataset


def load_image(url: str, timeout: int = 5) -> Image.Image:
    """Load an image from a URL.

    Args:
        url (str): The URL of the image to load.
        timeout (int, optional): The timeout for the request in seconds. Defaults to 5.

    Returns:
        Image.Image: The loaded image in RGB format.
    """
    response = requests.get(url, timeout=timeout)
    return Image.open(BytesIO(response.content)).convert("RGB")


class ClipDataset(Dataset):
    def __init__(self, csv_path, processor):
        self.df = pd.read_csv(csv_path)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = load_image(row["im"])
        text = row["text"]

        inputs = self.processor(
            images=image, text=text, return_tensors="pt", padding="max_length", truncation=True
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate a batch of data into a single dictionary.

    Args:
        batch (list[dict]): A list of dictionaries containing data samples.

    Returns:
        dict: A dictionary with stacked tensors for each key in the batch.
    """
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def upload_folder_to_gcs(local_folder: str, bucket_name: str, gcs_prefix: str) -> None:
    """Upload a folder to Google Cloud Storage.

    Args:
        local_folder (str): The path to the local folder to upload.
        bucket_name (str): The name of the GCS bucket.
        gcs_prefix (str): The prefix for the GCS path where files will be uploaded.

    Returns:
        None
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_folder)
            blob_path = os.path.join(gcs_prefix, rel_path)

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)

            print(f"⬆️ {local_path} → gs://{bucket_name}/{blob_path}")


def download_folder_from_gcs(bucket_name: str, gcs_prefix: str, local_folder: str) -> None:
    """Download a folder from Google Cloud Storage.

    Args:
        bucket_name (str): The name of the GCS bucket.
        gcs_prefix (str): The prefix for the GCS path to download.
        local_folder (str): The path to the local folder where files will be saved.

    Returns:
        None
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=gcs_prefix)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        rel_path = os.path.relpath(blob.name, gcs_prefix)
        local_path = os.path.join(local_folder, rel_path)

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)

        print(f"⬇️ gs://{bucket_name}/{blob.name} → {local_path}")
