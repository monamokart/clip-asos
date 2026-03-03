import argparse

import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

from data_processing import ClipDataset
from data_processing import collate_fn
from data_processing import upload_folder_to_gcs
from training_tools import evaluate
from training_tools import get_lora_model
from training_tools import train_one_epoch


def main(
    clip_version: str,
    train_file: str,
    val_file: str,
    batch_size: int,
    learning_rate: float,
    optimizer_weight_decay: float,
    epochs: int,
    save_dir: str,
    bucket: str,
    device: str,
) -> None:
    """Train a CLIP model with LoRA adaptation.

    Args:
        clip_version: HuggingFace model identifier for CLIP model.
        train_file: Path to training CSV file. Can be a local path or a GCS URI.
        val_file: Path to validation CSV file. Can be a local path or a GCS URI.
        batch_size: Number of samples per batch.
        learning_rate: Learning rate for AdamW optimizer.
        optimizer_weight_decay: Weight decay for AdamW optimizer.
        epochs: Number of training epochs.
        save_dir: Directory to save trained model and processor.
        bucket: GCS bucket name for uploading model artifacts.
        device: Device to use for training ('cuda' or 'cpu').
    """
    processor = CLIPProcessor.from_pretrained(clip_version)

    train_ds = ClipDataset(train_file, processor, bucket)
    val_ds = ClipDataset(val_file, processor, bucket)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn
    )

    model = get_lora_model(clip_version)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=optimizer_weight_decay
    )

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        r1 = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch+1}] loss={train_loss:.4f} | R@1={r1:.4f}")

        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        upload_folder_to_gcs(
            local_folder=save_dir, bucket_name=bucket, gcs_prefix=f"models/{save_dir}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP model with LoRA")
    parser.add_argument(
        "--clip_version", default="openai/clip-vit-base-patch32", help="CLIP model version"
    )
    parser.add_argument(
        "--train_file", default="gs://clip-asos/train.csv", help="Path to training file"
    )
    parser.add_argument(
        "--val_file", default="gs://clip-asos/val.csv", help="Path to validation file"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--optimizer_weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--save_dir", default="clip_lora_ckpt", help="Directory to save model")
    parser.add_argument("--bucket", default="clip-asos", help="GCS bucket name")
    args = parser.parse_args()

    clip_version = args.clip_version
    train_file = args.train_file
    val_file = args.val_file
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    optimizer_weight_decay = args.optimizer_weight_decay
    epochs = args.epochs
    save_dir = args.save_dir
    bucket = args.bucket

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    main(
        clip_version=clip_version,
        train_file=train_file,
        val_file=val_file,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_weight_decay=optimizer_weight_decay,
        epochs=epochs,
        save_dir=save_dir,
        bucket=bucket,
        device=device,
    )
