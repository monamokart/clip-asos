import torch
import torch.nn.functional as F
from peft import LoraConfig
from peft import PeftModel
from peft import get_peft_model
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from transformers import CLIPModel


def get_lora_model(
    clip_version: str, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1
) -> PeftModel:
    """Load a CLIP model with LoRA adapters using 8-bit quantization.

    Args:
        clip_version: Hugging Face model identifier for the CLIP model.
        r: LoRA rank dimension. Defaults to 8.
        lora_alpha: LoRA scaling factor. Defaults to 16.
        lora_dropout: Dropout probability for LoRA layers. Defaults to 0.1.

    Returns:
        A CLIP model wrapped with LoRA adapters.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = CLIPModel.from_pretrained(
        clip_version, quantization_config=bnb_config, device_map="auto"
    )

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )

    model = get_peft_model(model, lora_config)
    return model


def clip_loss(
    image_embeds: torch.Tensor, text_embeds: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """Compute the CLIP contrastive loss.

    Calculates the symmetric cross-entropy loss between image and text embeddings,
    commonly used in vision-language model training.

    Args:
        image_embeds: Image embedding tensor of shape (batch_size, embedding_dim).
        text_embeds: Text embedding tensor of shape (batch_size, embedding_dim).
        temperature: Temperature parameter for scaling logits. Defaults to 0.07.

    Returns:
        A scalar tensor representing the average contrastive loss.
    """
    logits = image_embeds @ text_embeds.T / temperature
    labels = torch.arange(len(logits), device=logits.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_t) / 2


def train_one_epoch(
    model: PeftModel, loader, optimizer: torch.optim.Optimizer, device: torch.device
) -> float:
    """Train the model for one epoch.

    Args:
        model: A CLIP model wrapped with LoRA adapters.
        loader: DataLoader providing batches of image-text pairs.
        optimizer: Optimizer for updating model parameters.
        device: Device to run the training on (CPU or GPU).

    Returns:
        Average loss over the epoch.
    """
    model.train()
    total_loss = 0

    for batch in tqdm(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)

        img = F.normalize(outputs.image_embeds, dim=-1)
        txt = F.normalize(outputs.text_embeds, dim=-1)

        loss = clip_loss(img, txt)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: PeftModel, loader, device: torch.device) -> float:
    """Evaluate the model on a dataset.

    Computes the Recall@1 metric by comparing image and text embeddings.

    Args:
        model: A CLIP model wrapped with LoRA adapters.
        loader: DataLoader providing batches of image-text pairs.
        device: Device to run the evaluation on (CPU or GPU).

    Returns:
        Recall@1 score as a float value between 0 and 1.
    """
    model.eval()

    img_embs = []
    txt_embs = []

    for batch in tqdm(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)

        img = F.normalize(outputs.image_embeds, dim=-1)
        txt = F.normalize(outputs.text_embeds, dim=-1)

        img_embs.append(img.cpu())
        txt_embs.append(txt.cpu())

    img_embs = torch.cat(img_embs)
    txt_embs = torch.cat(txt_embs)

    sim = img_embs @ txt_embs.T
    ranks = sim.argmax(dim=1)

    recall_at_1 = (ranks == torch.arange(len(ranks))).float().mean()
    return recall_at_1.item()
