"""Single-frame inference with the five released AiRICS ConvNeXt-Tiny folds."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


def build_model():
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Sequential(nn.Dropout(0.4), nn.Linear(768, 2))
    return model


def load_models(weights_dir, device):
    manifest_path = Path(__file__).with_name("weights_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ensemble = []
    for entry in manifest["files"]:
        path = Path(weights_dir) / entry["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing released checkpoint: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {path.name}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model()
        model.load_state_dict(state, strict=True)
        ensemble.append(model.to(device).eval())
    return ensemble


def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize(236, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image.convert("RGB"))


@torch.inference_mode()
def predict(image, ensemble, device):
    tensor = preprocess(image).unsqueeze(0).to(device)
    fold_probabilities = [torch.softmax(model(tensor), dim=1)[0, 1].item()
                          for model in ensemble]
    probability = sum(fold_probabilities) / len(fold_probabilities)
    return {
        "positive_probability": probability,
        "predicted_class": int(probability >= 0.5),
        "label": "struvite-positive" if probability >= 0.5 else "struvite-negative",
        "threshold": 0.5,
        "fold_positive_probabilities": fold_probabilities,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True,
                        help="QC-cleared endoscopic stone surface or cross-section image")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable; use --device cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ensemble = load_models(args.weights_dir, args.device)
    with Image.open(args.image) as image:
        print(json.dumps(predict(image, ensemble, args.device), indent=2))


if __name__ == "__main__":
    main()
