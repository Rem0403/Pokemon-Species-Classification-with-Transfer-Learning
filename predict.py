"""Name the Pokémon in one or more images with a trained model (for a live demo).

Run: .venv/Scripts/python predict.py my_image.png other.jpg
     .venv/Scripts/python predict.py my_image.png --run finetune_aug_all
Default model: the final refined one, results/refine/cosine30_s42.
"""
import argparse
import json

import torch

from data import build_transform, load_image
from models import build_model
from train import RESULTS_DIR


@torch.no_grad()
def predict_image(path, model, classes: list, device, k: int = 5) -> list[tuple[str, float]]:
    """Top-k (species, probability), most likely first."""
    model.eval()
    x = build_transform(train=False, aug=False)(load_image(path))[None].to(device)
    probs = model(x).float().softmax(dim=1)[0].cpu()
    top = probs.topk(min(k, len(classes)))
    return [(classes[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())]


def load_trained(run: str, device):
    """Rebuild the model a run trained (method, backbone, classes) and load its best weights."""
    m = json.loads((RESULTS_DIR / run / "metrics.json").read_text(encoding="utf-8"))
    model = build_model(m["method"], len(m["classes"]), pretrained=False, arch=m.get("config", {}).get("arch", "resnet18"))  # the first 11 runs predate "config"
    model.load_state_dict(torch.load(RESULTS_DIR / run / "best.pt", map_location=device))
    return model.to(device), m["classes"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--run", default="refine/cosine30_s42", help="run folder inside results/")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_trained(args.run, device)
    for path in args.images:
        print(f"\n{path}")
        for name, p in predict_image(path, model, classes, device):
            print(f"  {p:6.1%}  {name}")
