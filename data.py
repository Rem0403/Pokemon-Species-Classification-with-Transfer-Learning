"""Dataset handling: find images, drop broken files and duplicates, split, load."""
import hashlib
import json
import random
from collections import deque
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

SEED = 42
DATA_DIR = Path("data")
SPLITS_PATH = Path("splits.json")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def find_class_root(data_dir: Path) -> Path:
    """Return the shallowest folder whose subfolders (>= 2) each directly hold files.

    Works whether the zip unpacked to data/PokemonData or data/archive/PokemonData.
    """
    queue = deque([Path(data_dir)])
    while queue:
        d = queue.popleft()
        subdirs = sorted(p for p in d.iterdir() if p.is_dir())
        if len(subdirs) >= 2 and all(any(f.is_file() for f in s.iterdir()) for s in subdirs):
            return d
        queue.extend(subdirs)
    raise FileNotFoundError(f"No folder with one subfolder per class found under {data_dir}")


def load_image(path) -> Image.Image:
    """Open any image as RGB. Transparent areas become white, not black."""
    with Image.open(path) as im:
        if im.has_transparency_data:
            rgba = im.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(white, rgba).convert("RGB")
        return im.convert("RGB")


def scan(class_root: Path) -> dict:
    """List usable images with labels; skip unreadable files; keep one copy of exact duplicates.

    Duplicates found under two different species are dropped entirely (no single correct label).
    """
    classes = sorted(p.name for p in Path(class_root).iterdir() if p.is_dir())
    skipped, by_hash = [], {}
    for label, name in enumerate(classes):
        for f in sorted(p for p in (Path(class_root) / name).iterdir() if p.is_file()):
            path = f.as_posix()
            try:
                load_image(f)  # full decode: catches non-images and truncated files
            except Exception:
                skipped.append(path)
                continue
            digest = hashlib.md5(f.read_bytes()).hexdigest()
            by_hash.setdefault(digest, []).append((path, label))
    items = []
    for group in by_hash.values():
        if len({label for _, label in group}) == 1:  # keep the first copy
            items.append(list(group[0]))
        # else: the same picture is filed under two species -> label is ambiguous, drop all copies
    items.sort()
    duplicates = [[p for p, _ in group] for group in by_hash.values() if len(group) > 1]
    return {"classes": classes, "items": items, "skipped": skipped, "duplicates": duplicates}


def make_splits(data_dir: Path = DATA_DIR, seed: int = SEED) -> dict:
    """Stratified 70/15/15 split of the deduplicated images."""
    s = scan(find_class_root(data_dir))
    items = s["items"]
    train, rest = train_test_split(items, test_size=0.30, random_state=seed,
                                   stratify=[label for _, label in items])
    val, test = train_test_split(rest, test_size=0.50, random_state=seed,
                                 stratify=[label for _, label in rest])
    return {"classes": s["classes"], "train": train, "val": val, "test": test,
            "skipped": s["skipped"], "duplicates": s["duplicates"]}


def load_splits(data_dir: Path = DATA_DIR, path: Path = SPLITS_PATH, seed: int = SEED) -> dict:
    """Read the saved split, or create and save it once so every run uses the same one."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    splits = make_splits(data_dir, seed)
    path.write_text(json.dumps(splits, ensure_ascii=False, indent=1), encoding="utf-8")
    return splits


def subsample(items: list, k: int | None, seed: int = SEED) -> list:
    """Keep at most k items per class (all of them if a class has fewer). Order is preserved."""
    if k is None:
        return items
    rng = random.Random(seed)
    by_label = {}
    for i, (_, label) in enumerate(items):
        by_label.setdefault(label, []).append(i)
    keep = set()
    for label in sorted(by_label):
        idx = by_label[label]
        keep.update(rng.sample(idx, min(k, len(idx))))
    return [item for i, item in enumerate(items) if i in keep]


def build_transform(train: bool, aug: bool) -> transforms.Compose:
    """224x224 + ImageNet normalization; random augmentation only for training with aug on."""
    if train and aug:
        steps = [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # No hue jitter: color is a main clue for telling species apart.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    else:
        steps = [transforms.Resize((224, 224))]
    return transforms.Compose(steps + [transforms.ToTensor(),
                                       transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class ImageDataset(Dataset):
    """Turns a list of [path, label] into (image tensor, label) pairs."""

    def __init__(self, items: list, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(load_image(path)), int(label)
