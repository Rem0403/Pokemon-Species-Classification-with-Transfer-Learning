"""dataset stuff: find the images, drop broken files and duplicates, split, load"""
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
    """finds the folder that has the class folders in it. works whether the zip unzipped to
    data/PokemonData or data/archive/PokemonData"""
    queue = deque([Path(data_dir)])
    while queue:
        d = queue.popleft()
        subdirs = sorted(p for p in d.iterdir() if p.is_dir())
        if len(subdirs) >= 2 and all(any(f.is_file() for f in s.iterdir()) for s in subdirs):
            return d
        queue.extend(subdirs)
    raise FileNotFoundError(f"No folder with one subfolder per class found under {data_dir}")


def load_image(path) -> Image.Image:
    """opens any image as RGB. transparent parts turn white instead of black"""
    with Image.open(path) as im:
        if im.has_transparency_data:
            rgba = im.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(white, rgba).convert("RGB")
        return im.convert("RGB")


def scan(class_root: Path) -> dict:
    """usable images + labels. skips files that won't open and keeps one copy of exact duplicates.
    if the same pic shows up under two species every copy gets dropped since there's no right label"""
    classes = sorted(p.name for p in Path(class_root).iterdir() if p.is_dir())
    skipped, by_hash = [], {}
    for label, name in enumerate(classes):
        for f in sorted(p for p in (Path(class_root) / name).iterdir() if p.is_file()):
            path = f.as_posix()
            try:
                load_image(f)  # actually decode it so broken/truncated files get caught
            except Exception:
                skipped.append(path)
                continue
            digest = hashlib.md5(f.read_bytes()).hexdigest()
            by_hash.setdefault(digest, []).append((path, label))
    items = []
    for group in by_hash.values():
        if len({label for _, label in group}) == 1:
            items.append(list(group[0]))
    items.sort()
    duplicates = [[p for p, _ in group] for group in by_hash.values() if len(group) > 1]
    return {"classes": classes, "items": items, "skipped": skipped, "duplicates": duplicates}


def make_splits(data_dir: Path = DATA_DIR, seed: int = SEED) -> dict:
    """stratified 70/15/15 split"""
    s = scan(find_class_root(data_dir))
    items = s["items"]
    train, rest = train_test_split(items, test_size=0.30, random_state=seed,
                                   stratify=[label for _, label in items])
    val, test = train_test_split(rest, test_size=0.50, random_state=seed,
                                 stratify=[label for _, label in rest])
    return {"classes": s["classes"], "train": train, "val": val, "test": test,
            "skipped": s["skipped"], "duplicates": s["duplicates"]}


def load_splits(data_dir: Path = DATA_DIR, path: Path = SPLITS_PATH, seed: int = SEED) -> dict:
    """loads the saved split or makes it once so every run uses the same test images"""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    splits = make_splits(data_dir, seed)
    path.write_text(json.dumps(splits, ensure_ascii=False, indent=1), encoding="utf-8")
    return splits


def subsample(items: list, k: int | None, seed: int = SEED) -> list:
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


def build_transform(train: bool, aug: bool, fill: int = 0) -> transforms.Compose:
    """224x224 + imagenet normalization, augmentation only when training with aug on.
    fill is the color of the corners rotation leaves behind (0 black, 255 white)"""
    if train and aug:
        steps = [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15, fill=fill),
            # no hue jitter, color is one of the main ways to tell species apart
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    else:
        steps = [transforms.Resize((224, 224))]
    return transforms.Compose(steps + [transforms.ToTensor(),
                                       transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class ImageDataset(Dataset):

    def __init__(self, items: list, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(load_image(path)), int(label)
