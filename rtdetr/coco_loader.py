import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import json
from typing import List, Tuple, Optional


class CocoDetectionDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        img_size: int = 640,
        transform: Optional[transforms.Compose] = None,
    ):
        self.img_dir = img_dir
        self.img_size = img_size

        with open(ann_file) as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]
        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            self.img_to_anns.setdefault(img_id, []).append(ann)

        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        orig_w, orig_h = image.size
        image = self.transform(image)

        anns = self.img_to_anns.get(img_info["id"], [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x = x / orig_w
            y = y / orig_h
            w = w / orig_w
            h = h / orig_h
            boxes.append([x, y, w, h])
            labels.append(ann["category_id"] - 1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "orig_size": torch.tensor([orig_h, orig_w]),
            "img_id": img_info["id"],
        }

        return image, target


def build_coco_loader(
    img_dir: str,
    ann_file: str,
    batch_size: int = 4,
    img_size: int = 640,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    dataset = CocoDetectionDataset(img_dir, ann_file, img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=_collate_fn)


def _collate_fn(batch: list) -> Tuple[torch.Tensor, List[dict]]:
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets
