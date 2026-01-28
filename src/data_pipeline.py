import os
import random

import cv2
import numpy as np
from cv2.typing import MatLike, Size


def load_image(path: str) -> MatLike:
    """loads an image in grayscale"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image at {path}")
    return img


def apply_clahe(
    img: MatLike, clip_limit: float = 2.0, grid_size: Size = (8, 8)
) -> MatLike:
    """Applies Contrast Limited Adaptive Histogram Equalization"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)


def preprocess_image(img: MatLike, target_size: Size = (224, 224)) -> MatLike:
    """Applies CLAHE, resizes and normalize the image."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = apply_clahe(img)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    # Add channel dimension (H, W, 1)
    img = np.expand_dims(img, axis=-1)
    return img


class ChestXrayDataset:
    def __init__(
        self,
        root_dir: str,
        subset: str = "train",
        target_size: Size = (224, 224),
        augment: bool = False,
    ):
        self.root_dir = root_dir
        self.subset = subset
        self.target_size = target_size
        self.augment = augment
        self.classes = ["NORMAL", "PNEUMONIA"]
        self.image_paths = []
        self.labels = []

        subset_dir = os.path.join(root_dir, subset)
        for i, cls in enumerate(self.classes):
            cls_dir = os.path.join(subset_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(i)

        # Shuffle initial list
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        if combined:
            self.image_paths, self.labels = zip(*combined)
        else:
            self.image_paths, self.labels = [], []

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = load_image(img_path)
        img = preprocess_image(img, self.target_size)

        if self.augment:
            img = self.apply_augmentation(img)

        return img, label

    def apply_augmentation(self, img: MatLike) -> MatLike:
        # Horizontal Flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            img = np.expand_dims(img, axis=-1)

        # Rotation
        angle = random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
        img = np.expand_dims(img, axis=-1)

        # Zoom (simple croop and resize)
        if random.random() > 0.5:
            zoom_factor = random.uniform(0.8, 1.0)
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            img_cropped = img[start_h : start_h + new_h, start_w : start_w + new_w]
            img = cv2.resize(img_cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            img = np.expand_dims(img, axis=-1)

        return img


def data_loader(dataset: ChestXrayDataset, batch_size: int = 32, shuffle: bool = True):
    n = len(dataset)
    indices = list(range(n))
    if shuffle:
        random.shuffle(indices)

    for i in range(0, n, batch_size):
        batch_indices = indices[i : i + batch_size]
        if len(batch_indices) < batch_size and i != 0:
            # Skip incomplete last batch
            pass

        batch_imgs = []
        batch_labels = []
        for idx in batch_indices:
            img, label = dataset[idx]
            batch_imgs.append(img)
            batch_labels.append(label)

        yield np.array(batch_imgs), np.array(batch_labels)
