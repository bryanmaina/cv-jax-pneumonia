import os

import numpy as np

from src.data_pipeline import (
    ChestXrayDataset,
    apply_clahe,
    data_loader,
    load_image,
    preprocess_image,
)

SAMPLE_IMAGE_PATH: str = "datasets/chest_xray/test/NORMAL/IM-0001-0001.jpeg"


def test_load_image():
    assert os.path.exists(SAMPLE_IMAGE_PATH)
    img = load_image(SAMPLE_IMAGE_PATH)
    assert img is not None
    assert isinstance(img, np.ndarray)


def test_apply_clahe():
    # Create a low contrast image
    img = np.full((100, 100), 100, dtype=np.uint8)
    img[25:75, 25:75] = 110
    processed = apply_clahe(img)
    assert processed.shape == img.shape
    assert processed.dtype == np.uint8
    # CLAHE should increase contrast/variace
    assert np.std(processed) > np.std(img)


def test_preprocess_image():
    img = np.zeros((300, 300), dtype=np.uint8)
    target_size = (224, 224)
    processed = preprocess_image(img, target_size)
    # Output should be (H, W, C) wher C=1 for grayscale
    assert processed.shape == (224, 224, 1)
    assert processed.max() <= 1.0
    assert processed.min() >= 0.0
    assert processed.dtype == np.float32


def test_dataset_loading():
    root_dir = "datasets/chest_xray"
    dataset = ChestXrayDataset(root_dir, subset="test", target_size=(224, 224))
    assert len(dataset) > 0
    img, label = dataset[0]
    assert img.shape == (224, 224, 1)
    assert label in [0, 1]


def test_data_loader_batching():
    root_dir = "datasets/chest_xray"
    dataset = ChestXrayDataset(root_dir, subset="test", target_size=(224, 224))
    batch_size = 8
    loader = data_loader(dataset, batch_size)
    batch_imgs, batch_labels = next(loader)
    assert batch_imgs.shape == (batch_size, 224, 224, 1)
    assert batch_labels.shape == (batch_size,)


def test_augmentations():
    root_dir = "datasets/chest_xray"
    dataset = ChestXrayDataset(root_dir, subset="test", target_size=(224, 224))
    img = np.random.rand(224, 224, 1).astype(np.float32)
    augmented = dataset.apply_augmentation(img)
    assert augmented.shape == img.shape
    assert augmented.dtype == img.dtype
