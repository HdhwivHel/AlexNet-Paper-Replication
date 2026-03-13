import os
import shutil
from torchvision import transforms
from torchvision.datasets import ImageFolder
from datasets import load_dataset
from tqdm import tqdm


DATA_ROOT = "data"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "validation")


def _split_has_images(split_dir):
    if not os.path.isdir(split_dir):
        return False

    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for file_name in os.listdir(class_dir):
            if os.path.isfile(os.path.join(class_dir, file_name)):
                return True
    return False


def _dataset_is_ready():
    return _split_has_images(TRAIN_DIR) and _split_has_images(VAL_DIR)


def download_datasets():
    if _dataset_is_ready():
        return

    # Remove partial/corrupt local copies so re-download starts cleanly.
    if os.path.exists(TRAIN_DIR):
        shutil.rmtree(TRAIN_DIR)
    if os.path.exists(VAL_DIR):
        shutil.rmtree(VAL_DIR)

    print("ImageNet100 not found. Downloading dataset...")
    dataset = load_dataset("clane9/imagenet-100")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    for split, target_dir in [("train", TRAIN_DIR), ("validation", VAL_DIR)]:
        for example in tqdm(dataset[split]):
            label = str(example["label"])
            img = example["image"]
            class_dir = os.path.join(target_dir, label)
            os.makedirs(class_dir, exist_ok=True)
            img_path = os.path.join(class_dir, f"{hash(img.tobytes())}.jpg")
            img.save(img_path)


def get_datasets():
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
        ]
    )
    download_datasets()
    train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset = ImageFolder(VAL_DIR, transform=test_transform)
    return train_dataset, test_dataset
