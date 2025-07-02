from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

from PIL import Image
from settings import batch_size

import torchvision.transforms as transforms


def is_valid_image(img_path):
    try:
        with Image.open(img_path) as img:
            img.verify()

        return True
    except (IOError, SyntaxError):
        return False


original_dataset_dir = Path("D:\\cats_vs_dogs\\PetImages")
working_dir = Path("D:\\cats_vs_dogs\\data")
working_dir.mkdir(parents=True, exist_ok=True)

train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15


def split_and_copy():
    for class_path in original_dataset_dir.iterdir():
        if not class_path.is_dir():
            continue

        images = list(class_path.iterdir())
        found_images = len(images)
        images = [image for image in tqdm(images, desc=f"Validating '{class_path.name}' images") if
                  is_valid_image(image)]
        print(f"Discarded {found_images - len(images)
                           } from the {class_path.name} set")
        train, temp = train_test_split(
            images, test_size=1 - train_ratio, random_state=42)
        val, test = train_test_split(
            temp, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)

        for folder, subset in zip(["train", "val", "test"], [train, val, test]):
            class_subfolder = working_dir / folder / class_path.name
            class_subfolder.mkdir(parents=True, exist_ok=True)
            for img in tqdm(subset, desc=f"Copying {folder} images of {class_path.name}s..."):
                link_path = class_subfolder / img.name
                if not link_path.exists():
                    link_path.symlink_to(img)


if not (working_dir / "train").exists():
    print("Splitting dataset into train/val/test folders...")
    split_and_copy()
    print('Done!')

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

datasets = {
    "train": ImageFolder(working_dir/"train", transform=train_transforms),
    "val": ImageFolder(working_dir/"val", transform=val_test_transforms),
    "test": ImageFolder(working_dir/"test", transform=val_test_transforms),
}

dataloaders = {
    phase: DataLoader(
        datasets[phase], batch_size=batch_size, shuffle=(phase == "train"))
    for phase in ["train", "val", "test"]
}
