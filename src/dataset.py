
# torch is the main PyTorch library
import torch

# DataLoader helps load images in batches
from torch.utils.data import DataLoader

# torchvision contains image utilities for deep learning
from torchvision import datasets, transforms

# os helps create proper file paths
import os

# =========================================================
# BASE PROJECT DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
# =========================================================
# DATASET PATHS
# =========================================================

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "train"
)

VAL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "val"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "test"
)


# =========================================================
# IMAGE TRANSFORMATIONS
# =========================================================
# Transformations are preprocessing steps applied to images
#
# Why needed?
# Neural networks cannot directly understand raw images.
# Images must be converted into a standardized format.
# =========================================================


# =========================================================
# TRAIN TRANSFORMS
# =========================================================
# These transformations are applied ONLY to training images.
#
# Why augmentation here?
# To help model generalize better and reduce overfitting.
# =========================================================

train_transform = transforms.Compose([

    # Resize all images to same size
    # CNN requires fixed image dimensions
    transforms.Resize((128, 128)),


    # Randomly flip image horizontally
    # Helps model learn different orientations
    transforms.RandomHorizontalFlip(),


    # Randomly rotate image slightly
    # Helps model become robust to small rotations
    transforms.RandomRotation(10),


    # Convert image into PyTorch tensor
    # Required before passing image into model
    transforms.ToTensor(),


    # Normalize image pixel values
    #
    # Original pixel range:
    # 0 -> 255
    #
    # After ToTensor():
    # 0 -> 1
    #
    # Normalize makes training more stable
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# =========================================================
# VALIDATION + TEST TRANSFORMS
# =========================================================
# No augmentation here.
#
# Why?
# Validation and test data should remain unchanged
# so evaluation is fair and realistic.
# =========================================================

test_transform = transforms.Compose([

    # Resize image
    transforms.Resize((128, 128)),

    # Convert image to tensor
    transforms.ToTensor(),

    # Normalize image
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# =========================================================
# LOAD DATASETS USING IMAGEFOLDER
# =========================================================
#
# ImageFolder automatically:
# - reads folder names as class labels
# - loads all images
#
# Your folder structure:
#
# train/
#    ok/
#    not_ok/
#
# Automatically becomes:
#
# ok      -> class 0
# not_ok  -> class 1
# =========================================================


# Training dataset
train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform
)

# Validation dataset
val_dataset = datasets.ImageFolder(
    root=VAL_DIR,
    transform=test_transform
)

# Test dataset
test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=test_transform
)


# =========================================================
# CREATE DATALOADERS
# =========================================================
#
# DataLoader helps:
# - load images in batches
# - shuffle training data
# - efficiently feed data into model
# =========================================================


# Batch size:
# Number of images processed at one time
BATCH_SIZE = 32


# Training DataLoader
train_loader = DataLoader(
    dataset=train_dataset,

    # Number of images per batch
    batch_size=BATCH_SIZE,

    # Shuffle training data
    # Helps model learn better
    shuffle=True
)


# Validation DataLoader
val_loader = DataLoader(
    dataset=val_dataset,

    batch_size=BATCH_SIZE,

    # No need to shuffle validation data
    shuffle=False
)


# Test DataLoader
test_loader = DataLoader(
    dataset=test_dataset,

    batch_size=BATCH_SIZE,

    # No need to shuffle test data
    shuffle=False
)


# =========================================================
# CLASS NAMES
# =========================================================
# Stores folder/class names
#
# Example:
# ['not_ok', 'ok']
# =========================================================

class_names = train_dataset.classes


# =========================================================
# PRINT DATASET INFORMATION
# =========================================================
# Useful for checking everything loaded correctly
# =========================================================
if __name__ == "__main__":
    print("===================================")
    print("Dataset Loaded Successfully")
    print("===================================")

    print(f"Classes: {class_names}")

    print(f"Training Images: {len(train_dataset)}")
    print(f"Validation Images: {len(val_dataset)}")
    print(f"Test Images: {len(test_dataset)}")

    print("===================================")


    # =========================================================
    # TEST ONE BATCH
    # =========================================================
    # Helps verify:
    # - image shape
    # - batch loading
    # - labels
    # =========================================================

    # Get one batch from training loader
    images, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")


    # Example expected output:
    #
    # Batch Image Shape:
    # torch.Size([32, 3, 128, 128])
    #
    # Meaning:
    # 32 -> batch size
    # 3 -> RGB channels
    # 128 -> height
    # 128 -> width