from pathlib import Path

RAW_PATH = Path("data/raw")

good_path = RAW_PATH / "train/good"

count = 0
for img in good_path.rglob("*"):
    if img.is_file():
        count += 1

##print("Good images in train:",count)

good_paths = [
    RAW_PATH / "train/good",
    RAW_PATH / "test/good"
]

# count = 0
# for path in good_paths:
#     for img in path.rglob("*"):
#         if img.is_file():
#             count += 1

##print("Total GOOD images:", count)


defect_folders = ["crack", "cut", "hole", "print"]

# count = 0

# for defect in defect_folders:
#     folder = RAW_PATH / f"test/{defect}"
#     for img in folder.rglob("*"):
#         if img.is_file():
#             count += 1

# print("Total DEFECT images:", count)

data = []

# GOOD → ok
for path in good_paths:
    for img in path.rglob("*"):
        if img.is_file():
            data.append((img, "ok"))

# DEFECT → not_ok
for defect in defect_folders:
    folder = RAW_PATH / f"test/{defect}"
    for img in folder.rglob("*"):
        if img.is_file():
            data.append((img, "not_ok"))

# print("Total samples:", len(data))
# print("First sample:", data[0])



import random

random.shuffle(data)

# print("After shuffle:", data[0])


total = len(data)

train_split = int(0.7 * total)
val_split = int(0.85 * total)

train_data = data[:train_split]
val_data = data[train_split:val_split]
test_data = data[val_split:]

print(len(train_data), len(val_data), len(test_data))


import shutil

# img_path, label = train_data[0]

# dest = Path("data/processed/train") / label / img_path.name

# print("Copying:", img_path, "→", dest)

# unique_name = f"{img_path.parent.name}_{img_path.name}"
# print(unique_name)


import os

for split in ["train", "val", "test"]:
    for cls in ["ok", "not_ok"]:
        os.makedirs(f"data/processed/{split}/{cls}", exist_ok=True)


def copy_files(dataset, split):
    for img_path, label in dataset:
        unique_name = f"{img_path.parent.name}_{img_path.name}"
        dest = Path("data/processed") / split / label / unique_name
        shutil.copy(img_path, dest)

copy_files(train_data, "train")
copy_files(val_data, "val")
copy_files(test_data, "test")