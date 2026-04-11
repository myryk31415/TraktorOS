import boto3
import json
import random

BUCKET_NAME = "traktoros-training-data"

LIGHT = [
    "data/HackHPI2026_release/data/2023-08-09_A550_autonomyTestRecord_Dissen/2023-08-09-17-43-23",
    "data/HackHPI2026_release/data/2023-08-14_A550_autonomyTestRecord_Dissen/2023-08-14-16-00-30",
    "data/HackHPI2026_release/data/2023-08-22_A550_autonomyTestRecord_Bielefeld/2023-08-22-15-37-52",
    "data/HackHPI2026_release/data/2023-08-22_A550_autonomyTestRecord_Bielefeld/2023-08-22-16-19-23",
    "data/HackHPI2026_release/data/2023-09-07_A550_autonomyTestRecord_Bielefeld/2023-09-07-15-27-11",
    "data/HackHPI2025_release/data/2023-09-07_A550_autonomyTestRecord_Bielefeld/2023-09-07-16-50-28",
    "data/HackHPI2026_release/data/2023-09-07_A550_autonomyTestRecord_Bielefeld/2023-09-07-16-56-33",
    "data/HackHPI2026_release/data/2023-09-11_A550_autonomyTestRecord_PrOldendorf/2023-09-11-18-58-32",
    "data/HackHPI2026_release/data/2023-09-07_A550_autonomyTestRecord_Bielefeld/2023-09-07-16-50-28",
]

LENSE_GLAIR = [
    "data/HackHPI2026_release/data/2023-09-07_A550_autonomyTestRecord_Bielefeld/2023-09-07-15-36-21",
    "data/HackHPI2026_release/data/2023-09-07_A550_autonomyTestRecord_Bielefeld/2023-09-07-16-24-19",
    "data/HackHPI2026_release/data/2023-09-07_A550_autonomyTestRecord_Bielefeld/2023-09-07-16-39-56",
    "data/HackHPI2026_release/data/2023-09-07_A550_autonomyTestRecord_Bielefeld/2023-09-07-16-51-32",
    "data/HackHPI2026_release/data/2023-09-07_A550_autonomyTestRecord_Bielefeld/2023-09-07-16-57-33",
    "data/HackHPI2026_release/data/2023-09-11_A550_autonomyTestRecord_PrOldendorf/2023-09-11-17-25-02",
    "data/HackHPI2026_release/data/2023-09-11_A550_autonomyTestRecord_PrOldendorf/2023-09-11-18-09-46",
    "data/HackHPI2026_release/data/2023-09-11_A550_autonomyTestRecord_PrOldendorf/2023-09-11-18-59-45",
]

DARK = [
    "data/HackHPI2026_release/data/2023-09-11_A550_autonomyTestRecord_PrOldendorf/2023-09-11-20-44-31",
    "data/HackHPI2026_release/data/2023-09-11_A550_autonomyTestRecord_PrOldendorf/2023-09-11-22-03-29",
    "data/HackHPI2026_release/data/2023-09-11_A550_autonomyTestRecord_PrOldendorf/2023-09-11-22-04-50",
]

DARK_INSECTS = [
    "data/HackHPI2026_release/data/2023-09-11_A550_autonomyTestRecord_PrOldendorf/2023-09-11-20-53-55",
]

def generate_annotation_split():

    light_information = {}
    glair_information = {}
    dark_information = {}
    insect_information = {}
    
    # get all annotation files
    s3 = boto3.client('s3')
    keys = []
    
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix="data/HackHPI2026_release/annotation/"):
        if 'Contents' in page:
            for obj in page['Contents']:
                keys.append(obj['Key'])
    
    
    for annotation_file in keys:
        name = annotation_file.split("_11")[0].replace("annotation", "data")
        if name in LIGHT:
            info = light_information
        elif name in LENSE_GLAIR:
            info = glair_information
        elif name in DARK:
            info = dark_information
        elif name in DARK_INSECTS:
            info = insect_information
    
        response = s3.get_object(Bucket=BUCKET_NAME, Key=annotation_file)
        content = response['Body'].read().decode('utf-8')
    
        data = json.loads(content)
    
        info.update({name : len(sorted(data.get("images", []), key=lambda item: item["id"]))})
    
    train = []
    val = []
    test = []
    
    light = sorted(light_information.items(), key=lambda x: x[1], reverse=True)

    train.append(light[0])
    train.append(light[1])
    train.append(light[2])
    val.append(light[3])
    test.append(light[4])
    test.append(light[5])
    val.append(light[6])
    
    glair = sorted(glair_information.items(), key=lambda x: x[1], reverse=True)

    train.append(glair[0])
    val.append(glair[1])
    test.append(glair[2])
    test.append(glair[3])
    train.append(glair[4])
    train.append(glair[5])

    dark = sorted(dark_information.items(), key=lambda x: x[1], reverse=True)
    
    val.append(dark[0])
    test.append(dark[1])

    insects = sorted(insect_information.items(), key=lambda x: x[1], reverse=True)
    
    train.append(insects[0])
    
    # Shuffle to avoid category clumping
    random.seed(42)
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    # --- Summary ---
    n_train = sum(s[1] for s in train)
    n_val   = sum(s[1] for s in val)
    n_test  = sum(s[1] for s in test)
    
    total = n_train + n_val + n_test
    print(f"Train: {len(train)} scenes, {n_train} images ({n_train/total:.1%})")
    print(f"Val:   {len(val)} scenes,  {n_val} images ({n_val/total:.1%})")
    print(f"Test:  {len(test)} scenes, {n_test} images ({n_test/total:.1%})")``

    return train, val, test


def split_scenes(info_dict, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    total_images = sum(info_dict.values())
    targets = {
        "train": total_images * train_ratio,
        "val":   total_images * val_ratio,
        "test":  total_images * test_ratio,
    }
    counts = {"train": 0, "val": 0, "test": 0}
    splits = {"train": [], "val": [], "test": []}

    # Largest days first for stable greedy assignment
    days_sorted = sorted(info_dict.items(), key=lambda x: x[1], reverse=True)

    for name, count in days_sorted:
        # Pick the split with the largest remaining deficit
        chosen = max(targets.keys(), key=lambda s: targets[s] - counts[s])
        splits[chosen].append(name)
        counts[chosen] += count

    return splits["train"], splits["val"], splits["test"]
    
if __name__ == "__main__":
   pass 