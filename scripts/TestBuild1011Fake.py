import multiprocessing
import os
from os.path import join
import json
import shutil

import random
import pandas as pd
from tqdm import tqdm

def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def write_json(json_file, data, indent = 4):
    with open(json_file, "w") as f:
        json.dump(data, f, indent = indent)

def read_fake_names(file):
    names = []
    data = pd.read_csv(file)
    data = data.iloc[: -1]
    for idx, row in data.iterrows():
        name = row["name"]
        error_num = sum(1 for i in range(1, 65) for t in ["d", "h", "w"] if row[f"size_{i}_{t}"] > 100)
        error_tooth_num = sum(1 for i in range(1, 33) for t in ["d", "h", "w"] if row[f"tooth_size_{i}_{t}"] > 100)
        error_block_w_num = sum(1 for i in range(1, 5) if row[f"block_size_{i}_w"] > 200)
        error_block_z_num = sum(1 for i in range(1, 5) if row[f"block_size_{i}_d"] > 150)
        error_block_num = error_block_w_num + error_block_z_num
        if error_num == 0 and error_tooth_num == 0 and error_block_num == 0:
            names.append(name)
    print(len(names))
    return names

def process_data(name, input_image_dir, input_label_dir, output_dir):
    name = name.replace(".nii.gz", "")
    image_name = name + "_0000.nii.gz"
    name = name + ".nii.gz"
    shutil.copyfile(join(input_image_dir, image_name), join(output_dir, "imagesTr", image_name))
    shutil.copyfile(join(input_label_dir, name), join(output_dir, "labelsTr", name))

def process_ds(root, out_root, input_ds, output_ds, labeled_ds, ref_fake = None, ref_fake_csv = "size.csv", processes = 8):
    output_dir = join(out_root, output_ds)
    output_image_dir = join(out_root, output_ds, "imagesTr")
    output_label_dir = join(out_root, output_ds, "labelsTr")
    os.makedirs(output_dir, exist_ok = True)
    os.makedirs(output_image_dir, exist_ok = True)
    os.makedirs(output_label_dir, exist_ok = True)
    if ref_fake is not None:
        names = read_fake_names(join(ref_fake, ref_fake_csv))
        input_image_dir = join(root, input_ds, "imagesTr")
        input_label_dir = ref_fake
        with multiprocessing.Pool(processes = processes) as pool:
            jobs = [
                pool.apply_async(process_data, args = [
                    name,
                    input_image_dir,
                    input_label_dir,
                    output_dir,
                ])
                for name in names
            ]
            for job in tqdm(jobs, desc = "Process Jobs"):
                _ = job.get()

    img_files = os.listdir(output_image_dir)
    img_names = [file.replace("_0000.nii.gz", "") for file in img_files]

    labeled_num = 0
    if labeled_ds is not None:
        labeled_split_json = load_json(join(out_root, labeled_ds, "splits_final.json"))[0]
        labeled_list = labeled_split_json["train"] + labeled_split_json["val"]
        with multiprocessing.Pool(processes = processes) as pool:
            input_image_dir = join(out_root, labeled_ds, "imagesTr")
            input_label_dir = join(out_root, labeled_ds, "labelsTr")
            jobs = [
                pool.apply_async(process_data, args = [
                    name,
                    input_image_dir,
                    input_label_dir,
                    output_dir,
                ])
                for name in labeled_list
            ]
            for job in tqdm(jobs, desc = "Process Jobs"):
                _ = job.get()
        labeled_num = len(labeled_list)

    dataset_json = load_json(join(root, input_ds, "dataset.json"))
    dataset_json["file_ending"] = ".nii.gz"
    dataset_json["name"] = output_ds
    dataset_json["numTraining"] = len(img_names) + labeled_num

    write_json(join(out_root, output_ds, "dataset.json"), dataset_json)

    random_seed = 42
    random.seed(random_seed)
    random.shuffle(img_names)

    # split_index = int(len(img_names) * 0.9)
    # train_files = img_names[:split_index]
    # val_files = img_names[split_index:]

    train_files = img_names
    val_files = []
    train_files.sort()
    val_files.sort()

    if labeled_ds is not None:
        split = [{"train": train_files + labeled_split_json["train"], "val": val_files + labeled_split_json["val"]}]
    else:
        split = [{"train": train_files, "val": val_files}]
    write_json(join(out_root, output_ds, "splits_final.json"), split)

process_ds(
    root = "/home/data2/xrs/dataset/MICCAI-Chllenge-STS26-Task1_S3/",
    out_root = "/home/data2/xrs/nnUNet_raw/",
    input_ds = "Train-Unlabeled",
    labeled_ds = "Dataset611_MICCAILabeled",
    output_ds = "Dataset1011_MICCAILabeled",
    ref_fake = "/home/data2/xrs/dataset/pred_out/pred2/Train-Unlabeled1000_1000/",
    ref_fake_csv = "size.csv",
)
