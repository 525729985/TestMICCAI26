from typing import Dict, Any
import os
from os.path import join
import json
import random
import multiprocessing
import shutil

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

def mapping_DS616() -> Dict[int, int]:
    mapping = {}
    mapping.update({i: 1 for i in range(1, 33)})
    mapping.update({i: 2 for i in range(33, 65)})
    return mapping

def mapping_DS626() -> Dict[int, int]:
    mapping = {}
    mapping.update({i: 1 for i in range(1, 65)})
    return mapping

def mapping_DS621() -> Dict[int, int]:
    mapping = {}
    mapping.update({i: i for i in range(1, 33)})
    mapping.update({i: i - 32 for i in range(33, 65)})
    return mapping

def load_json(json_file: str) -> Any:
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def write_json(json_file: str, data: Any, indent: int = 4) -> None:
    with open(json_file, "w") as f:
        json.dump(data, f, indent = indent)
def image_to_nifi(input_path: str, output_path: str) -> None:
    image_sitk = sitk.ReadImage(input_path)
    image_sitk = sitk.Clamp(image_sitk, upperBound = 5000)
    image_sitk = sitk.Cast(image_sitk, sitk.sitkInt16)
    sitk.WriteImage(image_sitk, output_path)

def label_mapping(input_path: str, output_path: str, mapping: Dict[int, int] = None) -> None:
    label_sitk = sitk.ReadImage(input_path)
    if mapping is not None:
        label_np = sitk.GetArrayFromImage(label_sitk)

        label_np_new = np.zeros_like(label_np, dtype = np.uint8)
        for org_id, new_id in mapping.items():
            label_np_new[label_np == org_id] = new_id

        label_sitk_new = sitk.GetImageFromArray(label_np_new)
        label_sitk_new.CopyInformation(label_sitk)
        sitk.WriteImage(label_sitk_new, output_path)
    else:
        sitk.WriteImage(label_sitk, output_path)

def process_images(files: str, img_dir_in: str, img_dir_out: str, n_processes: int = 12):

    os.makedirs(img_dir_out, exist_ok = True)

    iterable = [
        {
            "input_path": join(img_dir_in, file),
            "output_path": join(img_dir_out, file.replace(".mha", ".nii.gz")),
        }
        for file in files
    ]
    with multiprocessing.Pool(processes = n_processes) as pool:
        jobs = [pool.apply_async(image_to_nifi, kwds = {**args}) for args in iterable]
        _ = [job.get() for job in tqdm(jobs, desc = "Process Images")]


def process_labels(
    files: str, lbl_dir_in: str, lbl_dir_out: str, mapping: Dict[int, int], n_processes: int = 12
) -> None:

    os.makedirs(lbl_dir_out, exist_ok = True)

    iterable = [
        {
            "input_path": join(lbl_dir_in, file),
            "output_path": join(lbl_dir_out, file.replace(".mha", ".nii.gz")),
            "mapping": mapping,
        }
        for file in files
    ]
    with multiprocessing.Pool(processes = n_processes) as pool:
        jobs = [pool.apply_async(label_mapping, kwds = {**args}) for args in iterable]
        _ = [job.get() for job in tqdm(jobs, desc = "Process Labels...")]


def process_ds(
    root: str, out_root: str, input_ds: str, output_ds: str, mapping: dict, image_link: str = None
) -> None:
    os.makedirs(join(out_root, output_ds), exist_ok = True)
    os.makedirs(join(out_root, output_ds, "labelsTr"), exist_ok = True)
    # --- Handle Labels --- #
    lbl_files = os.listdir(join(root, input_ds, "labelsTr"))
    lbl_dir_in = join(root, input_ds, "labelsTr")
    lbl_dir_out = join(out_root, output_ds, "labelsTr")

    process_labels(lbl_files, lbl_dir_in, lbl_dir_out, mapping, n_processes = 12)

    # --- Handle Images --- #
    img_files = os.listdir(join(root, input_ds, "imagesTr"))
    dataset = {}
    if image_link is None:
        img_dir_in = join(root, input_ds, "imagesTr")
        img_dir_out = join(out_root, output_ds, "imagesTr")

        process_images(img_files, img_dir_in, img_dir_out, n_processes = 12)
    else:
        base_name = [file.replace("_0000.mha", "").replace("_0000.nii.gz", "") for file in img_files]
        for name in base_name:
            dataset[name] = {
                "images": [join("..", image_link, "imagesTr", name + "_0000.nii.gz")],
                "label": join("labelsTr", name + ".nii.gz"),
            }

    # --- Generate dataset.json --- #
    dataset_json = load_json(join(root, input_ds, "dataset.json"))
    dataset_json["file_ending"] = ".nii.gz"
    dataset_json["name"] = output_ds
    dataset_json["numTraining"] = len(lbl_files)
    if dataset !=  {}:
        dataset_json["dataset"] = dataset

    label_dict = dataset_json["labels"]
    label_dict_new = {"background": 0}
    for k, v in label_dict.items():
        if v in mapping.keys() and k not in label_dict_new:
            label_dict_new[k] = mapping[v]
    dataset_json["labels"] = label_dict_new
    write_json(join(out_root, output_ds, "dataset.json"), dataset_json)

    # --- Generate splits_final.json --- #
    split_input_path = join(root, input_ds, "splits_final.json")
    split_output_path = join(out_root, output_ds, "splits_final.json")
    if os.path.isfile(split_input_path):
        shutil.copyfile(split_input_path, split_output_path)
    else:
        img_names = [file.replace("_0000.mha", "").replace("_0000.nii.gz", "") for file in img_files]

        random_seed = 42
        random.seed(random_seed)
        random.shuffle(img_names)

        split_index = int(len(img_names) * 0.7)  # 70:30 split
        train_files = img_names[:split_index]
        val_files = img_names[split_index:]
        train_files.sort()
        val_files.sort()

        split = [{"train": train_files, "val": val_files}]
        write_json(split_output_path, split)


if __name__ ==  "__main__":
    input = "/home/data2/xrs/nnUNet_raw/"
    output = "/home/data2/xrs/nnUNet_raw/"

    # input_ds = "Dataset1011_MICCAILabeled"
    # output_ds = "Dataset1021_MICCAILabeled"
    # process_ds(
    #     root = input,
    #     out_root = output,
    #     input_ds = input_ds,
    #     output_ds = output_ds,
    #     mapping = mapping_DS621(),
    #     image_link = input_ds,
    # )

    input_ds = "Dataset1111_MICCAIUnlabeled"
    output_ds = "Dataset1121_MICCAIUnlabeled"
    process_ds(
        root = input,
        out_root = output,
        input_ds = input_ds,
        output_ds = output_ds,
        mapping = mapping_DS621(),
        image_link = input_ds,
    )
