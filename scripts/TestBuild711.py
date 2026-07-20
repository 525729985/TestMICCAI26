import os
import json
import multiprocessing

from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import shutil
from TestBuild611 import process_ds, mapping_DS616, mapping_DS621

def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def open_json(json_file):
    with open(json_file) as f:
        json_data = json.load(f)
        return json_data

def write_json(json_file: str, data, indent: int = 4):
    with open(json_file, "w") as f:
        json.dump(data, f, indent = indent)

def get_up_low_mapping():
    mapping = {}
    mapping.update({i: 1 for i in range(1, 17)})
    mapping.update({i: 2 for i in range(17, 33)})
    mapping.update({i: 3 for i in range(33, 49)})
    mapping.update({i: 4 for i in range(49, 65)})
    return mapping

def process(image_path, label_path, output_image_dir, output_label_dir, up_low_mapping):
    image_sitk = sitk.ReadImage(image_path)
    image_np = sitk.GetArrayFromImage(image_sitk)

    label_sitk = sitk.ReadImage(label_path)
    label_np = sitk.GetArrayFromImage(label_sitk)

    label_np_up_low = np.zeros_like(label_np, dtype = np.uint8)
    for org_id, new_id in up_low_mapping.items():
        label_np_up_low[label_np == org_id] = new_id
    if label_np_up_low[label_np_up_low == 3].size == 0:
        up_mask = label_np_up_low == 1
        image_np[up_mask] = 3500
    elif label_np_up_low[label_np_up_low == 4].size == 0:
        low_mask = label_np_up_low == 2
        image_np[low_mask] = 3500

    image_sitk_new = sitk.GetImageFromArray(image_np)
    image_sitk_new.CopyInformation(image_sitk)
    label_sitk_new = sitk.GetImageFromArray(label_np)
    label_sitk_new.CopyInformation(label_sitk)

    image_save_path = os.path.join(output_image_dir, os.path.basename(image_path))
    label_save_path = os.path.join(output_label_dir, os.path.basename(label_path))

    sitk.WriteImage(image_sitk_new, image_save_path)
    sitk.WriteImage(label_sitk_new, label_save_path)

def main(input_dir, output_dir, processes = 8):
    input_path = Path(input_dir)
    output_ds = input_path.name
    input_image_dir = (input_path / "imagesTr")
    input_label_dir = (input_path / "labelsTr")
    
    output_path = Path(output_dir)
    output_image_dir = (output_path / "imagesTr")
    output_label_dir = (output_path / "labelsTr")

    output_path.mkdir(exist_ok = True)
    output_image_dir.mkdir(exist_ok = True)
    output_label_dir.mkdir(exist_ok = True)

    data_list = list(input_image_dir.glob("*.nii.gz"))
    with multiprocessing.Pool(processes = processes) as pool:
        jobs = [
            pool.apply_async(process, args = [
                str(file),
                str(input_label_dir / file.name.replace("_0000", "")),
                str(output_image_dir),
                str(output_label_dir),
                get_up_low_mapping(),
            ])
            for file in data_list
        ]
        for job in tqdm(jobs, desc = "Process Jobs"):
            _ = job.get()

    dataset_json = load_json(str(input_path / "dataset.json"))
    dataset_json["file_ending"] = ".nii.gz"
    dataset_json["name"] = output_ds
    dataset_json["numTraining"] = len(data_list)

    write_json(str(output_path / "dataset.json"), dataset_json)

    split_input_path = str(input_path / "splits_final.json")
    split_output_path = str(output_path / "splits_final.json")
    shutil.copyfile(split_input_path, split_output_path)
    print("ok")
if __name__ == "__main__":
    root = "/home/data2/xrs/nnUNet_raw/"

    # Dataset 711 & 911
    # For half-mouth cases labeled only with teeth and lacking pulp labels: set CT density value to 3500 to simulate implant teeth.
    input_dir = root + "Dataset611_MICCAILabeled"
    output_dir = root + "Dataset711_MICCAILabeled"
    main(input_dir, output_dir)

    input_ds = "Dataset611_MICCAILabeled"
    output_ds = "Dataset621_MICCAILabeled"
    process_ds(
        root = root,
        out_root = root,
        input_ds = input_ds,
        output_ds = output_ds,
        mapping = mapping_DS621(),
        image_link = input_ds,
    )

    input_ds = "Dataset711_MICCAILabeled"
    output_ds = "Dataset716_MICCAILabeled"
    process_ds(
        root = root,
        out_root = root,
        input_ds = input_ds,
        output_ds = output_ds,
        mapping = mapping_DS616(),
        image_link = input_ds,
    )