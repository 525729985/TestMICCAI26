import multiprocessing
import csv

import numpy as np
from tqdm import tqdm

import SimpleITK as sitk
from pathlib import Path

def save_to_csv(names, datas, out_channels, csv_path = "output.csv"):
    header = ["name"]
    for i in out_channels:
        header += [f"size_{i}_d", f"size_{i}_h", f"size_{i}_w"]
    for i in range(1, 33):
        header += [f"tooth_size_{i}_d", f"tooth_size_{i}_h", f"tooth_size_{i}_w"]
    for i in range(1, 5):
        header += [f"block_size_{i}_d", f"block_size_{i}_h", f"block_size_{i}_w"]
    with open(csv_path, mode = "w", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for name, data in zip(names, datas):
            writer.writerow([name] + data)
        writer.writerow(["total_max"] + np.max(datas, axis = 0).tolist())
    print("ok")

def calc_size(np_arr):
    if np_arr.size == 0:
        return [0, 0, 0]
    no_zero = np.nonzero(np_arr)
    return [(no_zero[axis].max() - no_zero[axis].min()) if no_zero[axis].size > 0 else 0 for axis in range(3)]
def mask_tooth(np_arr, tooth_id):
    output = (np_arr == tooth_id) | (np_arr == tooth_id + 32)
    return output
def mask_quarter(np_arr, block_id):
    output = np.zeros_like(np_arr, dtype = np.bool)
    start_tooth_idx = (block_id - 1) * 8 + 1
    start_pulp_idx = start_tooth_idx + 32
    for i in range(start_tooth_idx, start_tooth_idx + 8):
        output |= (np_arr == i)
    for i in range(start_pulp_idx, start_pulp_idx + 8):
        output |= (np_arr == i)
    return output

def process(file, out_channels):
    row = []
    label_sitk = sitk.ReadImage(file)
    np_labels = sitk.GetArrayFromImage(label_sitk)
    size_list = []
    for i in out_channels:
        item_size_list = calc_size(np_labels == i)
        size_list.extend(item_size_list)
    for tooth_idx in range(1, 33):
        item_size_list = calc_size(mask_tooth(np_labels, tooth_idx))
        size_list.extend(item_size_list)
    for block_idx in range(1, 5):
        item_size_list = calc_size(mask_quarter(np_labels, block_idx))
        size_list.extend(item_size_list)
    row += size_list
    return file.name, row

def main(input_dir, output_dir = "./", processes = 8):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    out_channels = range(1, 65)
    names = []
    datas = []

    with multiprocessing.Pool(processes = processes) as pool:
        jobs = [
            pool.apply_async(process, args = [file, out_channels])
            for file in input_path.glob("*.nii.gz")
        ]
        for job in tqdm(jobs, desc = "Process Jobs"):
            name, data = job.get()
            names.append(name)
            datas.append(data)
    save_to_csv(names, datas, out_channels, str(output_path / "size.csv"))
if __name__ == "__main__":
    # label_path = "/home/data2/xrs/nnUNet_raw/Dataset611_MICCAILabeled/labelsTr/"
    # main(label_path)

    label_path = "/home/data2/xrs/dataset/pred_out/pred/Train-Unlabeled250_250/"
    main(label_path, label_path)

