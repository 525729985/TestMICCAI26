import multiprocessing
import csv

import numpy as np
from tqdm import tqdm

import SimpleITK as sitk
from pathlib import Path

def save_to_csv(names, header, datas, csv_path = "output.csv"):
    with open(csv_path, mode = "w", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for name, data in zip(names, datas):
            writer.writerow([name] + data)
    print("ok")

def get_origin_str(direction):
    return sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(direction)
def process(file):
    row = []
    data_sitk = sitk.ReadImage(file)
    row.extend(data_sitk.GetSize())
    row.extend(data_sitk.GetSpacing())
    row.extend(data_sitk.GetOrigin())
    row += [data_sitk.GetPixelIDTypeAsString(), data_sitk.GetDirection(), get_origin_str(data_sitk.GetDirection())]
    return file.name, row

def main(input_dir, output_dir = "./", processes = 8):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    names = []
    datas = []

    with multiprocessing.Pool(processes = processes) as pool:
        jobs = [
            pool.apply_async(process, args = [file])
            for file in input_path.glob("*.nii.gz")
        ]
        for job in tqdm(jobs, desc = "Process Jobs"):
            name, data = job.get()
            names.append(name)
            datas.append(data)
    header = ["name"]
    header += [f"size_d", f"size_h", f"size_w"]
    header += [f"spacing_d", f"spacing_h", f"spacing_w"]
    header += [f"origin_d", f"origin_h", f"origin_w"]
    header += [f"pixel_type", "direction", "direction_type"]
    save_to_csv(names, header, datas, str(output_path / "info.csv"))
if __name__ == "__main__":
    label_path = "/home/data2/xrs/dataset/MICCAI-Chllenge-STS26-Task1/Validation/images"
    main(label_path)

