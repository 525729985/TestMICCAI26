import os
import json
import multiprocessing

from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
from pathlib import Path

def mapping_DSMirror():
    mapping = {}
    mapping.update({i: i + 8 for i in range(1, 9)})
    mapping.update({i: i - 8 for i in range(9, 17)})
    mapping.update({i: i + 8 for i in range(17, 25)})
    mapping.update({i: i - 8 for i in range(25, 33)})
    return mapping

def process(data_input, data_output, mapping):
    label_sitk = sitk.ReadImage(data_input)
    label_np = sitk.GetArrayFromImage(label_sitk)

    label_np_new = np.zeros_like(label_np, dtype = np.uint8)
    for org_id, new_id in mapping.items():
        label_np_new[label_np == org_id] = new_id

    label_sitk_new = sitk.GetImageFromArray(label_np_new)
    label_sitk_new.CopyInformation(label_sitk)
    sitk.WriteImage(label_sitk_new, data_output)


def main(input_dir, output_dir, processes = 8):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)

    mapping = mapping_DSMirror()
    with multiprocessing.Pool(processes = processes) as pool:
        jobs = [
            pool.apply_async(process, args = [
                str(file),
                str(output_dir / file.name),
                mapping,
            ])
            for file in input_dir.glob("*.nii.gz")
        ]
        for job in tqdm(jobs, desc="Process Jobs"):
            _ = job.get()
    print("ok")

if __name__ == "__main__":
    # Validation Train-Unlabeled Train-Labeled
    input_dir = "/home/data2/xrs/dataset/pred_out/pred0/Validation250"
    output_dir = "/home/data2/xrs/dataset/pred_out/pred0/Validation250"
    main(input_dir, output_dir)