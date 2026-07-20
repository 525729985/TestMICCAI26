import os
import json
import multiprocessing
import numpy as np
from tqdm import tqdm

import random
from pathlib import Path
import SimpleITK as sitk

from scipy.ndimage import distance_transform_edt

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label

def write_json(json_file: str, data, indent: int = 4):
    with open(json_file, "w") as f:
        json.dump(data, f, indent = indent)

def write_dataset_json(output_ds, num_training, output_path):
    new_json = {
        "file_ending": ".nii.gz",
        "name": output_ds,
        "labels": {
            "background": 0,
            "Upper Right Central Incisor": 1,
            "Upper Right Lateral Incisor": 2,
            "Upper Right Canine": 3,
            "Upper Right First Premolar": 4,
            "Upper Right Second Premolar": 5,
            "Upper Right First Molar": 6,
            "Upper Right Second Molar": 7,
            "Upper Right Third Molar (Wisdom Tooth)": 8,
            "Upper Left Central Incisor": 9,
            "Upper Left Lateral Incisor": 10,
            "Upper Left Canine": 11,
            "Upper Left First Premolar": 12,
            "Upper Left Second Premolar": 13,
            "Upper Left First Molar": 14,
            "Upper Left Second Molar": 15,
            "Upper Left Third Molar (Wisdom Tooth)": 16,
            "Lower Left Central Incisor": 17,
            "Lower Left Lateral Incisor": 18,
            "Lower Left Canine": 19,
            "Lower Left First Premolar": 20,
            "Lower Left Second Premolar": 21,
            "Lower Left First Molar": 22,
            "Lower Left Second Molar": 23,
            "Lower Left Third Molar (Wisdom Tooth)": 24,
            "Lower Right Central Incisor": 25,
            "Lower Right Lateral Incisor": 26,
            "Lower Right Canine": 27,
            "Lower Right First Premolar": 28,
            "Lower Right Second Premolar": 29,
            "Lower Right First Molar": 30,
            "Lower Right Second Molar": 31,
            "Lower Right Third Molar (Wisdom Tooth)": 32,
            "Upper Right Central Incisor Pulp": 33,
            "Upper Right Lateral Incisor Pulp": 34,
            "Upper Right Canine Pulp": 35,
            "Upper Right First Premolar Pulp": 36,
            "Upper Right Second Premolar Pulp": 37,
            "Upper Right First Molar Pulp": 38,
            "Upper Right Second Molar Pulp": 39,
            "Upper Right Third Molar (Wisdom Tooth) Pulp": 40,
            "Upper Left Central Incisor Pulp": 41,
            "Upper Left Lateral Incisor Pulp": 42,
            "Upper Left Canine Pulp": 43,
            "Upper Left First Premolar Pulp": 44,
            "Upper Left Second Premolar Pulp": 45,
            "Upper Left First Molar Pulp": 46,
            "Upper Left Second Molar Pulp": 47,
            "Upper Left Third Molar (Wisdom Tooth) Pulp": 48,
            "Lower Left Central Incisor Pulp": 49,
            "Lower Left Lateral Incisor Pulp": 50,
            "Lower Left Canine Pulp": 51,
            "Lower Left First Premolar Pulp": 52,
            "Lower Left Second Premolar Pulp": 53,
            "Lower Left First Molar Pulp": 54,
            "Lower Left Second Molar Pulp": 55,
            "Lower Left Third Molar (Wisdom Tooth) Pulp": 56,
            "Lower Right Central Incisor Pulp": 57,
            "Lower Right Lateral Incisor Pulp": 58,
            "Lower Right Canine Pulp": 59,
            "Lower Right First Premolar Pulp": 60,
            "Lower Right Second Premolar Pulp": 61,
            "Lower Right First Molar Pulp": 62,
            "Lower Right Second Molar Pulp": 63,
            "Lower Right Third Molar (Wisdom Tooth) Pulp": 64
        },
        "channel_names": {
            "0": "CBCT"
        },
        "numTraining": num_training,
    }
    write_json(output_path, new_json)

def resample_image(
    input_image: sitk.Image,
    target_spacing: tuple,
    is_label: bool = False,
) -> sitk.Image:
    original_size = input_image.GetSize()
    original_spacing = input_image.GetSpacing()
    original_origin = input_image.GetOrigin()
    if all([i == j for i, j in zip(original_spacing, target_spacing)]):
        return input_image
    new_size = [
        int(np.round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(len(original_size))
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(input_image.GetOrigin())
    resampler.SetOutputDirection(input_image.GetDirection())
    resampler.SetTransform(sitk.Transform())

    if is_label:
        resampler.SetInterpolator(sitk.sitkLabelLinear)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(input_image.GetPixelIDValue())
    image = resampler.Execute(input_image)

    new_origin = [
        int(np.round(original_origin[i] * original_spacing[i] / target_spacing[i]))
        for i in range(len(original_origin))
    ]
    image.SetOrigin(new_origin)
    return image

def rotate_volume(volume):
    array = sitk.GetArrayFromImage(volume)
    rotated_array = array
    rotated_array = np.flip(rotated_array, axis = (0,))
    rotated_volume = sitk.GetImageFromArray(rotated_array)

    # Copy basic information (spacing, origin)
    rotated_volume.SetSpacing(volume.GetSpacing())
    rotated_volume.SetOrigin(volume.GetOrigin())
    # print('OriginalDirection:', rotated_volume.GetDirection())
    lps_direction = (1.0, 0.0, 0.0,  # X axis: Right to Left (negative)
                     0.0, 1.0, 0.0,  # Y axis: Anterior to Posterior (negative)
                     0.0, 0.0, 1.0)  # Z axis: Inferior to Superior (positive)
    rotated_volume.SetDirection(lps_direction)
    return rotated_volume

def process_image(input_path, output_path, is_label, target_spacing):
    image = sitk.ReadImage(input_path)
    if not is_label:
        image = sitk.Cast(image, sitk.sitkInt16)
        image = sitk.Clamp(image, upperBound = 5000)
    else:
        image = sitk.Cast(image, sitk.sitkInt8)
    image = resample_image(
        image,
        is_label = is_label,
        target_spacing = target_spacing,
    )
    sitk.WriteImage(image, output_path)
def get_by_adj_tooth(label_np, pulp_id):
    mask = label_np == pulp_id
    z_idx, y_idx, x_idx = np.nonzero(mask)
    max_z_idx = np.argmax(z_idx)

    # 上侧或下侧取5x5片
    max_z_coord = [z_idx[max_z_idx], y_idx[max_z_idx], x_idx[max_z_idx]]
    max_z_around = label_np[max_z_coord[0], max_z_coord[1] - 2:  max_z_coord[1] + 3, max_z_coord[2] - 2: max_z_coord[2] + 3]
    ids, counts = np.unique(max_z_around[max_z_around < 33], return_counts = True)
    id = ids[counts.argmax()] + 32
    if id > 32:
        return id
    min_z_idx = np.argmin(z_idx)
    min_z_coord = [z_idx[min_z_idx], y_idx[min_z_idx], x_idx[min_z_idx]]
    min_z_around = label_np[min_z_coord[0], min_z_coord[1] - 2:  min_z_coord[1] + 3, min_z_coord[2] - 2: min_z_coord[2] + 3]
    ids, counts = np.unique(min_z_around[min_z_around < 33], return_counts = True)
    id = ids[counts.argmax()] + 32
    return id

def watershed_segmentation(mask, min_distance = 8):
    binary = mask > 0
    distance = distance_transform_edt(binary)

    peaks = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=binary,
    )
    markers = np.zeros_like(binary, dtype=int)
    markers[tuple(peaks.T)] = 1
    markers = label(markers)
    seg = watershed(-distance, markers, mask=binary)

    return seg.astype(np.int8)

def mapping616(label_np):
    mapping = {}
    mapping.update({i: 1 for i in range(1, 32)})
    mapping.update({i: 2 for i in range(33, 65)})
    label_np_new = np.zeros_like(label_np, dtype=np.uint8)
    for org_id, new_id in mapping.items():
        label_np_new[label_np == org_id] = new_id
    return label_np_new

def fix_by_water(label_np, count_bg = True, pulp_diff = 32):
    # 修复神经编号错误或牙齿内有多编号神经, 神经编号统一按tooth_id+32训练
    source_np = mapping616(label_np)
    watershed = watershed_segmentation(source_np)
    labels = np.unique(watershed[watershed != 0])
    out_seg = np.zeros_like(source_np)
    count_mask = label_np > 0
    for area_idx in labels:
        block_mask = watershed == area_idx
        if not count_bg:
            block_mask &= count_mask
        out_id = 0
        if block_mask.size > 0:
            ids, counts = np.unique(label_np[block_mask], return_counts = True)
            out_id = ids[counts.argmax()]
        if out_id == 0 or out_id > 32:
            print(f"fix error {out_id}.")
        out_seg[block_mask] = np.where(source_np[block_mask] == 1, out_id, out_id + pulp_diff)
    for tooth_id in range(1, 33):
        out_seg[label_np == tooth_id] = tooth_id
    return out_seg

def process(image_path, label_path, out_image_dir, out_label_dir, target_spacing):
    name = image_path.name
    image_sitk = sitk.ReadImage(str(image_path))
    # Fix vertical inversion for alignment consistency. but is the left-right direction also flipped?
    is_reverse = image_sitk.GetPixelID() == sitk.sitkFloat64

    image_sitk = sitk.Cast(image_sitk, sitk.sitkInt16)
    image_sitk = sitk.Clamp(image_sitk, upperBound = 5000)
    image_sitk = resample_image(
        image_sitk,
        is_label = False,
        target_spacing = target_spacing,
    )
    if is_reverse:
        image_sitk = rotate_volume(image_sitk)
    image_name = name.replace(".nii.gz", "_0000.nii.gz").replace("_with-artifacts", "")
    sitk.WriteImage(image_sitk, str(out_image_dir / image_name))

    if label_path is None:
        return
    label_sitk = sitk.ReadImage(str(label_path))
    label_sitk = sitk.Cast(label_sitk, sitk.sitkInt8)
    label_sitk = resample_image(
        label_sitk,
        is_label = True,
        target_spacing = target_spacing,
    )
    if is_reverse:
        # print(f"fixed reversed image&label {name}.")
        label_sitk = rotate_volume(label_sitk)
    else:
        label_np = sitk.GetArrayFromImage(label_sitk)
        # 统一按周围牙齿使用tooth_id+32修复神经编号
        label_np_new = fix_by_water(label_np, count_bg = False)

        label_sitk_new = sitk.GetImageFromArray(label_np_new)
        label_sitk_new.CopyInformation(label_sitk)
        label_sitk = label_sitk_new

    label_name = name.replace("_with-artifacts", "")
    sitk.WriteImage(label_sitk, str(out_label_dir / label_name))


def main(in_image_dir: str, in_label_dir: str | None, output_dir: str, target_spacing: tuple = (0.3, 0.3, 0.3), processes: int = 8):
    in_image_dir = Path(in_image_dir)
    output_dir = Path(output_dir)
    out_image_dir = output_dir / "imagesTr"
    out_label_dir = output_dir / "labelsTr"

    out_image_dir.mkdir(exist_ok = True, parents = True)
    images = list(in_image_dir.glob("*.nii.gz"))

    if in_label_dir is not None:
        in_label_dir = Path(in_label_dir)

    out_label_dir.mkdir(exist_ok = True)
    num_training = len(images)
    write_dataset_json(output_dir.name, num_training, str(output_dir / "dataset.json"))
    with multiprocessing.Pool(processes = processes) as pool:
        jobs = [
            pool.apply_async(
                process,
                args = (
                    file,
                    (in_label_dir / file.name) if in_label_dir is not None else None,
                    out_image_dir,
                    out_label_dir,
                    target_spacing,
                )
            )
            for file in images
        ]
        _ = [job.get() for job in tqdm(jobs, desc = "Process Jobs")]

if __name__ == "__main__":
    target_spacing = (0.3, 0.3, 0.3)

    main(
        in_image_dir = "/home/data2/xrs/dataset/MICCAI-Chllenge-STS26-Task1/Train-Labeled/images",
        in_label_dir = "/home/data2/xrs/dataset/MICCAI-Chllenge-STS26-Task1/Train-Labeled/labels",
        output_dir = "/home/data2/xrs/dataset/MICCAI-Chllenge-STS26-Task1_S3/Train-Labeled",
        target_spacing = target_spacing,
    )
    main(
        in_image_dir = "/home/data2/xrs/dataset/MICCAI-Chllenge-STS26-Task1/Train-Unlabeled/images",
        in_label_dir = None,
        output_dir = "/home/data2/xrs/dataset/MICCAI-Chllenge-STS26-Task1_S3/Train-Unlabeled",
        target_spacing = target_spacing,
    )
    main(
        in_image_dir = "/home/data2/xrs/dataset/MICCAI-Chllenge-STS26-Task1/Validation/images",
        in_label_dir = None,
        output_dir = "/home/data2/xrs/dataset/MICCAI-Chllenge-STS26-Task1_S3/Validation",
        target_spacing = target_spacing,
    )
    print("ok")