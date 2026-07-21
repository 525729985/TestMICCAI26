import os
import multiprocessing
import numpy as np
from tqdm import tqdm

from pathlib import Path
import SimpleITK as sitk
import shutil

def resample_image(
    input_image: sitk.Image,
    target_spacing: tuple,
    is_label: bool = False,
) -> sitk.Image:
    original_size = input_image.GetSize()
    original_spacing = input_image.GetSpacing()
    original_origin = input_image.GetOrigin()
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

def process(input_path, output_path, is_label, target_spacing):
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

def main(in_dir, out_dir, is_label, target_spacing = (0.3, 0.3, 0.3), processes = 8):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok = True, parents = True)
    jobs = [
        {
            "input_path": str(file),
            "output_path": f"{str(out_dir / file.name)}",
            "is_label": is_label,
            "target_spacing": target_spacing,
        }
        for file in in_dir.glob("*.nii.gz")
    ]

    with multiprocessing.Pool(processes = processes) as pool:
        jobs = [pool.apply_async(process, kwds = {**args}) for args in jobs]
        _ = [job.get() for job in tqdm(jobs, desc = "Process Jobs")]

if __name__ == "__main__":
    target_spacing = (1.0, 1.0, 1.0)
    root = "/home/data2/xrs/nnUNet_raw/"
    input_ds = "Dataset1011_MICCAIUnlabeled"
    output_ds = "Dataset1211_MICCAIUnlabeled"

    main(
        in_dir = f"{root}/{input_ds}/imagesTr",
        out_dir = f"{root}/{output_ds}/imagesTr",
        is_label = False,
        target_spacing = target_spacing,
        processes = 8,
    )

    main(
        in_dir = f"{root}/{input_ds}/labelsTr",
        out_dir = f"{root}/{output_ds}/labelsTr",
        is_label = True,
        target_spacing = target_spacing,
        processes = 8,
    )
    shutil.copy(f"{root}/{input_ds}/dataset.json", f"{root}/{output_ds}/dataset.json")
    shutil.copy(f"{root}/{input_ds}/splits_final.json", f"{root}/{output_ds}/splits_final.json")
    print("ok")