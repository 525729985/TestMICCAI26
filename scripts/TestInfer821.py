import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import cupy as cp
import numpy as np
import torch
from pathlib import Path
import time
import gc

from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import SimpleITK as sitk

from cucim.core.operations.morphology import distance_transform_edt
from cucim.skimage.feature import peak_local_max
from cucim.skimage.measure import label
from skimage.segmentation import watershed


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

def check_spacing(input_image, target_spacing):
    original_spacing = input_image.GetSpacing()
    if all([round(i, 3) == j for i, j in zip(original_spacing, target_spacing)]):
        return False
    else:
        return True

def resample_image(
    input_image: sitk.Image,
    target_spacing: tuple,
    target_size: tuple | None = None,
    is_label: bool = False,
) -> sitk.Image:

    original_size = input_image.GetSize()
    original_spacing = input_image.GetSpacing()
    original_origin = input_image.GetOrigin()
    # if all([round(i, 3) == j for i, j in zip(original_spacing, target_spacing)]):
    #     return input_image
    if target_size is None:
        target_size = tuple(
            int(np.round(original_size[i] * original_spacing[i] / target_spacing[i]))
            for i in range(len(original_size))
        )

    print(f"resample_image spacing from {original_spacing} to {target_spacing} and size from {original_size} to {target_size}.")
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
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
        int(np.round(original_origin[i] * original_size[i] / target_size[i]))
        for i in range(len(original_origin))
    ]
    image.SetOrigin(new_origin)
    return image

def watershed_segmentation(mask, min_distance = 8):
    mask = cp.asarray(mask, dtype = cp.uint8)
    mask = mask > 0
    distance = distance_transform_edt(mask)

    peaks = peak_local_max(
        distance,
        min_distance = min_distance,
        labels = mask,
    )
    markers = cp.zeros_like(mask, dtype = cp.int32)
    markers[tuple(peaks.T)] = 1
    markers = label(markers)
    # seg = watershed(-distance, markers, mask = mask)
    seg = watershed(-distance.get(), markers.get(), mask = mask.get())

    return seg.astype(np.int8)

def get_pulp_id(tooth_id, is_mirror = False, pulp_diff = 32):
    pulp_id = tooth_id + pulp_diff
    if is_mirror:
        if 33 <= pulp_id < 41 or 49 <= pulp_id < 57:
            pulp_id += 8
        elif 41 <= pulp_id < 49 or 57 <= pulp_id < 65:
            pulp_id -= 8
    return pulp_id

def seg_by_water(source_np, ref_np, watershed_np, is_mirror_pulp = False, count_bg = True):
    labels = np.unique(watershed_np[watershed_np != 0])
    out_seg = np.zeros_like(source_np)
    count_mask = ref_np > 0
    for area_idx in labels:
        block_mask = watershed_np == area_idx
        if not count_bg:
            block_mask &= count_mask
        out_id = 0
        if block_mask.size > 0:
            ids, counts = np.unique(ref_np[block_mask], return_counts = True)
            out_id = ids[counts.argmax()]
        pulp_id = get_pulp_id(out_id, is_mirror = is_mirror_pulp)
        out_seg[block_mask] = np.where(source_np[block_mask] == 1, out_id, pulp_id)
    return out_seg

def read_image(file, target_spacing):
    origin_itk = itk_image = sitk.ReadImage(file)
    original_size = itk_image.GetSize()
    original_spacing = itk_image.GetSpacing()
    original_origin = itk_image.GetOrigin()
    original_direction = itk_image.GetDirection()

    is_reverse = itk_image.GetPixelID() == sitk.sitkFloat64
    itk_image = sitk.Cast(itk_image, sitk.sitkInt16)
    itk_image = sitk.Clamp(itk_image, upperBound = 5000)

    use_resample = check_spacing(itk_image, target_spacing)
    if use_resample:
        itk_image = resample_image(itk_image, target_spacing)
    if is_reverse:
        itk_image = rotate_volume(itk_image)
    img = sitk.GetArrayFromImage(itk_image)
    img = img[np.newaxis, :]

    props = {
        "sitk_stuff": {
            "use_resample": use_resample,
            "origin_itk": origin_itk,
            "spacing": original_spacing,
            "size": original_size,
            "origin": original_origin,
            "direction": original_direction,
        },
        "spacing": target_spacing,
    }
    return img, props, is_reverse

def write_seg(seg, output_name, properties, is_reverse):
    itk_image = sitk.GetImageFromArray(seg.astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy = False))
    if is_reverse:
        itk_image = rotate_volume(itk_image)
    origin_itk = properties["sitk_stuff"]["origin_itk"]

    itk_image.SetSpacing(properties["spacing"])
    itk_image.SetOrigin(properties["sitk_stuff"]["origin"])
    itk_image.SetDirection(properties["sitk_stuff"]["direction"])

    use_resample = properties["sitk_stuff"]["use_resample"]
    if use_resample:
        itk_image = resample_image(
            itk_image,
            target_spacing = properties["sitk_stuff"]["spacing"],
            target_size = properties["sitk_stuff"]["size"],
            is_label = True,
        )
    itk_image.CopyInformation(origin_itk)
    sitk.WriteImage(itk_image, output_name, True)

def init_predictors(cuda = 0):
    predictor916 = nnUNetPredictor(
        # tile_step_size = 0.5,
        tile_step_size = 0.8,
        use_gaussian = True,
        # use_mirroring = True,
        use_mirroring = False,
        perform_everything_on_device = True,
        device = torch.device("cuda", cuda),
        verbose = False,
        verbose_preprocessing = False,
        allow_tqdm = True,
    )
    predictor916.initialize_from_trained_model_folder(
        join(nnUNet_results, "Dataset916_MICCAIUnlabeled/nnUNetTrainer__nnUNetPlans__3d_new"),
        use_folds = (0,),
        checkpoint_name = "checkpoint_final.pth",
    )

    predictor821 = nnUNetPredictor(
        tile_step_size = 0.5,
        use_gaussian = True,
        use_mirroring = False,
        perform_everything_on_device = True,
        device = torch.device("cuda", cuda),
        verbose = False,
        verbose_preprocessing = False,
        allow_tqdm = True,
    )
    predictor821.initialize_from_trained_model_folder(
        join(nnUNet_results, "Dataset821_MICCAIUnlabeled/nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_new"),
        use_folds = (0,),
        checkpoint_name = "checkpoint_final.pth",
    )
    return predictor916, predictor821

def infer_by_predictor(predictor, img, props):
    result = predictor.predict_single_npy_array(img, props, None, None, False)
    return result

def infer_step(input_path, output_path, predictor916, predictor821):
    infer_spacing = (0.3, 0.3, 0.3)
    img, props, is_reverse = read_image(input_path, infer_spacing)

    ret916_np = infer_by_predictor(predictor916, img, props)

    gc.collect()
    torch.cuda.empty_cache()

    watershed_np = watershed_segmentation(ret916_np)
    ret821_np = infer_by_predictor(predictor821, img, props)

    out_np = seg_by_water(ret916_np, ret821_np, watershed_np, is_mirror_pulp = not is_reverse)
    write_seg(out_np, output_path, props, is_reverse)

def main(input_dir, output_dir, verbose = False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok = True, parents = True)

    predictor916, predictor821 = init_predictors()

    for file in input_dir.glob("*.nii.gz"):
        start_time = time.time()
        input_path = str(file)
        output_path = str(output_dir / file.name)
        infer_step(input_path, output_path, predictor916, predictor821)
        if verbose:
            print(f"{file.name} time {(time.time() - start_time):.2f}s")

if __name__ == "__main__":
    main(
        input_dir = "/home/data2/xrs/dataset/MICCAI-Chllenge-STS26-Task1/Validation/images",
        output_dir = "./pred_test5",
        verbose = True,
    )
    print("ok")
