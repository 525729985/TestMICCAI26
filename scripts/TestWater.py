import multiprocessing
import numpy as np
from tqdm import tqdm
from pathlib2 import Path

import SimpleITK as sitk
import numpy as np

from scipy.ndimage import distance_transform_edt

import cupy as cp
from cucim.core.operations.morphology import distance_transform_edt
from cucim.skimage.feature import peak_local_max
from cucim.skimage.measure import label
from skimage.segmentation import watershed

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
    seg = watershed(-distance.get(), markers.get(), mask = mask.get())

    return seg.astype(np.int8)

def seg_by_water(source_file, target_file, ref_file, small_max_size, count_bg = True, pulp_diff = 32):
    source_itk = sitk.ReadImage(source_file)
    source_np = sitk.GetArrayFromImage(source_itk)

    ref_itk = sitk.ReadImage(ref_file)
    ref_np = sitk.GetArrayFromImage(ref_itk)
    
    watershed = watershed_segmentation(source_np, small_max_size = small_max_size)
    labels = np.unique(watershed[watershed != 0])
    out_seg = np.zeros_like(source_np)
    count_mask = ref_np > 0
    for area_idx in labels:
        block_mask = watershed == area_idx
        if not count_bg:
            block_mask &= count_mask
        out_id = 0
        if block_mask.size > 0:
            ids, counts = np.unique(ref_np[block_mask], return_counts = True)
            out_id = ids[counts.argmax()]
        out_seg[block_mask] = np.where(source_np[block_mask] == 1, out_id, out_id + pulp_diff)

    out_itk = sitk.GetImageFromArray(out_seg)
    out_itk.CopyInformation(source_itk)
    sitk.WriteImage(out_itk, target_file)

def main(input_dir, output_dir, ref_dir = None, small_max_size = 0, processes = 8):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    ref_dir = Path(ref_dir)

    output_dir.mkdir(exist_ok = True, parents = True)

    with multiprocessing.Pool(processes = processes) as pool:
        jobs = [
            pool.apply_async(
                seg_by_water,
                args = (
                    str(file),
                    str(output_dir / file.name),
                    str(ref_dir / file.name),
                    small_max_size,
                )
            )
            for file in input_dir.glob("*.nii.gz")
        ]
        _ = [job.get() for job in tqdm(jobs, desc = "Process Jobs")]

if __name__ == "__main__":
    # type_num1 = "250"
    # type_num2 = "500"

    # type = "Train-Labeled"
    # input = f"/home/data2/xrs/dataset/pred_out/pred_716/{type}{type_num1}"
    # output = f"./pred/{type}{type_num1}_{type_num2}"
    # ref = f"/home/data2/xrs/dataset/pred_out/pred_621/{type}{type_num2}"
    # main(input, output, ref, small_max_size = 3000, processes = 8)

    # type = "Train-Unlabeled"
    # input = f"/home/data2/xrs/dataset/pred_out/pred_716/{type}{type_num1}"
    # output = f"/home/data2/xrs/dataset/pred_out/pred/{type}{type_num1}_{type_num2}"
    # ref = f"/home/data2/xrs/dataset/pred_out/pred_621/{type}{type_num2}"
    # main(input, output, ref, small_max_size = 3000, processes = 8)

    # type = "Validation"
    # input = f"/home/data2/xrs/dataset/pred_out/pred_716/{type}{type_num1}"
    # output = f"./pred/{type}{type_num1}_{type_num2}"
    # ref = f"/home/data2/xrs/dataset/pred_out/pred_621/{type}{type_num2}"
    # main(input, output, ref, small_max_size = 3000, processes = 8)

    # type = "Train-Labeled"
    # input = f"/home/data2/xrs/dataset/STS26-Task1_S3_Pred/pred_216/{type}"
    # output = f"/home/data2/xrs/dataset/STS26-Task1_S3_Pred/pred/{type}"
    # ref = f"/home/data2/xrs/dataset/STS26-Task1_S3_Pred/pred_121/{type}"
    # main(input, output, ref, small_max_size = 3000, processes = 8)

    # type = "Train-Unlabeled"
    # input = f"/home/data2/xrs/dataset/STS26-Task1_S3_Pred/pred_216/{type}"
    # output = f"/home/data2/xrs/dataset/STS26-Task1_S3_Pred/pred/{type}"
    # ref = f"/home/data2/xrs/dataset/STS26-Task1_S3_Pred/pred_121/{type}"
    # main(input, output, ref, small_max_size = 3000, processes = 8)

    # type = "Validation"
    # input = f"/home/data2/xrs/dataset/STS26-Task1_S3_Pred/pred_216/{type}"
    # output = f"/home/data2/xrs/dataset/STS26-Task1_S3_Pred/pred/{type}"
    # ref = f"/home/data2/xrs/dataset/STS26-Task1_S3_Pred/pred_121/{type}"
    # main(input, output, ref, small_max_size = 3000, processes = 8)

    type_num1 = "500"
    type_num2 = "250"

    type = "Validation"
    input = f"/home/data2/xrs/dataset/pred_out/pred_716/{type}{type_num1}"
    output = f"/home/data2/xrs/dataset/pred_out/pred/{type}{type_num1}_{type_num2}"
    ref = f"/home/data2/xrs/dataset/pred_out/pred_621/{type}{type_num2}"
    main(input, output, ref, small_max_size = 3000, processes = 8)


    print("ok")