import numpy as np
import torch
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment

def get_indexes_sort_bb(bb_xyxy, dataset_name):
    if dataset_name == "single-col-sorting":
        return get_indexes_sort_bb_single_columns(bb_xyxy)
    elif dataset_name == "multi-col-sorting":
        return get_indexes_sort_bb_multiple_columns(bb_xyxy)
    elif dataset_name == "IoU-matching":
        return find_IoU_with_GT_Match(bb_xyxy[0], bb_xyxy[1])
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")           


def get_indexes_sort_bb_single_columns(bb_xyxy):
    # Sort y top
    bb_xyxy = np.array(bb_xyxy)
    index_sorted = np.argsort(bb_xyxy[:,1])

    return index_sorted
    
### GPU versio of the code which runs faster
def find_IoU_with_GT_Match(gt_xyxy, pred_xyxy, device="cpu"):
    # Convert to tensors on GPU
    gt = torch.as_tensor(gt_xyxy, dtype=torch.float32, device=device)
    pr = torch.as_tensor(pred_xyxy, dtype=torch.float32, device=device)

    # Make sure shapes are always (N, 4) and (M, 4)
    if gt.numel() == 0:
        gt = gt.reshape(0, 4)
    elif gt.ndim == 1:
        gt = gt.unsqueeze(0)

    if pr.numel() == 0:
        pr = pr.reshape(0, 4)
    elif pr.ndim == 1:
        pr = pr.unsqueeze(0)

    # Handle empty inputs safely
    if gt.shape[0] == 0 or pr.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # IoU matrix on GPU: (n_gt, n_pr)
    ious = box_iou(gt, pr)  

    # Hungarian algorithm (maximize IoU → minimize 1 - IoU)
    cost_matrix = 1 - ious
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)   
    
    return gt_indices, pred_indices

def get_indexes_sort_bb_multiple_columns(bb_xyxy):
    all_width = [one_bb[2] - one_bb[0] for one_bb in bb_xyxy]
    mean_width = np.mean(all_width)

    t_x = mean_width / 3

    # Sort x center
    bb_xyxy_sorted = sorted(bb_xyxy, key=lambda x: (x[0] + x[2]) / 2.0, reverse=False)

    # only_x1 = [one_bb[0] for one_bb in bb_xyxy_sorted]
    only_x1 = [(one_bb[0] + one_bb[2]) / 2 for one_bb in bb_xyxy_sorted]
    # Compute diff x
    diff = np.diff(only_x1)

    # Form group column
    all_bb_sorted = []
    group_one_column = [bb_xyxy_sorted[0]]

    # debug
    # nb_cut = 0

    for i in range(len(diff)):
        current_diff = diff[i]

        # Cut
        if current_diff > t_x:
            # nb_cut += 1
            # Sort y
            group_one_column = sorted(group_one_column, key=lambda x: x[1], reverse=False)
            all_bb_sorted = all_bb_sorted + group_one_column

            # Reset current column
            group_one_column = []

        group_one_column.append(bb_xyxy_sorted[i+1])

    # print("nb_cut: " + str(nb_cut))

    # Last group
    if len(group_one_column) > 0:
        # Sort y
        group_one_column = sorted(group_one_column, key=lambda x: x[1], reverse=False)
        all_bb_sorted = all_bb_sorted + group_one_column

    # return index sorted of orignal list
    index_sorted = []

    for i_s in range(len(all_bb_sorted)):
        for i_origin in range(len(bb_xyxy)):
            if bb_xyxy[i_origin] == all_bb_sorted[i_s]:
                index_sorted.append(i_origin)

    return index_sorted


