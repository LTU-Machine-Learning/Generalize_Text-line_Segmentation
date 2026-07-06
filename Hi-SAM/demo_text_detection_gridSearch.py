import json
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import skimage
import os
import argparse
from hi_sam.modeling.build import model_registry
from hi_sam.modeling.auto_mask_generator import AutoMaskGenerator
import glob
from tqdm import tqdm
from PIL import Image
import random
from utils import utilities
from shapely.geometry import Polygon
import pyclipper
import datetime
import pickle
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment
import xml.etree.ElementTree as ET
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pandas as pd
from utils.f1_score_for_bb import compute_f1, compute_f1_gpu 


import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('Hi-SAM', add_help=False)

    parser.add_argument("--input", type=str, required=True, nargs="+",
                        help="Path to the input image")
    parser.add_argument("--output", type=str, default='./demo',
                        help="A file or directory to save output visualizations.")
    parser.add_argument("--model-type", type=str, default="vit_l",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")
    parser.add_argument("--hier_det", default=True)
    parser.add_argument("--dataset", type=str, required=True, default='totaltext',
                        help="'totaltext' or 'ctw1500', or 'ic15'.")
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--zero_shot", action='store_true')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)
    parser.add_argument("--pretrained_path", type=str, default="pretrained_checkpoint",
                        help="pretrained checkpoint path. Only the folder")
    # self-prompting
    parser.add_argument('--attn_layers', default=1, type=int,
                        help='The number of image to token cross attention layers in model_aligner')
    parser.add_argument('--prompt_len', default=12, type=int, help='The number of prompt token')
    parser.add_argument('--layout_thresh', type=float, default=0.5)

    parser.add_argument('--res_csv', type=str, default="results.csv")
    parser.add_argument("--new_csv", type=bool, default=False, help="If True, create a new CSV file")
    parser.add_argument("--exp_id", type=int, default=0, help="Experiment ID")
    parser.add_argument("--results_log", type=str, default="results.txt", help="results file")
    parser.add_argument("--save_boxes", type=bool, default=False, help="Export predicted Boundary boxes")
    parser.add_argument('--save_boxes_dir', type=str, default="save_box")

    parser.add_argument("--target_dataset_name", type=str,  default="READ_2016")
    parser.add_argument("--text", type=str,  default="")
    return parser.parse_args()


def unclip(p, unclip_ratio=2.0):
    poly = Polygon(p)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(p, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def polygon2rbox(polygon, image_height, image_width):
    rect = cv2.minAreaRect(polygon)
    corners = cv2.boxPoints(rect)
    corners = np.array(corners, dtype="int")
    pts = get_tight_rect(corners, 0, 0, image_height, image_width, 1)
    pts = np.array(pts).reshape(-1, 2)
    return pts


def get_tight_rect(points, start_x, start_y, image_height, image_width, scale):
    points = list(points)
    ps = sorted(points, key=lambda x: x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0] * scale + start_x
        py1 = ps[0][1] * scale + start_y
        px4 = ps[1][0] * scale + start_x
        py4 = ps[1][1] * scale + start_y
    else:
        px1 = ps[1][0] * scale + start_x
        py1 = ps[1][1] * scale + start_y
        px4 = ps[0][0] * scale + start_x
        py4 = ps[0][1] * scale + start_y
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0] * scale + start_x
        py2 = ps[2][1] * scale + start_y
        px3 = ps[3][0] * scale + start_x
        py3 = ps[3][1] * scale + start_y
    else:
        px2 = ps[3][0] * scale + start_x
        py2 = ps[3][1] * scale + start_y
        px3 = ps[2][0] * scale + start_x
        py3 = ps[2][1] * scale + start_y

    px1 = min(max(px1, 1), image_width - 1)
    px2 = min(max(px2, 1), image_width - 1)
    px3 = min(max(px3, 1), image_width - 1)
    px4 = min(max(px4, 1), image_width - 1)
    py1 = min(max(py1, 1), image_height - 1)
    py2 = min(max(py2, 1), image_height - 1)
    py3 = min(max(py3, 1), image_height - 1)
    py4 = min(max(py4, 1), image_height - 1)
    return [px1, py1, px2, py2, px3, py3, px4, py4]


def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = color if color is not None else np.array([30/255, 144/255, 255/255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_masks(masks, filename, image):
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    for i, mask in enumerate(masks):
        mask = mask[0].astype(np.uint8)
        # contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # for cont in contours:
        #     epsilon = 0.002 * cv2.arcLength(cont, True)
        #     approx = cv2.approxPolyDP(cont, epsilon, True)
        #     pts = approx.reshape((-1, 2))
        #     if pts.shape[0] < 4:
        #         continue
        #     pts = pts.astype(np.int32)
        #     mask = cv2.fillPoly(np.zeros(mask.shape), [pts], 1)
        show_mask(mask, plt.gca(), random_color=True)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def polygon_to_bbox(poly):
    xs = [point[0] for point in poly]
    ys = [point[1] for point in poly]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return [x_min, y_min, x_max, y_max]

def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None  # Empty mask, no bounding box
    x_min = np.min(xs)
    y_min = np.min(ys)
    x_max = np.max(xs)
    y_max = np.max(ys)
    return [x_min, y_min, x_max, y_max]

def get_GT_Boxes_onePage(gt_path, image_id):
    tree = ET.parse(os.path.join(gt_path, f"{image_id}.xml"))
    root = tree.getroot()
    ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    w = int(root.find("pc:Page", ns).attrib['imageWidth'])
    h = int(root.find("pc:Page", ns).attrib['imageHeight'])
    
    ## Extract GT Points ######################
    gt_points = []
    for line in root.findall(".//pc:TextLine", ns):
        coords_elem = line.find("pc:Coords", ns)
        points_str = coords_elem.attrib['points']
        points = [tuple(map(int, p.split(','))) for p in points_str.split()]
        gt_points.append(points)

    gt_boxes = []
    for poly in gt_points:
        gt_boxes.append(polygon_to_bbox(poly))
    return  gt_boxes


def get_GT_Boxes_onePage_IAM(gt_path, image_id):
    tree = ET.parse(os.path.join(gt_path, f"{image_id}.xml"))
    root = tree.getroot()

    line_codes = []
    for line in root.findall(".//line"):
        cmp_boxes = []
        for word in line.findall(".//word"):
            for cmp_elem in word.findall("cmp"):
                x = int(cmp_elem.attrib['x'])
                y = int(cmp_elem.attrib['y'])
                w = int(cmp_elem.attrib['width'])
                h = int(cmp_elem.attrib['height'])
                cmp_boxes.append((x, y, w, h))
        
        # Calculate union bounding box
        x_min = min(x for x, y, w, h in cmp_boxes)
        y_min = min(y for x, y, w, h in cmp_boxes)
        x_max = max(x + w for x, y, w, h in cmp_boxes)
        y_max = max(y + h for x, y, w, h in cmp_boxes)
        
        # print(f"{x_min}, {y_min}, {x_max}, {y_max}")
        line_codes.append([x_min, y_min, x_max, y_max])
    return line_codes

def yolo_to_xyxy(path, file_id, img_h, img_w):
    new_coords = []
    with open(os.path.join(path, f"{file_id}.txt"), "r") as file:
        for line in file:
            parts = line.strip().split()
            cls = int(parts[0])  # class ID, if you care
            x_c, y_c, w, h = map(float, parts[1:])

            # convert normalized YOLO to pixels
            x_c *= img_w
            y_c *= img_h
            w *= img_w
            h *= img_h

            xmin = int(x_c - w / 2)
            ymin = int(y_c - h / 2)
            xmax = int(x_c + w / 2)
            ymax = int(y_c + h / 2)

            new_coords.append([xmin, ymin, xmax, ymax])
    return new_coords
# def get_Page_mAP_Score(gt_path, predictions):
#     gt_boxes = []
#     pred_boxes = []
#     pred_conf = []
#     for img_id, (masks, scores) in predictions.items():
#         gt_boxes_per_page = get_GT_Boxes_onePage(gt_path, img_id)
#         gt_boxes.extend(gt_boxes_per_page)

#         pred_boxes_per_page = []
#         for mask in masks:
#             mask = np.squeeze(mask)
#             pred_box = mask_to_bbox(mask)
#             pred_boxes_per_page.append(pred_box)   
#         pred_boxes.extend(pred_boxes_per_page)
#         pred_conf.extend(scores)


#     ### Construct input for mAP
#     targets = [{
#         "boxes": torch.tensor(gt_boxes, dtype=torch.float),
#         "labels": torch.tensor([0] * len(gt_boxes))
#     }]

#     preds = [{
#         "boxes": torch.tensor(pred_boxes),
#         "scores": torch.tensor(pred_conf),
#         "labels": torch.tensor([0] * len(pred_boxes))
#     }]

#     metric = MeanAveragePrecision(iou_type="bbox")
#     metric.update(preds, targets)
#     result_ap = metric.compute()

#     return result_ap

if __name__ == '__main__':
    args = get_args_parser()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hisam = model_registry[args.model_type](args)
    hisam.eval()
    hisam.to(args.device)
    print("Loaded model")
    amg = AutoMaskGenerator(hisam)
    gt_path = os.path.join(os.path.dirname(args.input[0]), "gt_xml")
    # gt_path = "../../DATASETS/IAM/xml"  # For IAM dataset
    new_csv = args.new_csv
    df_name = args.res_csv

    os.makedirs(args.output, exist_ok=True)

    if args.model_type == "vit_b":
        csv_model_size = "B"
    elif args.model_type == "vit_l":
        csv_model_size = "L"
    elif args.model_type == "vit_h":
        csv_model_size = "H"   
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    if new_csv: 
        pd.DataFrame([], columns = ["Experiment", "Model_Size", "score_thresh","mAP","mAP_50","mAP_75"]).to_csv(df_name, index=False)
    # df_data = []

    if args.dataset == 'totaltext':
        if args.zero_shot:
            fg_points_num = 50  # assemble text kernel
            score_thresh = 0.3
            unclip_ratio = 1.5
        else:
            fg_points_num = 500
            score_thresh = 0.95
    elif args.dataset == 'ctw1500':
        if args.zero_shot:
            fg_points_num = 100
            score_thresh = 0.6
        else:
            fg_points_num = 300
            score_thresh = 0.7
    else:
        raise ValueError
    
    # score_threshs = [0.6, 0.65, 0.7, 0.75, 0.8]
    score_threshs = np.arange(0.3, 0.96, 0.05).round(2).tolist()
    nms_threshs = np.arange(0.3, 0.96, 0.05).round(2).tolist()
    # score_threshs = [0.6]
    # nms_threshs = [0.6]

    

    if os.path.isdir(args.input[0]):
        args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
    elif len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"

    best_score_thresh, best_nms_thresh0, best_mAP = (0, 0, 0)
    best_score_threshf1, best_nms_thresh0f1, best_f1 = (0, 0, 0)

    for score_thresh in score_threshs:
        for nms_thresh0 in nms_threshs:
            # df = pd.read_csv(df_name)
            metric = MeanAveragePrecision(iou_type="bbox")
            f1_scores = []
            for path in tqdm(args.input):
                img_id = os.path.basename(path).split('.')[0]

                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    img_name = os.path.basename(path).split('.')[0] + '.png'
                    out_filename = os.path.join(args.output, img_name)
                else:
                    assert len(args.input) == 1
                    out_filename = args.output

                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # h, w, 3
                img_h, img_w = image.shape[:2]
                page_size = (img_w, img_h)

                amg.set_image(image)      
            
                masks, scores = amg.predict_text_detection(
                    from_low_res=False,
                    fg_points_num=fg_points_num,
                    batch_points_num=min(fg_points_num, 100),
                    score_thresh=score_thresh,
                    nms_thresh=nms_thresh0,
                    zero_shot=args.zero_shot,
                    dataset=args.dataset
                )

                # Calculate mAP for the page
                pred_boxes, pred_scores = [], []
                if masks is not None:
                    for mask, score in zip(masks, scores):
                        mask = np.squeeze(mask)
                        pred_box = mask_to_bbox(mask)
                        if pred_box is not None:
                            pred_boxes.append(pred_box)
                            pred_scores.append(score)

                gt_boxes = get_GT_Boxes_onePage(gt_path, img_id)
                # gt_boxes = get_GT_Boxes_onePage_IAM(gt_path, img_id)
                # gt_boxes = yolo_to_xyxy(gt_path, img_id, img_h, img_w)
                
                # f1_scores.append(compute_f1(gt_boxes, pred_boxes, page_size=page_size)["f1"])
                f1_scores.append(compute_f1_gpu(gt_boxes, pred_boxes)["f1"])

                

                # if args.save_boxes: 
                #     with open(f"{args.save_boxes_dir}/{img_id}.txt", "w") as f:
                #         for box in pred_boxes:
                #             line = " ".join(str(int(x)) for x in box)
                #             f.write(line + "\n")

                ### Construct input for mAP..
                targets = [{
                    "boxes": torch.tensor(gt_boxes, dtype=torch.float),
                    "labels": torch.tensor([0] * len(gt_boxes))
                }]
            
                if len(pred_boxes) == 0:
                    preds = [{
                        "boxes": torch.empty((0, 4), dtype=torch.float),   # empty tensor
                        "scores": torch.empty((0,), dtype=torch.float),
                        "labels": torch.empty((0,), dtype=torch.int64)
                    }]
                else:
                    preds = [{
                        "boxes": torch.tensor(pred_boxes),
                        "scores": torch.tensor(pred_scores),  # Here the score is actually a confident score predicted by Hi-SAM
                        "labels": torch.tensor([0] * len(pred_boxes))
                    }]

                metric.update(preds, targets)
                # show_masks(masks, out_filename, image)  # Comment / Uncomment this to visualize the predicted boundary boxes
            result_ap = metric.compute()

            res_mAP = round(result_ap["map"].item(), 4)

            f1_score = np.mean(f1_scores, axis=0)
            print(f"F1 Score: {f1_score}")

            print(f"score_thresh: {score_thresh} | nms_thresh0: {nms_thresh0} | mAP: {res_mAP}")

            if best_mAP < res_mAP :
                best_mAP = res_mAP
                best_score_thresh = score_thresh
                best_nms_thresh0 = nms_thresh0

            if best_f1 < f1_score :
                best_f1 = f1_score
                best_score_threshf1 = score_thresh
                best_nms_thresh0f1 = nms_thresh0
                best_mAP_at_bestF1 = res_mAP
    print(f"BEST | score_thresh: {best_score_thresh} | nms_thresh0: {best_nms_thresh0} | mAP: {best_mAP}")
    print(f"BEST | score_thresh: {best_score_threshf1} | nms_thresh0: {best_nms_thresh0f1} | F1: {best_f1}")

    with open(args.results_log, "a") as f:
        f.write(f"{args.target_dataset_name};{best_nms_thresh0f1};{best_score_threshf1};{best_f1};{best_mAP_at_bestF1};{args.text}\n")