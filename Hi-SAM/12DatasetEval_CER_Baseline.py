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
from utils.f1_score_for_bb import compute_f1_gpu, compute_bayes_map_50_95_gpu

from utils.HTR_CRNN.data.readers.read_bb_export import read_line_bb_xyxy_txt
from utils.HTR_CRNN.evaluate.htr_eval_fn import HTRScores

import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('Hi-SAM', add_help=False)

    parser.add_argument("--input", type=str, nargs="+", default="",
                        help="Path to the input image")
    parser.add_argument("--output", type=str, default='./demo',
                        help="A file or directory to save output visualizations.")
    parser.add_argument("--model-type", type=str, default="vit_h",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")
    parser.add_argument("--hier_det", default=True)
    parser.add_argument("--dataset", type=str, default='ctw1500',
                        help="'totaltext' or 'ctw1500', or 'ic15'.")
    parser.add_argument("--vis", action='store_true', default="")
    parser.add_argument("--zero_shot", action='store_true')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)
    parser.add_argument("--pretrained_path", type=str, default="pretrained_checkpoint",
                        help="pretrained checkpoint path. Only the folder")
    parser.add_argument("--img_ext", type=str, default="JPG",
                        help="Image file extension. Options: JPG, jpg, PNG, png")
    # self-prompting
    parser.add_argument('--attn_layers', default=1, type=int,
                        help='The number of image to token cross attention layers in model_aligner')
    parser.add_argument('--prompt_len', default=12, type=int, help='The number of prompt token')
    parser.add_argument('--layout_thresh', type=float, default=0.5)

    parser.add_argument('--res_csv', type=str, default="results.csv")
    parser.add_argument("--new_csv", type=bool, default=False, help="If True, create a new CSV file")
    parser.add_argument("--exp_id", type=int, default=0, help="Experiment ID")
    parser.add_argument("--results_log", type=str, default="results.txt", help="results file")
    parser.add_argument("--save_boxes", type=bool, default=True, help="Export predicted Boundary boxes")
    parser.add_argument('--save_boxes_dir', type=str, default="save_box")
    parser.add_argument("--save_visualization", type=bool, default=False, help="Export predicted masks on the input image")
    parser.add_argument("--map", type=bool, default=True)
    parser.add_argument("--nms", type=float, nargs=2,   help="Tuple input, e.g., 3.5 6.5")
    parser.add_argument("--text", type=str,  default="")

    # HTR related
    parser.add_argument('--htr_charset', type=str, default=None)
    parser.add_argument('--htr_height_max', type=int, default=128)
    parser.add_argument('--htr_width_max', type=int, default=1024)
    parser.add_argument('--htr_crnn_model_path', type=str, default=None)
    parser.add_argument('--htr_crnn_model_config', type=str, default=None)
    parser.add_argument('--htr_bb_sorting', type=str, default="IoU-matching", help="Options: 'single-col-sorting', 'multi-col-sorting', 'IoU-matching'")

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
    # w = int(root.find("pc:Page", ns).attrib['imageWidth'])
    # h = int(root.find("pc:Page", ns).attrib['imageHeight'])
    
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


def Main_function(args, gt_path):

    # HTR As a Scotre ###
    htr_scores = HTRScores(
        args.device,
        f"{args.htr_charset}",
        f"{args.input}",
        gt_path,
        args.htr_crnn_model_path,
        args_height_max_line=args.htr_height_max,
        args_width_max_line=args.htr_width_max,
        args_ext_img= args.img_ext,
        args_bb_sorting=args.htr_bb_sorting,
        args_model_config_path=args.htr_crnn_model_config
    )

    if os.path.isdir(args.input):
        image_paths = [os.path.join(args.input, fname) for fname in os.listdir(args.input)]

    cer_all_norm_nb_letter, nb_total_letter = 0, 0

    for path in tqdm(image_paths):
        img_id = os.path.basename(path).split('.')[0]

        gt_boxes = get_GT_Boxes_onePage(gt_path, img_id)

        _, _, cer_page_sum, page_total_letter, _, _ = htr_scores(img_id, gt_boxes)   
        cer_all_norm_nb_letter += cer_page_sum
        nb_total_letter += page_total_letter 


    cer = cer_all_norm_nb_letter / nb_total_letter
    # cer = 0

    return cer


if __name__ == '__main__':
    args = get_args_parser()


    
    root = "/home/x_gapat/PROJECTS"
    pathlog=f"{root}/logs/ATS/Hi-SAM/12DatasetEval2"



    inference_datasets = {
        # "READ_2016": {
        #     "dataset_sufix": "READ_2016",
        #     "htr_crnn_model": "047_2026-04-07_ID_16124606",
        #     "htr_bb_sorting": "IoU-matching",
        #     "img_ext": "JPG",
        #     },
        # "Bentham": {
        #     "dataset_sufix": "Bentham",
        #     "htr_crnn_model": "044_2026-04-07_ID_16124600",
        #     "htr_bb_sorting": "IoU-matching",
        #     "img_ext": "jpg",
        #     },       
        # "NorHandV3_331861": {
        #     "dataset_sufix": "NorHandV3_331861",
        #     "htr_crnn_model": "046_2026-04-07_ID_16124605",
        #     "htr_bb_sorting": "IoU-matching",
        #     "img_ext": "jpg",
        #     },        
        # "NorHandV3_72702": {
        #     "dataset_sufix": "NorHandV3_72702",
        #     "htr_crnn_model": "045_2026-04-07_ID_16124604",
        #     "htr_bb_sorting": "IoU-matching",
        #     "img_ext": "jpg",
        #     },        
        # "Nuremberg_Letterbooks_Band2": {
        #     "dataset_sufix": "Nuremberg_Letterbooks/Band2",
        #     "htr_crnn_model": "051_2026-04-07_ID_16124623",
        #     "htr_bb_sorting": "IoU-matching",
        #     "img_ext": "jpg",
        #     },        
        "Nuremberg_Letterbooks_Band3": {
            "dataset_sufix": "Nuremberg_Letterbooks/Band3",
            "htr_crnn_model": "052_2026-04-10_ID_16168196",
            "htr_bb_sorting": "IoU-matching",
            "img_ext": "jpg",
            },        
        "Nuremberg_Letterbooks_Band4": {
            "dataset_sufix": "Nuremberg_Letterbooks/Band4",
            "htr_crnn_model": "053_2026-04-07_ID_16124625",
            "htr_bb_sorting": "IoU-matching",
            "img_ext": "jpg",
            },        
        "Nuremberg_Letterbooks_Band5": {
            "dataset_sufix": "Nuremberg_Letterbooks/Band5",
            "htr_crnn_model": "054_2026-04-10_ID_16168201",
            "htr_bb_sorting": "IoU-matching",
            "img_ext": "jpg",
            },        
        "Riksarkivet_Bergskollegium": {
            "dataset_sufix": "Riksarkivet_Bergskollegium",
            "htr_crnn_model": "048_2026-04-10_ID_16168088",
            "htr_bb_sorting": "IoU-matching",
            "img_ext": "png",
            },     
        "Riksarkivet_Bergskollegium2": {
            "dataset_sufix": "Riksarkivet_Bergskollegium2",
            "htr_crnn_model": "056_2026-04-09_ID_16155496",
            "htr_bb_sorting": "IoU-matching",
            "img_ext": "png",
            },     
        "Riksarkivet_Goteborgs1": {
            "dataset_sufix": "Riksarkivet_Goteborgs1",
            "htr_crnn_model": "049_2026-04-10_ID_16168185",
            "htr_bb_sorting": "IoU-matching",
            "img_ext": "png",
            },        
        "Riksarkivet_Goteborgs2": {
            "dataset_sufix": "Riksarkivet_Goteborgs2",
            "htr_crnn_model": "050_2026-04-10_ID_16168187",
            "htr_bb_sorting": "IoU-matching",
            "img_ext": "png",
            },        
        "Mix_ReNoNu": {
            "dataset_sufix": "Mix_ReNoNu",
            "htr_crnn_model": "055_2026-04-10_ID_16168210",
            "htr_bb_sorting": "IoU-matching",
            "img_ext": "jpg",
            }
    }

    args.htr_crnn_model_config = "utils/HTR_CRNN/configs/model_config_1.json"
    args.results_log = f"{pathlog}/stage2_results_cer_baseline.txt"
    



    for dataset_inf, value_inf in inference_datasets.items():
        print(f"Inference dataset: {dataset_inf}")
        args.htr_charset = f"{root}/DATASETS/{value_inf['dataset_sufix']}/box_lines_dataset/charset.txt"
        args.htr_crnn_model_path = f"{root}/logs/CRNN_Center_loss/ATS_HTR_Training/{value_inf['htr_crnn_model']}/crnn_best.torch"
        args.htr_bb_sorting = value_inf["htr_bb_sorting"]
        args.input = f"{root}/DATASETS/{value_inf['dataset_sufix']}/test/Images"
        args.img_ext = value_inf["img_ext"]


        
        gt_path = os.path.join(os.path.dirname(args.input), "gt_xml")

        cer = Main_function(args, gt_path)
        
        with open(args.results_log, "a") as f:
            f.write(f"{dataset_inf};{cer}\n")

