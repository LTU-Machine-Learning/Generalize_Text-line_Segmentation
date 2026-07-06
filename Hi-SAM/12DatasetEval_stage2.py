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

def Main_function(args, gt_path):
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
    score_thresh, nms_thresh0  = tuple(args.nms) #for model 60-> (0.65, 0.55)| for model 57-> (0.2, 0.35)
    print(f"score_thresh: {score_thresh} | nms_thresh: {nms_thresh0}")
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


    metric = MeanAveragePrecision(iou_type="bbox")
    f1_scores = []
    my_mAP = []
    cer_all_norm_nb_letter, nb_total_letter = 0, 0


    for path in tqdm(image_paths):
        img_id = os.path.basename(path).split('.')[0]

        if os.path.isdir(args.output):
            assert os.path.isdir(args.output), args.output
            img_name = img_id + '.png'
            out_filename = os.path.join(args.output, img_name)
        else:
            assert len(image_paths) == 1
            out_filename = args.output

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # h, w, 3

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

        pred_boxes, pred_boxes2, pred_scores = [], [], []
        if masks is not None:
            for mask, score in zip(masks, scores):
                mask = np.squeeze(mask)

                ########## Connected component filtering and Bux duplicate filter #############
                binary_mask = (mask > 0).astype(np.uint8)  # 0/1
                binary_mask = binary_mask * 255  # convert to 0/255 for OpenCV
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

                if num_labels <= 1:  # no components
                    continue
                
                areas = stats[1:, 4]  # skip background (index 0) # stats[:, 4] is the area of each component
                max_area = areas.max()
                threshold = 0.05 * max_area  # 5% of largest area

                # Create new mask with only large components
                filtered_mask = np.zeros_like(binary_mask)

                for i, area in enumerate(areas):
                    if area >= threshold:
                        filtered_mask[labels == i + 1] = 255  # i+1 because stats skips background

                filtered_mask = (filtered_mask > 0).astype(np.uint8)

                mask = filtered_mask # I'm too lazy to change variable names further down the code :D
                #########################     
                 
                pred_box = mask_to_bbox(mask)
                box_with_score = np.append(pred_box, score)
                if pred_box is not None:
                    pred_boxes.append(pred_box)
                    pred_scores.append(score)
                    pred_boxes2.append(box_with_score)

        if args.save_boxes: 
            os.makedirs(args.save_boxes_dir, exist_ok=True)
            with open(f"{args.save_boxes_dir}/{img_id}.txt", "w") as f:
                for box in pred_boxes2:
                    # line = " ".join(str(int(x)) for x in box)
                    line = " ".join([str(int(x)) for x in box[:4]] + [str(box[4])])
                    f.write(line + "\n")

        if args.save_visualization:
            show_masks(masks, out_filename, image)  # visualize the predicted boundary boxes

        if args.map :
            # Calculate mAP for the page
            gt_boxes = get_GT_Boxes_onePage(gt_path, img_id)

            f1_scores.append(compute_f1_gpu(gt_boxes, pred_boxes)["f1"])
            my_mAP.append(compute_bayes_map_50_95_gpu(gt_boxes, pred_boxes, bayes=True))

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

            _, _, cer_page_sum, page_total_letter, _, _ = htr_scores(img_id, pred_boxes)   
            cer_all_norm_nb_letter += cer_page_sum
            nb_total_letter += page_total_letter 


    result_ap = metric.compute()
    cer = cer_all_norm_nb_letter / nb_total_letter
    # cer = 0

    res_mAP = result_ap["map"].item() # round(result_ap["map"].item(), 4)
    res_mAP50 = round(result_ap["map_50"].item(), 4)
    res_mAP75 = round(result_ap["map_75"].item(), 4)

    f1_score = np.mean(f1_scores, axis=0) #round(np.mean(f1_scores, axis=0), 4)
    my_mAP_score = np.mean(my_mAP, axis=0) #round(np.mean(my_mAP, axis=0), 4)
    print(f"F1 Score: {f1_score}")
    print(f"bayes-mAP Score: {my_mAP_score}")

    print(f" mAP: {res_mAP} | mAP_50: {res_mAP50} | mAP_75: {res_mAP75}")
    # print(f"mAP: {res_mAP}")

    return res_mAP, f1_score, cer


if __name__ == '__main__':
    args = get_args_parser()


    
    root = "/home/x_gapat/PROJECTS"
    pathlog=f"{root}/logs/ATS/Hi-SAM/12DatasetEval"

    Segmenter_trained_datasets = {
        # "READ_2016": {
        #     "seg_model_tag": "H060",
        #     "seg_model": "60_2025-10-01_ID_14189314",
        #     "nms": (0.55, 0.5)
        #     },
        # "NorHandV3_331861": {
        #     "seg_model_tag": "H094",
        #     "seg_model": "94_2026-04-12_ID_16223622",
        #     "nms": (0.65, 0.4)
        #     },
        # "Nuremberg_Letterbooks_Band3": {
        #     "seg_model_tag": "H095",
        #     "seg_model": "95_2026-04-17_ID_16360065",
        #     "nms": (0.7, 0.5)
        #     },
        "Mix_ReNoNu": {
            "seg_model_tag": "H096",
            "seg_model": "96_2026-04-15_ID_16302713",
            "nms": (0.65, 0.45)
            },
        # "READ_2016": {
        #     "seg_model_tag": "H091",
        #     "seg_model": "91_2026-02-05_ID_15442927",
        #     "nms": (0.7, 0.7)
        #     },
        # "Mix_ReNoNu": {
        #     "seg_model_tag": "H103",
        #     "seg_model": "103_2026-05-13_ID_16519201",
        #     "nms": (0.55, 0.45)
        #     }
    }


    inference_datasets = {
        "READ_2016": {
            "dataset_sufix": "READ_2016",
            "htr_crnn_model": "047_2026-04-07_ID_16124606",
            "htr_bb_sorting": "IoU-matching",
            "img_ext": "JPG",
            },
        "Bentham": {
            "dataset_sufix": "Bentham",
            "htr_crnn_model": "044_2026-04-07_ID_16124600",
            "htr_bb_sorting": "IoU-matching",
            "img_ext": "jpg",
            },       
        "NorHandV3_331861": {
            "dataset_sufix": "NorHandV3_331861",
            "htr_crnn_model": "046_2026-04-07_ID_16124605",
            "htr_bb_sorting": "IoU-matching",
            "img_ext": "jpg",
            },        
        "NorHandV3_72702": {
            "dataset_sufix": "NorHandV3_72702",
            "htr_crnn_model": "045_2026-04-07_ID_16124604",
            "htr_bb_sorting": "IoU-matching",
            "img_ext": "jpg",
            },        
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
        # "Nuremberg_Letterbooks_Band4": {
        #     "dataset_sufix": "Nuremberg_Letterbooks/Band4",
        #     "htr_crnn_model": "053_2026-04-07_ID_16124625",
        #     "htr_bb_sorting": "IoU-matching",
        #     "img_ext": "jpg",
        #     },        
        # "Nuremberg_Letterbooks_Band5": {
        #     "dataset_sufix": "Nuremberg_Letterbooks/Band5",
        #     "htr_crnn_model": "054_2026-04-10_ID_16168201",
        #     "htr_bb_sorting": "IoU-matching",
        #     "img_ext": "jpg",
        #     },        
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

    args.pretrained_path = f"{root}/logs/Hi-SAM_Doc/pretrained_checkpoint"
    args.htr_crnn_model_config = "utils/HTR_CRNN/configs/model_config_1.json"
    args.results_log = f"{pathlog}/stage2_results_newCER.txt"
    out_boxes_dir = "few-shot-0"
    

    for dataset_seg, value_seg in Segmenter_trained_datasets.items():
        print(f"Segmentation model Trained on: {dataset_seg}")
        args.checkpoint = f"{root}/logs/Hi-SAM_Doc/{value_seg['seg_model']}/saved_model/READ_2016_best_mAP.pth"
        args.nms = value_seg["nms"]

        for dataset_inf, value_inf in inference_datasets.items():
            print(f"Inference dataset: {dataset_inf}")
            args.htr_charset = f"{root}/DATASETS/{value_inf['dataset_sufix']}/box_lines_dataset/charset.txt"
            args.htr_crnn_model_path = f"{root}/logs/CRNN_Center_loss/ATS_HTR_Training/{value_inf['htr_crnn_model']}/crnn_best.torch"
            args.htr_bb_sorting = value_inf["htr_bb_sorting"]
            args.input = f"{root}/DATASETS/{value_inf['dataset_sufix']}/test/Images"
            args.output = f"{pathlog}/{value_seg['seg_model_tag']}/{value_inf['dataset_sufix']}/images"
            args.img_ext = value_inf["img_ext"]
            args.save_boxes_dir = f"{pathlog}/{value_seg['seg_model_tag']}/{value_inf['dataset_sufix']}/{out_boxes_dir}/boxes"
            args.text = f"TrainOn_{dataset_seg}-TestOn_{dataset_inf}"
            
            gt_path = os.path.join(os.path.dirname(args.input), "gt_xml")
            os.makedirs(args.output, exist_ok=True)

            res_mAP, f1_score, cer = Main_function(args, gt_path)

            modelname = os.path.basename(args.checkpoint)
            score_thresh, nms_thresh0  = tuple(args.nms)
            if args.exp_id == 0:
                args.exp_id = args.checkpoint.split('/')[-3].split('_')[0]
            with open(args.results_log, "a") as f:
                f.write(f"{value_seg['seg_model_tag']};{score_thresh};{nms_thresh0};{res_mAP};{f1_score};{cer};{modelname};{args.text}\n")
    
