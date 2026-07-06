import os
import glob
import pickle
import json

import cv2
import torch

from ultralytics import YOLO
from ultralytics.utils.metrics import bbox_iou

from HTR_CRNN.data.readers.read_xml_page import read_xml_page_gt
from HTR_CRNN.data.readers.read_yolo_bb_gt import read_yolo_gt


# need gt annotation format page.xml
def save_yolo_pred(dir_data_img,
                   dir_data_xml,
                   dir_save,
                   path_model,
                   ext_img,
                   img_size,
                   iou_t,
                   conf_t,
                   save_label_txt,
                   agnostic_nms=True):
    # Create directories result
    save_img_line = os.path.join(dir_save, "image_line")
    os.makedirs(save_img_line, exist_ok=True)

    # Load model
    model = YOLO(path_model)

    files_img = glob.glob(dir_data_img + '/**/*.' + ext_img, recursive=True)

    counter_iou_small = 0
    counter_gt_is_None = 0

    nb_line_pred_ok = 0
    nb_line_pred_more = 0
    nb_line_pred_less = 0

    total_items = 0
    # iou_all = 0
    nb_total_line = 0
    nb_total_line_eval = 0

    iou_all_find_closest = 0

    dict_label_line_gt = {}

    for one_file in files_img:
        print(one_file)
        total_items += 1
        img = cv2.imread(one_file)

        # Get id file
        split_name = os.path.split(one_file)
        split_name = split_name[1].split(sep=".")  # Filename and extension
        id_file = split_name[0]

        # Read GT
        if save_label_txt:
            path_gt_xml = os.path.join(dir_data_xml, id_file + ".xml")
            bb_gt_xyxy, labels = read_xml_page_gt(path_gt_xml)
            bb_gt_xyxy = torch.Tensor(bb_gt_xyxy)
            nb_line_gt = bb_gt_xyxy.shape[0]

        results = model(one_file, verbose=False, iou=iou_t, conf=conf_t, agnostic_nms=agnostic_nms, imgsz=img_size)

        bb_pred_xyxy = []

        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            # Not in correct order
            pred_bb = boxes.xywh

            # Sort by y center
            pred_bb_2, indices = torch.sort(pred_bb, dim=0)
            indices_y = indices[:, 1]

            pred_bb_sorted = pred_bb[indices_y]

            for x, y, w, h in pred_bb_sorted:
                # nb_line_pred += 1
                half_height = int(h.item() / 2.0)
                y1 = int(y.item() - half_height)
                y2 = int(y.item() + half_height)

                half_width = int(w.item() / 2.0)
                x1 = int(x.item() - half_width)
                x2 = int(x.item() + half_width)

                bb_pred_xyxy.append(torch.Tensor([x1, y1, x2, y2]))

            nb_line_pred = pred_bb_sorted.shape[0]

            if save_label_txt:
                if nb_line_pred == nb_line_gt:
                    nb_line_pred_ok += 1
                if nb_line_pred > nb_line_gt:
                    nb_line_pred_more += 1
                if nb_line_pred < nb_line_gt:
                    nb_line_pred_less += 1

            bb_pred_xyxy = torch.stack(bb_pred_xyxy)

        # Compute metrics
        print("nb line pred:" + str(nb_line_pred))

        # Save image
        index_line = 0
        # nb_gt_bb = bb_xywh_gt.shape[0]
        for one_pred_bb in bb_pred_xyxy:
            x_min = int(one_pred_bb[0])
            y_min = int(one_pred_bb[1])
            x_max = int(one_pred_bb[2])
            y_max = int(one_pred_bb[3])

            # Extract line img
            # dim img: heigh, width, channel
            img_line = img[y_min:y_max, x_min:x_max, :]
            path_img_line = os.path.join(save_img_line, id_file + "_" + str(index_line) + ".jpg")
            cv2.imwrite(path_img_line, img_line)
            index_line += 1

        # Save text label
        if save_label_txt:
            print("nb line gt:" + str(nb_line_gt))

            # Stats
            # ~sub segmentation
            if nb_line_gt > nb_line_pred:
                bb_gt_xyxy = bb_gt_xyxy[:nb_line_pred]

            nb_total_line += nb_line_gt

            nb_line_gt_eval = bb_gt_xyxy.shape[0]
            nb_total_line_eval += nb_line_gt_eval

            # Iou close gt
            iou_all_close = 0
            index_line = 0

            for one_pred_bb in bb_pred_xyxy:
                one_pred_bb_extend = one_pred_bb.repeat(nb_line_gt_eval, 1)

                all_iou = bbox_iou(bb_gt_xyxy, one_pred_bb_extend, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-07)
                best_iou_one = torch.max(all_iou)

                iou_all_find_closest += best_iou_one
                iou_all_close += best_iou_one

                if best_iou_one < 0.5:
                    print("IoU best small: " + str(best_iou_one))
                    counter_iou_small += 1
                else:
                    index_max = torch.argmax(all_iou)
                    gt_label = labels[index_max]

                    if gt_label is None:
                        dict_label_line_gt[id_file + "_" + str(index_line)] = ""
                        counter_gt_is_None += 1
                    else:
                        dict_label_line_gt[id_file + "_" + str(index_line)] = gt_label

                index_line += 1

            mean_iou_one = iou_all_close / nb_line_pred
            print(f"IoU closest: {100 * mean_iou_one:.2f}% ")

    if save_label_txt:
        print("----Summary-----")
        print("total_items: " + str(total_items))
        print("nb_total_line gt: " + str(nb_total_line))
        print("counter_iou_small: " + str(counter_iou_small))
        print("counter_gt_is_None: " + str(counter_gt_is_None))

        print("nb_line_pred_ok: " + str(nb_line_pred_ok))
        print("nb_line_pred_more: " + str(nb_line_pred_more))
        print("nb_line_pred_less: " + str(nb_line_pred_less))

        iou_all_find_closest /= nb_total_line_eval
        print(f"Mean IoU closest: {100 * iou_all_find_closest:.2f}% ")

        path_save_label = os.path.join(dir_save, "all_labels.json")

        json_object = json.dumps(dict_label_line_gt, indent=4)

        with open(path_save_label, "w") as outfile:
            outfile.write(json_object)


# need gt: yolo format, texts file in "labels.pkl"
def save_yolo_pred_iam_gt_format(dir_data_img,
                                 dir_label_yolo_page,
                                 path_all_label,
                                 dir_save,
                                 path_model,
                                 ext_img,
                                 img_size,
                                 iou_t,
                                 conf_t,
                                 agnostic_nms=True,
                                 save_label_txt=True):
    # Create directories result
    save_img_line = os.path.join(dir_save, "image_line")
    os.makedirs(save_img_line, exist_ok=True)

    # Load model
    model = YOLO(path_model)

    files_img = glob.glob(dir_data_img + '/**/*.' + ext_img, recursive=True)

    counter_iou_small = 0

    nb_line_pred_ok = 0
    nb_line_pred_more = 0
    nb_line_pred_less = 0

    total_items = 0
    nb_total_line = 0
    nb_total_line_eval = 0

    iou_all_find_closest = 0

    dict_label_line_gt = {}

    if save_label_txt:
        gt_txt = []
        with (open(path_all_label, "rb")) as openfile:
            while True:
                try:
                    gt_txt.append(pickle.load(openfile))
                except EOFError:
                    break

        gt_txt = gt_txt[0]["ground_truth"]

        all_labels = {}

        for split_name, dict_all_txt_split in gt_txt.items():
            for id_page, dict_txt in dict_all_txt_split.items():
                lines_infos = dict_txt["lines"]  # list of lines
                all_labels[id_page] = [one_line_info["text"] for one_line_info in lines_infos]  # keep only text

    for one_file in files_img:
        print(one_file)
        total_items += 1
        img = cv2.imread(one_file)

        height_page = img.shape[0]
        width_page = img.shape[1]

        # Get id file
        split_name = os.path.split(one_file)
        split_name = split_name[1].split(sep=".")  # Filename and extension
        id_file = split_name[0]

        # Read GT
        if save_label_txt:
            # Read txt labels
            labels = all_labels[id_file]

            # Read bb
            path_gt_yolo = os.path.join(dir_label_yolo_page, id_file + ".txt")
            # format: class x_center y_center width height
            bb_gt_xywh = read_yolo_gt(path_gt_yolo)

            # remove class index and unnormalize
            bb_gt_xyxy = []
            for one_p in bb_gt_xywh:
                x1 = (one_p[1] - one_p[3] / 2.0) * width_page
                y1 = (one_p[2] - one_p[4] / 2.0) * height_page

                x2 = (one_p[1] + one_p[3] / 2.0) * width_page
                y2 = (one_p[2] + one_p[4] / 2.0) * height_page

                bb_gt_xyxy.append(torch.Tensor([x1, y1, x2, y2]))

            bb_gt_xyxy = torch.stack(bb_gt_xyxy)
            nb_line_gt = bb_gt_xyxy.shape[0]

        results = model(one_file, verbose=False, iou=iou_t, conf=conf_t, agnostic_nms=agnostic_nms, imgsz=img_size)

        bb_pred_xyxy = []

        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            # Not in correct order
            pred_bb = boxes.xywh

            # Sort by y center
            pred_bb_2, indices = torch.sort(pred_bb, dim=0)
            indices_y = indices[:, 1]

            pred_bb_sorted = pred_bb[indices_y]

            for x, y, w, h in pred_bb_sorted:
                # int -> for coordinate in image
                half_height = int(h.item() / 2.0)
                y1 = int(y.item() - half_height)
                y2 = int(y.item() + half_height)

                half_width = int(w.item() / 2.0)
                x1 = int(x.item() - half_width)
                x2 = int(x.item() + half_width)

                bb_pred_xyxy.append(torch.Tensor([x1, y1, x2, y2]))

            nb_line_pred = pred_bb_sorted.shape[0]

            if save_label_txt:
                if nb_line_pred == nb_line_gt:
                    nb_line_pred_ok += 1
                if nb_line_pred > nb_line_gt:
                    nb_line_pred_more += 1
                if nb_line_pred < nb_line_gt:
                    nb_line_pred_less += 1

            bb_pred_xyxy = torch.stack(bb_pred_xyxy)

        # Compute metrics
        print("nb line pred:" + str(nb_line_pred))

        # Save image
        index_line = 0
        # nb_gt_bb = bb_xywh_gt.shape[0]
        for one_pred_bb in bb_pred_xyxy:
            x_min = int(one_pred_bb[0])
            y_min = int(one_pred_bb[1])
            x_max = int(one_pred_bb[2])
            y_max = int(one_pred_bb[3])

            # Extract line img
            # dim img: heigh, width, channel
            img_line = img[y_min:y_max, x_min:x_max, :]
            path_img_line = os.path.join(save_img_line, id_file + "_" + str(index_line) + ".jpg")
            cv2.imwrite(path_img_line, img_line)
            index_line += 1

        # Save text label
        if save_label_txt:
            print("nb line gt:" + str(nb_line_gt))

            # Stats
            # sub segmentation
            if nb_line_gt > nb_line_pred:
                bb_gt_xyxy = bb_gt_xyxy[:nb_line_pred]

            nb_total_line += nb_line_gt

            nb_line_gt_eval = bb_gt_xyxy.shape[0]
            nb_total_line_eval += nb_line_gt_eval

            # Iou close gt
            iou_all_close = 0
            index_line = 0

            for one_pred_bb in bb_pred_xyxy:
                one_pred_bb_extend = one_pred_bb.repeat(nb_line_gt_eval, 1)

                all_iou = bbox_iou(bb_gt_xyxy, one_pred_bb_extend, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-07)
                best_iou_one = torch.max(all_iou)

                iou_all_find_closest += best_iou_one
                iou_all_close += best_iou_one

                if best_iou_one < 0.5:
                    print(id_file)
                    print("IoU best small: " + str(best_iou_one))
                    print("Do not save. \n")
                    counter_iou_small += 1
                else:
                    index_max = torch.argmax(all_iou)
                    gt_label = labels[index_max]

                    dict_label_line_gt[id_file + "_" + str(index_line)] = gt_label

                index_line += 1

            mean_iou_one = iou_all_close / nb_line_pred
            print(f"IoU closest: {100 * mean_iou_one:.2f}% ")

    if save_label_txt:
        print("----Summary-----")
        print("total_items: " + str(total_items))
        print("nb_total_line gt: " + str(nb_total_line))
        print("counter_iou_small: " + str(counter_iou_small))

        print("nb_line_pred_ok: " + str(nb_line_pred_ok))
        print("nb_line_pred_more: " + str(nb_line_pred_more))
        print("nb_line_pred_less: " + str(nb_line_pred_less))

        iou_all_find_closest /= nb_total_line_eval
        print(f"Mean IoU closest: {100 * iou_all_find_closest:.2f}% ")

        path_save_label = os.path.join(dir_save, "all_labels.json")

        json_object = json.dumps(dict_label_line_gt, indent=4)

        with open(path_save_label, "w") as outfile:
            outfile.write(json_object)


def print_yolo_bb(dir_data_img, dir_save, path_model, ext_img, iou_t, conf_t):
    # Load model
    model = YOLO(path_model)

    files_img = glob.glob(dir_data_img + '/**/*.' + ext_img, recursive=True)

    agnostic_nms = True

    total_items = 0

    for one_file in files_img:
        print(one_file)
        total_items += 1
        img = cv2.imread(one_file)

        # Get id file
        split_name = os.path.split(one_file)
        split_name = split_name[1].split(sep=".")  # Filename and extension
        id_file = split_name[0]

        results = model(one_file, verbose=False, iou=iou_t, conf=conf_t, agnostic_nms=agnostic_nms)

        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            # Not in correct order
            pred_bb = boxes.xywh
            conf = boxes.conf

            # Sort by y center
            pred_bb_2, indices = torch.sort(pred_bb, dim=0)
            indices_y = indices[:, 1]

            pred_bb_sorted = pred_bb[indices_y]
            conf = conf[indices_y]

            for (x, y, w, h), c in zip(pred_bb_sorted, conf):
                # nb_line_pred += 1
                half_height = int(h.item() / 2.0)
                y1 = int(y.item() - half_height)
                y2 = int(y.item() + half_height)

                half_width = int(w.item() / 2.0)
                x1 = int(x.item() - half_width)
                x2 = int(x.item() + half_width)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

                print(f"conf line: {c:.4f}")

            nb_line_pred = pred_bb_sorted.shape[0]

        path_img = os.path.join(dir_save, id_file + ".jpg")
        cv2.imwrite(path_img, img)
        print("nb line pred:" + str(nb_line_pred))


def print_yolo_bb_and_gt(dir_data_img, dir_save, path_model, dir_gt_yolo, ext_img, iou_t, conf_t, img_size):
    # Load model
    model = YOLO(path_model)

    files_img = glob.glob(dir_data_img + '/**/*.' + ext_img, recursive=True)

    agnostic_nms = True

    total_items = 0

    for one_file in files_img:
        print(one_file)
        total_items += 1
        img = cv2.imread(one_file)

        height_page = img.shape[0]
        width_page = img.shape[1]

        # Get id file
        split_name = os.path.split(one_file)
        split_name = split_name[1].split(sep=".")  # Filename and extension
        id_file = split_name[0]

        path_gt_page = os.path.join(dir_gt_yolo, id_file + ".txt")
        yolo_gt = read_yolo_gt(path_gt_page)

        for one_gt in yolo_gt:
            x_c = one_gt[1] * width_page
            y_c = one_gt[2] * height_page
            width_line = one_gt[3] * width_page
            height_line = one_gt[4] * height_page

            half_height = int(height_line/ 2.0)
            y1 = int(y_c - half_height)
            y2 = int(y_c + half_height)

            half_width = int(width_line / 2.0)
            x1 = int(x_c - half_width)
            x2 = int(x_c + half_width)

            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=4)

        results = model(one_file, verbose=False, iou=iou_t, conf=conf_t, agnostic_nms=agnostic_nms, imgsz=img_size)

        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            # Not in correct order
            pred_bb = boxes.xywh
            conf = boxes.conf

            # Sort by y center
            pred_bb_2, indices = torch.sort(pred_bb, dim=0)
            indices_y = indices[:, 1]

            pred_bb_sorted = pred_bb[indices_y]
            conf = conf[indices_y]

            for (x, y, w, h), c in zip(pred_bb_sorted, conf):
                # nb_line_pred += 1
                half_height = int(h.item() / 2.0)
                y1 = int(y.item() - half_height)
                y2 = int(y.item() + half_height)

                half_width = int(w.item() / 2.0)
                x1 = int(x.item() - half_width)
                x2 = int(x.item() + half_width)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

                print(f"conf line: {c:.4f}")

            nb_line_pred = pred_bb_sorted.shape[0]

        path_img = os.path.join(dir_save, id_file + ".jpg")
        cv2.imwrite(path_img, img)
        print("nb line pred:" + str(nb_line_pred))
        print("nb line gt:" + str(len(yolo_gt)))


if __name__ == "__main__":
    ext_img = "jpg"
    save_label_txt = True
    path_model = "C:/Users/simcor/dev/logs/htr-page/iam/line_detection/2025-07-22_ID_13839591/2025-07-22_18_56_58_845113/weights/best.pt"

    # IAM
    img_size = 960
    iou_t = 0.4
    conf_t = 0.7
    dir_db_iam = "C:/Users/simcor/dev/data/IAM/paragraph_new/"
    path_all_label = os.path.join(dir_db_iam, "labels.pkl")

    dir_save_yolo = "C:/Users/simcor/dev/logs/htr-page/iam/line_detection/2025-07-22_ID_13839591/pred_line_yolo/"
    os.makedirs(dir_save_yolo, exist_ok=True)

    # Debug
    # print case iou < 50
    dir_data_img_debug = os.path.join(dir_db_iam, "debug_img")
    dir_save_debug = os.path.join(dir_save_yolo, "debug")
    # print_yolo_bb(dir_data_img_debug, dir_save_debug, path_model, "png", iou_t, conf_t)

    dir_gt_yolo = os.path.join(dir_db_iam, "train", "labels_bb")  # train, test
    # print_yolo_bb_and_gt(dir_data_img_debug, dir_save_debug, path_model, dir_gt_yolo, "png", iou_t, conf_t)

    # # Train
    # dir_data_img = os.path.join(dir_db_iam, "train", "images")
    # dir_label_yolo_page = os.path.join(dir_db_iam, "train", "labels_bb")
    # dir_save = os.path.join(dir_save_yolo, "train")
    #
    # save_yolo_pred_iam_gt_format(dir_data_img, dir_label_yolo_page, path_all_label, dir_save, path_model, "png",
    #                              img_size,
    #                              iou_t,
    #                              conf_t)

    # # Validation
    # dir_data_img = os.path.join(dir_db_iam, "valid", "images")
    # dir_label_yolo_page = os.path.join(dir_db_iam, "valid", "labels_bb")
    # dir_save = os.path.join(dir_save_yolo, "valid")
    #
    # save_yolo_pred_iam_gt_format(dir_data_img, dir_label_yolo_page, path_all_label, dir_save, path_model, "png",
    #                              img_size,
    #                              iou_t,
    #                              conf_t)

    # # Test
    # dir_data_img = os.path.join(dir_db_iam, "test", "images")
    # dir_label_yolo_page = os.path.join(dir_db_iam, "test", "labels_bb")
    # dir_save = os.path.join(dir_save_yolo, "test")
    #
    # save_yolo_pred_iam_gt_format(dir_data_img, dir_label_yolo_page, path_all_label, dir_save, path_model, "png",
    #                              img_size,
    #                              iou_t,
    #                              conf_t)

    # Read 2016
    ext_img = "jpg"
    path_model = "C:/Users/simcor/dev/logs/htr-page/read2016/line_detection/2025-07-23_ID_13851804/2025-07-23_15_31_53_742003/weights/best.pt"

    img_size = 960
    iou_t = 0.4
    conf_t = 0.3
    dir_db_read = "C:/Users/simcor/dev/data/READ/2016/"
    # path_all_label = os.path.join(dir_db_iam, "labels.pkl")

    dir_save_read = "C:/Users/simcor/dev/logs/htr-page/read2016/line_detection/2025-07-23_ID_13851804/pred_line_yolo/"
    os.makedirs(dir_save_read, exist_ok=True)

    # debug
    dir_data_img_debug = os.path.join(dir_db_read, "debug_img")
    dir_save_debug = os.path.join(dir_save_read, "debug")
    os.makedirs(dir_save_debug, exist_ok=True)
    dir_gt_yolo = os.path.join(dir_db_read, "convert_line_polygon_val", "label_yolo")
    # print_yolo_bb_and_gt(dir_data_img_debug, dir_save_debug, path_model, dir_gt_yolo, ext_img, iou_t, conf_t, img_size)

    # # # Train
    dir_data_img = os.path.join(dir_db_read, "convert_line_polygon_train", "image_page")
    dir_save = os.path.join(dir_save_read, "train")
    dir_data_xml_all = "C:/Users/simcor/dev/data/READ/2016/origin/Training/page/page/"
    # save_yolo_pred(dir_data_img, dir_data_xml_all, dir_save, path_model, ext_img, img_size, iou_t, conf_t, save_label_txt)

    # # # Validation
    # dir_data_img = os.path.join(dir_db_read, "convert_line_polygon_val", "image_page")
    # # dir_label_yolo_page = os.path.join(dir_db_iam, "valid", "labels_bb")
    # dir_save = os.path.join(dir_save_read, "valid")
    # dir_data_xml_all = "C:/Users/simcor/dev/data/READ/2016/origin/Validation/page/page/"
    # save_yolo_pred(dir_data_img, dir_data_xml_all, dir_save, path_model, ext_img, img_size, iou_t, conf_t, save_label_txt)

    # # Test
    dir_data_img = os.path.join(dir_db_read, "convert_line_polygon_test", "image_page")
    dir_save = os.path.join(dir_save_read, "test")
    dir_data_xml_all = "C:/Users/simcor/dev/data/READ/2016/origin/Test/page/"
    # save_yolo_pred(dir_data_img, dir_data_xml_all, dir_save, path_model, ext_img, img_size, iou_t, conf_t, save_label_txt)

    # # # ALL
    # dir_data_xml_all = "C:/Users/simcor/dev/data/HTR-Simancas/origin/Track2/PAGE"
    #
    # dir_save = "C:/Users/simcor/dev/data/HTR-Simancas/13324653_yolo_lines_best_grid_search/"
    # os.makedirs(dir_save, exist_ok=True)
    #
    # # Train
    # dir_data_img = "C:/Users/simcor/dev/data/HTR-Simancas/Track2_split/train_berzelius/image_page/"
    # dir_save_train = os.path.join(dir_save, "train")
    #
    # save_yolo_pred(dir_data_img, dir_data_xml_all, dir_save_train, path_model, ext_img, save_label_txt)
    #
    # # Validation
    # dir_data_img = "C:/Users/simcor/dev/data/HTR-Simancas/Track2_split/validation_berzelius/image_page/"
    # dir_save_val = os.path.join(dir_save, "validation")
    #
    # save_yolo_pred(dir_data_img, dir_data_xml_all, dir_save_val, path_model, ext_img, save_label_txt)

    # dir_data_img = "C:/Users/simcor/dev/data/Historian_Dag_Avango/origin/"
    # dir_save = "C:/Users/simcor/dev/data/Historian_Dag_Avango/pred/"
    # os.makedirs(dir_save, exist_ok=True)
    #
    # # save_yolo_pred(dir_data_img, "", dir_save, path_model, ext_img, save_label_txt=False)
    # print_yolo_bb(dir_data_img, dir_save, path_model, ext_img, iou_t=0.52, conf_t=0.14)
    #
    # print("End")

