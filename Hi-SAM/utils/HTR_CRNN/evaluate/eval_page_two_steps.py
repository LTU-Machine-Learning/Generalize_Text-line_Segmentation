import time
import glob
import os
import re
from math import floor

import editdistance
import numpy as np
from skimage import io
import torch
import cv2

from utils.HTR_CRNN.data.globalvalue.text_comon_values import BLANK_STR_TOKEN
from utils.HTR_CRNN.data.image.preprocess_img import preprocess_img_line
from utils.HTR_CRNN.data.readers.read_xml_page import read_xml_page_gt
from utils.HTR_CRNN.data.sorter.sort_bb_label import get_indexes_sort_bb_multiple_columns
from utils.HTR_CRNN.data.text.best_path import ctc_best_path_one
from utils.HTR_CRNN.evaluate.metrics.bag_of_word_wer import compute_bwer_one_sample


def eval_page_level(ext_img, dir_img, dir_label, model_reco, model_yolo, iou_t, conf_t, agnostic_nms, imgsz,
                    fixed_size_img_line, pad_left_line, pad_right_line, char_list, char_dict,
                    x_reduced_len, device, log_dir, save_prediction):
    # To refactor: split into smaller function
    model_reco.eval()

    begin_train = time.time()

    time_all_yolo = 0
    time_all_crnn = 0
    nb_page = 0

    cer_all_norm_nb_letter = 0
    nb_total_letter = 0

    cer_all_norm_nb_letter_filter_return_line = 0

    wer_all_norm_nb_words = 0
    nb_total_words = 0

    wer_all_norm_nb_words_dan = 0
    nb_total_words_dan = 0

    all_gt_txt = ""
    all_pred_txt = ""

    # Process img by img
    if ext_img == "pngjpg":
        files_img = glob.glob(dir_img + '/**/*.png', recursive=True)

        files_img.extend(glob.glob(dir_img + '/**/*.jpg', recursive=True))

    else:
        files_img = glob.glob(dir_img + '/**/*.' + ext_img, recursive=True)

    for one_file in files_img:
        nb_page += 1
        # Get id file
        split_name = os.path.split(one_file)
        split_name = split_name[1].split(sep=".")  # Filename and extension
        id_file = split_name[0]
        print(id_file)

        path_label = os.path.join(dir_label, id_file + ".xml")

        if not os.path.isfile(path_label):
            print(one_file)
            print("label text associated doesn't exist. Data not loaded")
            continue

        img_page = io.imread(one_file, as_gray=True)
        max_v = np.max(img_page)

        # Color image are converted to grayscale -> value [0 ; 1]
        # Grayscale image are not converted -> value [0 ; 255]
        if max_v < 1:
            img_page *= 255.0

        list_img_line_gt = []

        bb_xyxy, labels = read_xml_page_gt(path_label)
        i_s = get_indexes_sort_bb_multiple_columns(bb_xyxy)

        labels_sorted = [labels[i] for i in i_s]

        if save_prediction == 1:
            img = cv2.imread(one_file)

            bb_gt_sorted = [bb_xyxy[i] for i in i_s]

            index_line = 0
            for one_bb in bb_gt_sorted:
                x1 = int(one_bb[0])
                y1 = int(one_bb[1])
                x2 = int(one_bb[2])
                y2 = int(one_bb[3])

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 230, 0), thickness=4)  # Green
                img = cv2.putText(img, str(index_line), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

                index_line += 1
                img_line = img_page[y1:y2, x1:x2]
                list_img_line_gt.append(img_line)

        # Detect line with yolo
        begin_yolo = time.time()
        results_yolo = model_yolo(one_file,
                                  verbose=False,
                                  iou=iou_t,
                                  conf=conf_t,
                                  agnostic_nms=agnostic_nms,
                                  imgsz=imgsz)

        # Extract images line
        # CRNN train with grayscale
        list_img_line = []

        # List of one
        for result in results_yolo:
            boxes = result.boxes  # Boxes object for bounding box outputs

            # # Not order
            pred_bb = boxes.xyxy
            pred_bb = [one_bb_t.tolist() for one_bb_t in pred_bb]

            # Sort multi columns
            i_s = get_indexes_sort_bb_multiple_columns(pred_bb)
            pred_bb_sorted = [pred_bb[i] for i in i_s]

            index_line = 0
            for x1, y1, x2, y2 in pred_bb_sorted:
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                img_line = img_page[y1:y2, x1:x2]
                list_img_line.append(img_line)

                if save_prediction == 1:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
                    img = cv2.putText(img, str(index_line), (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                index_line += 1

        end_yolo = time.time()
        time_all_yolo += end_yolo - begin_yolo

        # Recognition line with CRNN
        begin_crnn = time.time()
        pred_page_txt = []

        index_line = 0     # list_img_line    list_img_line_gt

        for one_line_img in list_img_line:
            # Resize line
            img_line = preprocess_img_line(one_line_img, fixed_size_img_line, pad_left_line, pad_right_line)
            img_tensor = torch.as_tensor(img_line, dtype=torch.float32)
            img_tensor = img_tensor.to(device)

            # # # # Debug debug_dir
            # path_save_img = os.path.join(log_dir, id_file + "_" + str(index_line) + "_preprocess.jpg")
            # save_image(img_tensor, path_save_img)

            img_tensor = img_tensor.unsqueeze(0)  # Add channel dim
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dim

            y_pred, _, _ = model_reco(img_tensor)
            output, aux_output = y_pred

            # Main head
            output_log = torch.nn.functional.log_softmax(output, dim=-1)

            # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
            output_log = output_log.transpose(0, 1)

            top = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len] for j, lp in enumerate(output_log)]
            predictions_text = [ctc_best_path_one(p, char_list, char_dict[BLANK_STR_TOKEN]) for p in top]
            predictions_text = [t.strip() for t in predictions_text]  # Remove text padding

            # Batch of one element
            pred_page_txt.append(predictions_text[0])

            # # Debug
            # if index_line < len(labels_sorted):
            #     label_gt = labels_sorted[index_line]
            #     label_gt = text_read.remove_space_before_after_one_item(label_gt)
            #     cer_line = editdistance.eval(label_gt, predictions_text[0])
            #     cer_all_line_level += cer_line
            #
            #     nb_char_line_gt = len(labels_sorted[index_line])
            #     nb_letter_all_line_letter += nb_char_line_gt
            #
            #     cer_line /= nb_char_line_gt
            #     print("Ground truth : " + label_gt)
            #     print("Predictions  : " + predictions_text[0])
            #     print(f"CER line: {100 * cer_line:.2f}% ")

            index_line += 1

        # Concatenate
        pred_page_txt = " ".join(pred_page_txt)
        end_crnn = time.time()
        time_all_crnn += end_crnn - begin_crnn

        gt_page_txt = " ".join(labels_sorted)

        # Remove double space
        gt_page_txt = gt_page_txt.replace("  ", " ")
        pred_page_txt = pred_page_txt.replace("  ", " ")

        # Evaluation
        # Cf. DAN git  -> char only in val
        gt_page_txt = re.sub('([ ])+', " ", gt_page_txt).strip()
        pred_page_txt = re.sub('([ ])+', " ", pred_page_txt).strip()

        print("Gt:")
        print(gt_page_txt)
        print("Pred:")
        print(pred_page_txt)

        nb_char_gt = len(gt_page_txt)

        cer_no_fitler = editdistance.eval(gt_page_txt, pred_page_txt)
        cer_all_norm_nb_letter += cer_no_fitler
        nb_total_letter += nb_char_gt

        cer_no_fitler /= nb_char_gt

        all_gt_txt += gt_page_txt + " "
        all_pred_txt += pred_page_txt + " "

        gt_page_split = gt_page_txt.split(" ")
        pred_page_split = pred_page_txt.split(" ")

        nb_word_gt = len(gt_page_split)
        wer = editdistance.eval(gt_page_split, pred_page_split)

        wer_all_norm_nb_words += wer
        nb_total_words += nb_word_gt

        wer /= nb_word_gt

        b_wer = compute_bwer_one_sample(gt_page_txt, pred_page_txt)

        # Other variant of WER from DAN git
        gt_page_txt = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', gt_page_txt)  # punctuation processed as word
        pred_page_txt = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', pred_page_txt)  # punctuation processed as word

        gt_page_split = gt_page_txt.split(" ")
        pred_page_split = pred_page_txt.split(" ")

        nb_word_gt_2 = len(gt_page_split)
        wer_2 = editdistance.eval(gt_page_split, pred_page_split)

        wer_all_norm_nb_words_dan += wer_2
        nb_total_words_dan += nb_word_gt_2

        wer_2 /= nb_word_gt_2

        print(f"CER: {100 * cer_no_fitler:.2f}% ")
        print(f"WER: {100 * wer:.2f}% ")
        print(f"WER DAN: {100 * wer_2:.2f}% ")
        print(f"B_WER: {100 * b_wer:.2f}% ")
        print()

        if save_prediction == 1:
            filename = os.path.join(log_dir, id_file + "_pred.png")
            cv2.imwrite(filename, img)

    end_train = time.time()

    t_one_page_yolo = time_all_yolo / nb_page
    t_one_page_crnn = time_all_crnn / nb_page

    print("nb_page: " + str(nb_page))
    print("Time yolo one page (s): " + str(t_one_page_yolo))
    print("Time crnn one page (s): " + str(t_one_page_crnn))
    print("Time yolo (s): " + str(time_all_yolo))
    print("Time crnn (s): " + str(time_all_crnn))
    print("Time all (s): " + str((end_train - begin_train)))

    b_wer_all_norm_nb_word = compute_bwer_one_sample(all_gt_txt, all_pred_txt)

    cer_all_norm_nb_letter /= nb_total_letter
    wer_all_norm_nb_words /= nb_total_words
    wer_all_norm_nb_words_dan /= nb_total_words_dan

    print()
    print("Performance all:")
    print(f"CER: {100 * cer_all_norm_nb_letter:.2f}% ")
    print(f"WER : {100 * wer_all_norm_nb_words:.2f}% ")
    print(f"WER DAN : {100 * wer_all_norm_nb_words_dan:.2f}% ")
    print(f"B_WER: {100 * b_wer_all_norm_nb_word:.2f}% ")
    print()

    return cer_all_norm_nb_letter_filter_return_line


def eval_page_level_batch_line(ext_img, dir_img, dir_label, model_reco, model_yolo, iou_t, conf_t, agnostic_nms, imgsz,
                    fixed_size_img_line, pad_left_line, pad_right_line, char_list, char_dict,
                    x_reduced_len, device, log_dir, save_prediction):
    # To refactor: split into smaller function
    model_reco.eval()

    begin_train = time.time()

    time_all_yolo = 0
    time_all_crnn = 0
    nb_page = 0

    cer_all_norm_nb_letter = 0
    nb_total_letter = 0

    cer_all_norm_nb_letter_filter_return_line = 0

    wer_all_norm_nb_words = 0
    nb_total_words = 0

    wer_all_norm_nb_words_dan = 0
    nb_total_words_dan = 0

    all_gt_txt = ""
    all_pred_txt = ""

    # Process img by img
    if ext_img == "pngjpg":
        files_img = glob.glob(dir_img + '/**/*.png', recursive=True)

        files_img.extend(glob.glob(dir_img + '/**/*.jpg', recursive=True))

    else:
        files_img = glob.glob(dir_img + '/**/*.' + ext_img, recursive=True)

    for one_file in files_img:
        nb_page += 1
        # Get id file
        split_name = os.path.split(one_file)
        split_name = split_name[1].split(sep=".")  # Filename and extension
        id_file = split_name[0]
        print(id_file)

        path_label = os.path.join(dir_label, id_file + ".xml")

        if not os.path.isfile(path_label):
            print(one_file)
            print("label text associated doesn't exist. Data not loaded")
            continue

        img_page = io.imread(one_file, as_gray=True)
        max_v = np.max(img_page)

        # Color image are converted to grayscale -> value [0 ; 1]
        # Grayscale image are not converted -> value [0 ; 255]
        if max_v < 1:
            img_page *= 255.0

        list_img_line_gt = []

        bb_xyxy, labels = read_xml_page_gt(path_label)
        i_s = get_indexes_sort_bb_multiple_columns(bb_xyxy)

        labels_sorted = [labels[i] for i in i_s]

        if save_prediction == 1:
            img = cv2.imread(one_file)

            bb_gt_sorted = [bb_xyxy[i] for i in i_s]

            index_line = 0
            for one_bb in bb_gt_sorted:
                x1 = int(one_bb[0])
                y1 = int(one_bb[1])
                x2 = int(one_bb[2])
                y2 = int(one_bb[3])

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 230, 0), thickness=4)  # Green
                img = cv2.putText(img, str(index_line), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

                index_line += 1
                img_line = img_page[y1:y2, x1:x2]
                list_img_line_gt.append(img_line)

        # Detect line with yolo
        begin_yolo = time.time()
        results_yolo = model_yolo(one_file,
                                  verbose=False,
                                  iou=iou_t,
                                  conf=conf_t,
                                  agnostic_nms=agnostic_nms,
                                  imgsz=imgsz)

        # Extract images line
        # CRNN train with grayscale
        batch_img_line = []

        # List of one
        for result in results_yolo:
            boxes = result.boxes  # Boxes object for bounding box outputs

            # # Not order
            pred_bb = boxes.xyxy
            pred_bb = [one_bb_t.tolist() for one_bb_t in pred_bb]

            # Sort multi columns
            i_s = get_indexes_sort_bb_multiple_columns(pred_bb)
            pred_bb_sorted = [pred_bb[i] for i in i_s]

            for x1, y1, x2, y2 in pred_bb_sorted:
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                # x1 = floor(x1)
                # y1 = floor(y1)
                # x2 = floor(x2)
                # y2 = floor(y2)

                img_line = img_page[y1:y2, x1:x2]
                # Resize line
                img_line = preprocess_img_line(img_line, fixed_size_img_line, pad_left_line, pad_right_line)
                img_tensor = torch.as_tensor(img_line, dtype=torch.float32)
                img_tensor = img_tensor.unsqueeze(0)  # Add channel dim
                batch_img_line.append(img_tensor)

                if save_prediction == 1:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
                    img = cv2.putText(img, str(index_line), (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        end_yolo = time.time()
        time_all_yolo += end_yolo - begin_yolo

        batch_img_line = torch.stack(batch_img_line)
        batch_img_line = batch_img_line.to(device)

        # Recognition line with CRNN
        begin_crnn = time.time()
        pred_page_txt = []

        y_pred, _, _ = model_reco(batch_img_line)
        output, aux_output = y_pred

        # Main head
        output_log = torch.nn.functional.log_softmax(output, dim=-1)

        # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
        output_log = output_log.transpose(0, 1)

        top = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len] for j, lp in enumerate(output_log)]
        predictions_text = [ctc_best_path_one(p, char_list, char_dict[BLANK_STR_TOKEN]) for p in top]
        predictions_text = [t.strip() for t in predictions_text]  # Remove text padding

        # Batch of one element
        for one_line_pred in predictions_text:
            pred_page_txt.append(one_line_pred)

        # Concatenate
        pred_page_txt = " ".join(pred_page_txt)
        end_crnn = time.time()
        time_all_crnn += end_crnn - begin_crnn

        gt_page_txt = " ".join(labels_sorted)

        # Remove double space
        gt_page_txt = gt_page_txt.replace("  ", " ")
        pred_page_txt = pred_page_txt.replace("  ", " ")

        # Evaluation
        # Cf. DAN git  -> char only in val
        gt_page_txt = re.sub('([ ])+', " ", gt_page_txt).strip()
        pred_page_txt = re.sub('([ ])+', " ", pred_page_txt).strip()

        print("Gt:")
        print(gt_page_txt)
        print("Pred:")
        print(pred_page_txt)

        nb_char_gt = len(gt_page_txt)

        cer_no_fitler = editdistance.eval(gt_page_txt, pred_page_txt)
        cer_all_norm_nb_letter += cer_no_fitler
        nb_total_letter += nb_char_gt

        cer_no_fitler /= nb_char_gt

        all_gt_txt += gt_page_txt + " "
        all_pred_txt += pred_page_txt + " "

        gt_page_split = gt_page_txt.split(" ")
        pred_page_split = pred_page_txt.split(" ")

        nb_word_gt = len(gt_page_split)
        wer = editdistance.eval(gt_page_split, pred_page_split)

        wer_all_norm_nb_words += wer
        nb_total_words += nb_word_gt

        wer /= nb_word_gt

        b_wer = compute_bwer_one_sample(gt_page_txt, pred_page_txt)

        # Other variant of WER from DAN git
        gt_page_txt = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', gt_page_txt)  # punctuation processed as word
        pred_page_txt = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', pred_page_txt)  # punctuation processed as word

        gt_page_split = gt_page_txt.split(" ")
        pred_page_split = pred_page_txt.split(" ")

        nb_word_gt_2 = len(gt_page_split)
        wer_2 = editdistance.eval(gt_page_split, pred_page_split)

        wer_all_norm_nb_words_dan += wer_2
        nb_total_words_dan += nb_word_gt_2

        wer_2 /= nb_word_gt_2

        print(f"CER: {100 * cer_no_fitler:.2f}% ")
        print(f"WER: {100 * wer:.2f}% ")
        print(f"WER DAN: {100 * wer_2:.2f}% ")
        print(f"B_WER: {100 * b_wer:.2f}% ")
        print()

        if save_prediction == 1:
            filename = os.path.join(log_dir, id_file + "_pred.png")
            cv2.imwrite(filename, img)

    end_train = time.time()

    t_one_page_yolo = time_all_yolo / nb_page
    t_one_page_crnn = time_all_crnn / nb_page

    print("nb_page: " + str(nb_page))
    print("Time yolo + preprocess line one page (s): " + str(t_one_page_yolo))
    print("Time crnn one page (s): " + str(t_one_page_crnn))
    print("Time yolo (s): " + str(time_all_yolo))
    print("Time crnn (s): " + str(time_all_crnn))
    print("Time all (s): " + str((end_train - begin_train)))

    b_wer_all_norm_nb_word = compute_bwer_one_sample(all_gt_txt, all_pred_txt)

    cer_all_norm_nb_letter /= nb_total_letter
    wer_all_norm_nb_words /= nb_total_words
    wer_all_norm_nb_words_dan /= nb_total_words_dan

    print()
    print("Performance all:")
    print(f"CER: {100 * cer_all_norm_nb_letter:.2f}% ")
    print(f"WER : {100 * wer_all_norm_nb_words:.2f}% ")
    print(f"WER DAN : {100 * wer_all_norm_nb_words_dan:.2f}% ")
    print(f"B_WER: {100 * b_wer_all_norm_nb_word:.2f}% ")
    print()

    return cer_all_norm_nb_letter_filter_return_line
