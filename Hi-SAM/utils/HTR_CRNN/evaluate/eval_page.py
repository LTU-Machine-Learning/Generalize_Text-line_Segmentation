import argparse
import faulthandler
import glob
import os
import time
from math import floor

import editdistance
import torch
import cv2
from skimage.color import rgb2gray
from ultralytics import YOLO
from skimage import io

from utils.HTR_CRNN.data.globalvalue.text_comon_values import BLANK_STR_TOKEN
from utils.HTR_CRNN.data.image.preprocess_img import preprocess_img_line
from utils.HTR_CRNN.data.readers.read_config import read_json_config_line_htr, read_labels_files, read_json_confi_page
from utils.HTR_CRNN.data.readers.read_txt_file import read_page_txt_gt
from utils.HTR_CRNN.data.text.best_path import ctc_best_path_one
from utils.HTR_CRNN.data.text.charset_token import CharsetToken
from utils.HTR_CRNN.models.crnn.crnn import CRNN
from utils.HTR_CRNN.models.utils.load_model import load_pretrained_model

parser = argparse.ArgumentParser()

parser.add_argument("config_file")

# Line detector
parser.add_argument("--path_model_yolo", default="", type=str)
parser.add_argument('--iou_t', default=0.5, type=float)
parser.add_argument('--conf_t', default=0.2, type=float)
parser.add_argument('--agnostic_nms', default=1, type=float)
parser.add_argument('--imgsz', default=960, type=int)

# Line recognition
parser.add_argument("--path_model_crnn", default="", type=str)
parser.add_argument('--height_max_line', default=70, type=int)
parser.add_argument('--width_max_line', default=600, type=int)
parser.add_argument('--pad_left_line', default=32, type=int)
parser.add_argument('--pad_right_line', default=32, type=int)
# parser.add_argument('--resize_config', type=lambda tw: ResizeInputPolicy[tw], choices=list(ResizeInputPolicy),
#                     default=ResizeInputPolicy.DAS)
# CRNN Model
# parser.add_argument('--dropout_last_fc_crnn', default=0.2, type=float)
# parser.add_argument('--dropout_lstm', default=0.2, type=float)
#
# parser.add_argument('--activation_fct', type=lambda tw: Activationfunction[tw], choices=list(Activationfunction),
#                     default=Activationfunction.RELU)
# parser.add_argument('--encoder_config', type=lambda tw: EncoderConf[tw], choices=list(EncoderConf),
#                     default=EncoderConf.BASE)
# parser.add_argument('--add_squeeze_excitation', default=0, type=int)

print("===============================================================================")

begin = time.time()
args = parser.parse_args()
print(args)

faulthandler.enable()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device :")
print(device)
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
print("torch.cuda.device_count(): " + str(torch.cuda.device_count()))

# Paths
# _, val_info, test_info, charsets_path, dir_wandb = read_json_config(args.config_file)
_, val_info, test_info, charsets_path, dir_wandb = read_json_confi_page(args.config_file)

# Alphabet
charset = CharsetToken(charsets_path, use_blank=True)
char_list = charset.get_charset_list()
char_dict = charset.get_charset_dictionary()

print("Alphabet:")
print(char_list)

# Model
model_yolo = YOLO(args.path_model_yolo)

agnostic_nms = False
if args.agnostic_nms == 1:
    agnostic_nms = True

# CRNN
cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]
head_cfg = (256, 3)  # (hidden , num_layers)
width_divisor = 8
model_reco = CRNN(cnn_cfg,
                  head_cfg,
                  charset.get_nb_char(),
                  load_img_as_grayscale=args.load_img_as_grayscale)

# Data
fixed_size_img_line = (args.height_max_line, args.width_max_line)
width_with_pad = args.width_max_line + args.pad_left_line + args.pad_right_line
x_reduced_len = floor(width_with_pad / width_divisor)

if os.path.isfile(args.path_model_crnn):
    load_pretrained_model(args.path_model_crnn, model_reco, device)

print(f"Transferring model to {str(device)}...")
model_reco = model_reco.to(device)

number_parameters_reco = sum(p.numel() for p in model_reco.parameters() if p.requires_grad)
print(f"Recognition model has {number_parameters_reco:,} trainable parameters.")

# tmp = model_yolo.model.model
# for name, param in tmp.named_parameters():
#     print(name, param)
#
# for titi in tmp:
#    print(titi)
# number_parameters_detector = sum(p.numel() for p in model_yolo.model.model if p.requires_grad_)
# print(f"Line detection has {number_parameters_detector:,} trainable parameters.")
#
# total_parameters = number_parameters_reco + number_parameters_detector
# print(f"total_parameters: {total_parameters:,}")

model_reco.eval()
print_summary = ""

begin_train = time.time()

time_all_yolo = 0
time_all_crnn = 0
nb_page = 0

margin_x = 64

debug_dir = "C:/Users/simcor/dev/logs/htr-page/iam/reco/debug/"

print("Validation: ")
cer_all_norm_nb_letter = 0
nb_total_letter = 0
all_cer = []
all_wer = []
all_b_wer = []
# Separate validation sets
for one_db in val_info:
    print("--------------Evaluate-------------------------------")
    print(one_db[0])

    dir_img = one_db[1]
    dir_label = one_db[2]
    ext_img = one_db[3]

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

        # YOLO with color image
        path_label = os.path.join(dir_label, id_file + ".txt")

        if not os.path.isfile(path_label):
            print(one_file)
            print("label text associated doesn't exist. Data not loaded")
            continue

        gt_page_txt = read_page_txt_gt(path_label)

        # Detect line with yolo
        begin_yolo = time.time()
        results_yolo = model_yolo(one_file,
                                  verbose=False,
                                  iou=args.iou_t,
                                  conf=args.conf_t,
                                  agnostic_nms=agnostic_nms,
                                  imgsz=args.imgsz)

        # Extract images line
        # CRNN train with grayscale
        img_page = io.imread(one_file, as_gray=True)

        img_page_height = img_page.shape[0]
        img_page_width = img_page.shape[1]
        pred_bb_xyxy = []
        list_img_line = []

        index_line = 0
        # List of one
        for result in results_yolo:
            boxes = result.boxes  # Boxes object for bounding box outputs

            # Not in correct y order
            pred_bb = boxes.xywh

            # Sort by y
            pred_bb_2, indices = torch.sort(pred_bb, dim=0)
            indices_y = indices[:, 1]
            pred_bb_sorted = pred_bb[indices_y]
            # print(pred_bb_sorted)

            # index_line = 0
            for x, y, w, h in pred_bb_sorted:
                half_width = int(w.item() / 2.0)
                x1 = int(x.item() - half_width)
                x2 = int(x.item() + half_width)

                x2 = min(x2 + margin_x, img_page_width)

                half_height = int(h.item() / 2.0)
                y1 = int(y.item() - half_height)
                y2 = int(y.item() + half_height)

                img_line = img_page[y1:y2, x1:x2]
                list_img_line.append(img_line)

                # # Debug debug_dir
                # path_save_img = os.path.join(debug_dir, id_file + "_" + str(index_line) + ".jpg")
                # index_line += 1
                # cv2.imwrite(path_save_img, img_line)
        end_yolo = time.time()
        time_all_yolo += end_yolo - begin_yolo

        # Recognition line with CRNN
        begin_crnn = time.time()
        all_text_pred = []

        for one_line_img in list_img_line:
            # Resize line
            img_line = preprocess_img_line(one_line_img, fixed_size_img_line, args.pad_left_line, args.pad_right_line)
            img_tensor = torch.as_tensor(img_line, dtype=torch.float32)
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
            all_text_pred.append(predictions_text[0])

        # Concatenate
        all_text_pred = " ".join(all_text_pred)  # link between lines
        end_crnn = time.time()
        time_all_crnn += end_crnn - begin_crnn

        print("Gt:")
        print(gt_page_txt)
        print("Pred:")
        print(all_text_pred)

        # Evaluation
        nb_char_gt = len(gt_page_txt)

        cer = editdistance.eval(gt_page_txt, all_text_pred)
        cer_all_norm_nb_letter += cer
        nb_total_letter += nb_char_gt

        cer /= nb_char_gt

        gt_page_split = gt_page_txt.split(" ")
        pred_page_split = all_text_pred.split(" ")

        nb_word_gt = len(gt_page_split)
        wer = editdistance.eval(gt_page_split, pred_page_split)
        wer /= nb_word_gt

        b_wer = compute_bwer_one_sample(gt_page_txt, all_text_pred)

        print(f"CER: {100 * cer:.2f}% ")
        print(f"WER: {100 * wer:.2f}% ")
        print(f"B_WER: {100 * b_wer:.2f}% ")
        print()

        all_cer.append(cer)
        all_wer.append(wer)
        all_b_wer.append(b_wer)

end_train = time.time()

t_one_page_yolo = time_all_yolo / nb_page
t_one_page_crnn = time_all_crnn / nb_page

print("nb_page: " + str(nb_page))
print("Time yolo one page (s): " + str(t_one_page_yolo))
print("Time crnn one page (s): " + str(t_one_page_crnn))
print("Time yolo (s): " + str(time_all_yolo))
print("Time crnn (s): " + str(time_all_crnn))
print("Time all (s): " + str((end_train - begin_train)))

all_cer = sum(all_cer) / len(all_cer)
all_wer = sum(all_wer) / len(all_wer)
all_b_wer = sum(all_b_wer) / len(all_b_wer)

cer_all_norm_nb_letter /= nb_total_letter

print()
print("Performance all:")
print(f"CER norm nb letter: {100 * cer_all_norm_nb_letter:.2f}% ")
print(f"CER: {100 * all_cer:.2f}% ")
print(f"WER: {100 * all_wer:.2f}% ")
print(f"B_WER: {100 * all_b_wer:.2f}% ")
print()
