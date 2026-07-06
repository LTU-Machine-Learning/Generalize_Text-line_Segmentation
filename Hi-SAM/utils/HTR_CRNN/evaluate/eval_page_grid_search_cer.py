import argparse
import faulthandler
import os
import time
from math import floor

import torch
from ultralytics import YOLO

from utils.HTR_CRNN.data.readers.read_config import read_json_config_line_htr, read_json_confi_page
from utils.HTR_CRNN.data.text.charset_token import CharsetToken
from utils.HTR_CRNN.evaluate.eval_page_two_steps import eval_page_level_batch_line
from utils.HTR_CRNN.models.crnn.crnn import CRNN
from utils.HTR_CRNN.models.utils.load_model import load_pretrained_model

parser = argparse.ArgumentParser()

parser.add_argument("config_file")
parser.add_argument("log_dir")

# Line detector
parser.add_argument("--path_model_yolo", default="", type=str)
parser.add_argument('--agnostic_nms', default=1, type=float)
parser.add_argument('--imgsz', default=1024, type=int)

parser.add_argument('--save_prediction', default=0, type=int)

# Line recognition
parser.add_argument("--path_model_crnn", default="", type=str)
parser.add_argument('--height_max_line', default=128, type=int)
parser.add_argument('--width_max_line', default=1024, type=int)
parser.add_argument('--pad_left_line', default=64, type=int)
parser.add_argument('--pad_right_line', default=64, type=int)

# CRNN Model
parser.add_argument('--add_squeeze_excitation', default=0, type=int)

# Dataset specificity
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
_, val_info, test_info, charsets_path, _ = read_json_confi_page(args.config_file)

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
                  add_squeeze_excitation=args.add_squeeze_excitation)

# Data
fixed_size_img_line = (args.height_max_line, args.width_max_line)

width_with_pad = args.width_max_line + args.pad_left_line + args.pad_right_line
x_reduced_len = floor(width_with_pad / width_divisor)

if os.path.isfile(args.path_model_crnn):
    load_pretrained_model(args.path_model_crnn, model_reco, device)
else:
    print("CRNN model not loaded")

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

print("Validation: ")

best_cer_val = 12
best_iou_t = 0.5
best_conf_t = 0.25

# Separate validation sets
for one_db in val_info:
    print("--------------Evaluate-------------------------------")
    print(one_db[0])

    dir_img = one_db[1]
    dir_label = one_db[2]
    ext_img = one_db[3]

    for iou_t in [x / 100.0 for x in range(30, 75, 5)]:
        for conf_t in [x / 100.0 for x in range(10, 70, 5)]:
            print(f"iou_t: {iou_t:.2f}  conf_t: {conf_t:.2f}")

            cer = eval_page_level_batch_line(ext_img, dir_img, dir_label, model_reco, model_yolo, iou_t, conf_t, agnostic_nms,
                                  args.imgsz,
                                  fixed_size_img_line, args.pad_left_line, args.pad_right_line, char_list,
                                  char_dict,
                                  x_reduced_len, device, args.log_dir, args.save_prediction)

            if cer < best_cer_val:
                best_iou_t = iou_t
                best_conf_t = conf_t

print("Configuration best val")
print(f"iou_t: {best_iou_t:.2f}  conf_t: {best_conf_t:.2f}")
print()
print("Test: ")

# Separate validation sets
for one_db in test_info:
    print("--------------Evaluate-------------------------------")
    print(one_db[0])

    dir_img = one_db[1]
    dir_label = one_db[2]
    ext_img = one_db[3]

    eval_page_level_batch_line(ext_img, dir_img, dir_label, model_reco, model_yolo, best_iou_t, best_conf_t, agnostic_nms, args.imgsz,
                    fixed_size_img_line, args.pad_left_line, args.pad_right_line, char_list, char_dict,
                    x_reduced_len, device, args.log_dir, args.save_prediction)
