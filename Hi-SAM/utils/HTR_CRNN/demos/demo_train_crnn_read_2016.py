import argparse
import faulthandler
import os
import time

import torch
import wandb
from torch.utils.data import DataLoader

from utils.HTR_CRNN.clustering.compute_clusters_center import compute_cluster_center_crnn_k_1
from utils.HTR_CRNN.data.collate_batch.collate_line_reco_batch import CollateImageLabelHTR
from utils.HTR_CRNN.data.datasets.htr_line_dataset import HTRLineDataset
from utils.HTR_CRNN.data.globalvalue.text_comon_values import CTC_PAD, BLANK_STR_TOKEN
from utils.HTR_CRNN.data.image.augmentation_img import get_data_augmentation
from utils.HTR_CRNN.data.readers.read_config import read_json_config_line_htr, read_labels_files
from utils.HTR_CRNN.data.text.charset_token import CharsetToken
from utils.HTR_CRNN.evaluate.evaluate_crnn_one_epoch import evaluate_one_epoch_crnn
from utils.HTR_CRNN.models.crnn.crnn import CRNN
from utils.HTR_CRNN.models.utils.load_model import load_pretrained_model
from utils.HTR_CRNN.train.train_crnn_one_epoch import train_crnn_reg_one_epoch, train_crnn_one_epoch

parser = argparse.ArgumentParser()

parser.add_argument("config_file")
parser.add_argument("log_dir")

# Training hyper parameters
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--debug_pc', default=0, type=int)
parser.add_argument('--nb_epochs_max', default=3, type=int)
parser.add_argument("--path_model", default="", help="path of pretrained model", type=str)
parser.add_argument("--path_optimizer", default="", help="", type=str)
parser.add_argument('--height_max', default=128, type=int)
parser.add_argument('--width_max', default=1700, type=int)
parser.add_argument('--pad_left', default=64, type=int)
parser.add_argument('--pad_right', default=64, type=int)

parser.add_argument('--debug_print', default=0, type=int)

parser.add_argument('--milestones_lr_1', default=600, type=int)
parser.add_argument('--lr_decay_1', default=10, type=float)
parser.add_argument('--milestones_lr_2', default=800, type=int)
parser.add_argument('--lr_decay_2', default=20, type=float)


# Dataset specificity
parser.add_argument("--name_db_best_criteria", default="", type=str)
parser.add_argument('--use_grad_clip', default=0, type=int)

# Data preprocess
parser.add_argument('--load_img_as_grayscale', default=1, type=int)
parser.add_argument('--proba_augmentation', default=0.5, type=float)

parser.add_argument('--ratio_upscale', default=1.0, type=float)

# CRNN Model
# parser.add_argument('--weight_loss_ok', default=1.0, type=float)
# parser.add_argument('--weight_loss_ko', default=1.0, type=float)
parser.add_argument('--add_squeeze_excitation', default=1, type=int)


# Regularization == Center loss
parser.add_argument('--use_regularization', default=0, type=int)
parser.add_argument('--epoch_start_regularization', default=650, type=int)
parser.add_argument('--weight_loss_regularization_ok', default=0.55, type=float)

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
directory_log = args.log_dir
train_info, val_info, test_info, charsets_path, all_labels_files, dir_wandb = read_json_config_line_htr(args.config_file)
all_labels = read_labels_files(all_labels_files)

# Alphabet
charset = CharsetToken(charsets_path, use_blank=True)
char_list = charset.get_charset_list()
char_dict = charset.get_charset_dictionary()

print("Alphabet:")
print(char_list)

# Model
cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]
head_cfg = (256, 3)  # (hidden , num_layers)
width_divisor = 8

model_reco = CRNN(cnn_cfg,
                  head_cfg,
                  charset.get_nb_char(),
                  load_img_as_grayscale=args.load_img_as_grayscale,
                  add_squeeze_excitation=args.add_squeeze_excitation)

# Data
fixed_size_img = (args.height_max, args.width_max)
aug_transforms = get_data_augmentation(proba=args.proba_augmentation)

print("Training data:")
# Merge training sets
train_db = HTRLineDataset(train_info,
                          fixed_size_img,
                          width_divisor,
                          args.pad_left,
                          args.pad_right,
                          char_dict,
                          aug_transforms,
                          load_img_as_grayscale=args.load_img_as_grayscale,
                          apply_noise=1,
                          is_trainset=True,
                          ratio_upscale=args.ratio_upscale,
                          all_labels=all_labels)

print("Nb samples train total: " + str(len(train_db)))
print()
# Pad img with black = 0
c_collate_fn = CollateImageLabelHTR(imgs_pad_value=[0], pad_txt=CTC_PAD)
collate_fn = c_collate_fn.collate_fn

train_dataloader = DataLoader(train_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                              collate_fn=collate_fn, shuffle=True)

all_val_dataloader = []
all_val_dataset = []

print("Validation: ")
# Separate validation sets
for one_db in val_info:
    val_db = HTRLineDataset([one_db],
                            fixed_size_img,
                            width_divisor,
                            args.pad_left,
                            args.pad_right,
                            char_dict,
                            load_img_as_grayscale=args.load_img_as_grayscale,
                            ratio_upscale=args.ratio_upscale,
                            all_labels=all_labels)

    val_dataloader = DataLoader(val_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                                collate_fn=collate_fn, shuffle=False)

    all_val_dataloader.append(val_dataloader)
    all_val_dataset.append(val_db)  # To get name DB

all_test_dataloader = []
all_test_dataset = []
print("Test: ")
# Separate test sets
for one_db in test_info:
    test_db = HTRLineDataset([one_db],
                             fixed_size_img,
                             width_divisor,
                             args.pad_left,
                             args.pad_right,
                             char_dict,
                             load_img_as_grayscale=args.load_img_as_grayscale,
                             ratio_upscale=args.ratio_upscale,
                             all_labels=all_labels)

    test_dataloader = DataLoader(test_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                                 collate_fn=collate_fn, shuffle=False)

    all_test_dataloader.append(test_dataloader)
    all_test_dataset.append(test_db)

# Init model
print("Initializing model weights kaiming")
for p in model_reco.parameters():
    if p.dim() > 1:
        nonlinearity = "relu"
        torch.nn.init.kaiming_normal_(p, nonlinearity=nonlinearity)

if os.path.isfile(args.path_model):
    load_pretrained_model(args.path_model, model_reco, device)

print(f"Transferring model to {str(device)}...")
model_reco = model_reco.to(device)

number_parameters = sum(p.numel() for p in model_reco.parameters() if p.requires_grad)
print(f"Model has {number_parameters:,} trainable parameters.")

print_summary = ""

# Setup Wandb for log
if not args.debug_pc:
    wandb.init(project="CRNN_reg", entity="htr-analysis", dir=dir_wandb)
    wandb.config = {
        "learning_rate": args.learning_rate,
        "epochs": args.nb_epochs_max,
        "batch_size": args.batch_size
    }
    print("run name wand : " + str(wandb.run.name))

    print_summary += "run name wand : "
    print_summary += str(wandb.run.name)
    print_summary += "\n"

ctc_loss_fn = torch.nn.CTCLoss(zero_infinity=True, reduction="mean")

optimizer = torch.optim.Adam(model_reco.parameters(), lr=args.learning_rate)

if os.path.isfile(args.path_optimizer):
    try:
        checkpoint = torch.load(args.path_optimizer, map_location=device)
        optimizer.load_state_dict(checkpoint)
        print("Load optimizer")
    except:
        print("Error load optimizer")
        optimizer = torch.optim.Adam(model_reco.parameters(), lr=args.learning_rate)

best_cer = 1.0
# best_val_loss = 12000.0
best_epoch = 0

path_save_model_best = os.path.join(directory_log, "crnn_best.torch")
# path_save_model_best_val_loss = os.path.join(directory_log, "crnn_best_val_loss.torch")
path_save_model_last = os.path.join(directory_log, "crnn_last.torch")

path_save_optimizer_best = os.path.join(directory_log, "optimizer_best.torch")
path_save_optimizer_last = os.path.join(directory_log, "optimizer_last.torch")

lr = args.learning_rate
# Center loss
index_class_to_filter = [char_dict["<BLANK>"], char_dict[" "]]

loss_reg = torch.nn.MSELoss(reduction="mean")

conf_reg = {
    "index_class_to_filter": index_class_to_filter,
    "loss_reg": loss_reg,
    "weight_loss_regularization_ok": args.weight_loss_regularization_ok,
}

prototypes_value = []
compute_loss_reg = False


begin_train = time.time()
# Training
for epoch in range(0, args.nb_epochs_max):
    begin_time_epoch = time.time()
    print('EPOCH {}:'.format(epoch))

    # Learning rate values  -> refactor with step scheduler
    if epoch < args.milestones_lr_1:
        lr = args.learning_rate
    # First decay
    elif epoch < args.milestones_lr_2:
        lr = args.learning_rate / args.lr_decay_1
    # Second decay
    else:
        lr = args.learning_rate / args.lr_decay_2

    for g in optimizer.param_groups:
        g['lr'] = lr
    print("lr:" + str(lr))

    # Training
    if compute_loss_reg:
        dict_losses = train_crnn_reg_one_epoch(train_dataloader,
                                               optimizer,
                                               model_reco,
                                               device,
                                               ctc_loss_fn,
                                               conf_reg,
                                               prototypes_value,
                                               char_list,
                                               char_dict[BLANK_STR_TOKEN],
                                               use_grad_clip=args.use_grad_clip)
    else:
        dict_losses = train_crnn_one_epoch(train_dataloader,
                                           optimizer,
                                           model_reco,
                                           device,
                                           ctc_loss_fn,
                                           char_list,
                                           char_dict[BLANK_STR_TOKEN],
                                           use_grad_clip=args.use_grad_clip)

    # print('train_loss_main {}'.format(dict_losses["loss_main"]))
    print('train_loss_main_ok_main {}'.format(dict_losses["loss_main_ok_epoch"]))
    print('train_loss_main_ko_main {}'.format(dict_losses["loss_main_ko_epoch"]))

    print('train_loss_shortcut {}'.format(dict_losses["loss_shortcut"]))

    if compute_loss_reg:
        print('train_loss_reg_epoch {}'.format(dict_losses["loss_reg_epoch"]))

    if not args.debug_pc:
        # wandb.log({"train_loss_main": dict_losses["loss_main"]}, step=epoch)
        wandb.log({"train_loss_main_ok_main": dict_losses["loss_main_ok_epoch"]}, step=epoch)
        wandb.log({"train_loss_main_ko_main": dict_losses["loss_main_ko_epoch"]}, step=epoch)
        wandb.log({"train_loss_shortcut": dict_losses["loss_shortcut"]}, step=epoch)

        if compute_loss_reg:
            wandb.log({"train_loss_reg_epoch": dict_losses["loss_reg_epoch"]}, step=epoch)

    for i_db in range(len(all_val_dataloader)):
        print("--------------Evaluate-------------------------------")
        current_db_name = all_val_dataset[i_db].name_db
        print(current_db_name)
        dict_result = evaluate_one_epoch_crnn(all_val_dataloader[i_db],
                                              model_reco,
                                              device,
                                              char_list,
                                              char_dict[BLANK_STR_TOKEN],
                                              ctc_loss_fn)

        dict_result["metrics_main"].print_cer_wer()
        dict_result["metrics_shortcut"].print_cer_wer()

        # Save model
        if args.name_db_best_criteria == current_db_name:
            if args.eval_icdar_2025 == 1:
                if dict_result["metrics_main_icdar_2025"].get_cer() < best_cer_icdar_filter:
                    best_cer_icdar_filter = dict_result["metrics_main_icdar_2025"].get_cer()

            if dict_result["metrics_main"].get_cer() < best_cer:
                best_cer = dict_result["metrics_main"].get_cer()
                best_epoch = epoch
                print("Best cer final, save model.")

                torch.save(model_reco.state_dict(), path_save_model_best)
                torch.save(optimizer.state_dict(), path_save_optimizer_best)

        # Log WandB
        if not args.debug_pc:
            l_val_cer_main = "val_cer_main_" + current_db_name
            wandb.log({l_val_cer_main: dict_result["metrics_main"].get_cer()}, step=epoch)

            l_val_loss_main = "val_loss_main_" + current_db_name
            wandb.log({l_val_loss_main: dict_result["metrics_main"].get_loss()}, step=epoch)

    # Compute cluster if activate
    if args.use_regularization == 1:
        if epoch >= args.epoch_start_regularization:
            print("Compute prototype:")
            compute_loss_reg = True

            prototypes_value = compute_cluster_center_crnn_k_1(train_dataloader,
                                                               model_reco,
                                                               device,
                                                               char_list,
                                                               char_dict["<BLANK>"],
                                                               index_class_to_filter)

    end_time_epoch = time.time()
    print("Time one epoch (s): " + str((end_time_epoch - begin_time_epoch)))
    print("")

    torch.save(model_reco.state_dict(), path_save_model_last)
    torch.save(optimizer.state_dict(), path_save_optimizer_last)

end_train = time.time()
print("best_epoch: " + str(best_epoch))
print("best_cer val: " + str(best_cer))
print("Time all (s): " + str((end_train - begin_train)))
print("End training")

print_summary += "best_epoch: " + str(best_epoch) + "\n"
print_summary += "best_cer val: "
print_summary += f"{100 * best_cer:.2f}% \n"
print_summary += "\n"

for i_db in range(len(all_test_dataloader)):
    print("--------------Testing-------------------------------")
    current_db_name = all_test_dataset[i_db].name_db
    print(current_db_name)
    print_summary += current_db_name
    print_summary += " \n"

    print()
    print("--------Begin Testing last-----------")
    dict_result = evaluate_one_epoch_crnn(all_test_dataloader[i_db],
                                          model_reco,
                                          device,
                                          char_list,
                                          char_dict[BLANK_STR_TOKEN],
                                          ctc_loss_fn)

    dict_result["metrics_main"].print_cer_wer()

    print_summary += "Testing last \n"
    str_cer_wer = dict_result["metrics_main"].str_cer_wer()
    print_summary += str_cer_wer + "\n"

    dict_result["metrics_shortcut"].print_cer_wer()

    if not args.debug_pc:
        label_wb = "test_cer_last_" + current_db_name
        wandb.log({label_wb: dict_result["metrics_main"].get_cer()}, step=epoch)

    if len(val_info) > 0:
        print("--------Begin Testing best cer val-----------")
        # Load best model
        if os.path.isfile(path_save_model_best):
            load_pretrained_model(path_save_model_best, model_reco, device, print_load_ok=False)

        dict_result = evaluate_one_epoch_crnn(all_test_dataloader[i_db],
                                              model_reco,
                                              device,
                                              char_list,
                                              char_dict[BLANK_STR_TOKEN],
                                              ctc_loss_fn)

        dict_result["metrics_main"].print_cer_wer()
        dict_result["metrics_shortcut"].print_cer_wer()

        print_summary += "\n"
        print_summary += "Testing best val cer \n"
        str_cer_wer = dict_result["metrics_main"].str_cer_wer()
        print_summary += str_cer_wer + "\n"

        if not args.debug_pc:
            label_wb = "test_cer_bestval_cer_" + current_db_name
            wandb.log({label_wb: dict_result["metrics_main"].get_cer()}, step=epoch)

print()
print("Summary:")
print(print_summary)
