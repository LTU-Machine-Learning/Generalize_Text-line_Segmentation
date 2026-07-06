import os
from math import floor
import re

import editdistance
import torch
import numpy as np
import json
from skimage import io

from utils.HTR_CRNN.data.globalvalue.text_comon_values import BLANK_STR_TOKEN
from utils.HTR_CRNN.data.readers.read_xml_page import read_xml_page_gt
from utils.HTR_CRNN.data.sorter.sort_bb_label import get_indexes_sort_bb
from utils.HTR_CRNN.data.text.best_path import ctc_best_path_one
from utils.HTR_CRNN.data.text.charset_token import CharsetToken
from utils.HTR_CRNN.models.crnn.crnn import CRNN
from utils.HTR_CRNN.models.utils.load_model import load_pretrained_model
from utils.HTR_CRNN.data.image.preprocess_img import preprocess_img_line

'''
 HTRScores class to compute HTR scores (CER, WER) at page level or prevides sum of CER and WER with numer of letters/words for all pages to compute global scores later
    Usage: 
    initialize HTRScores instance with (
        device: the torch device,
        args_path_charset: HTR model trained charset (should be provided with the HTR model), 
        args_dir_img: Page images, 
        args_dir_gt_xml: the gound truth (GT) shoudld be in the following format
            <TextLine>
                <Coords points="1858,271 1632,267 1632,331 1858,335"/>
                <TextEquiv>
                    <Unicode>ground truth text</Unicode>
                </TextEquiv>
            , 
        args_path_model_crnn: path to trained CRNN model, 
        args_add_squeeze_excitation (should be provided with the HTR model), 
        args_height_max_line (should be provided with the HTR model), 
        args_width_max_line (should be provided with the HTR model), 
        args_pad_left_line (should be provided with the HTR model), 
        args_pad_right_line (should be provided with the HTR model), 
        args_ext_img: image extension
        )
    Then
    call HTRScores instance with (file_id, lines_bb_pred) where
        file_id : str, file identifier to locate image and gt
        lines_bb_pred : list of list, each sublist is [x1, y1, x2, y2] of predicted line bounding box

    Returns:
        cer: page level Character Error Rate (CER)
        wer: page level Word Error Rate (WER)
        cer_all_norm_nb_letter: sum of edit distance over all letters (for all pages)
        nb_total_letter: total number of letters (for all pages)
        wer_all_norm_nb_words: sum of edit distance over all words (for all pages)
        nb_total_words: total number of words (for all pages)
        
        
    for DAN WER variant, uncomment the commented "return" line and comment the return line after
        wer_2: page level Word Error Rate (WER) (DAN variant)
        wer_all_norm_nb_words_dan: sum of edit distance over all words (DAN variant) (for all pages)
        nb_total_words_dan: total number of words (for all pages)

'''

'''
    ### Explaning DAN WER variant ###
    The DAN WER variant processes punctuation as separate words by adding spaces around punctuation characters before splitting the text into words. 
    This means that punctuation marks are treated as individual tokens during the word error rate (WER) calculation, which can lead to a more granular 
    assessment of recognition errors, especially in texts with significant punctuation usage.

    All punctuation is split as its own word

    "word," becomes "word ,"

    "hello-world" becomes "hello - world"

    Then WER is computed on this expanded token list

    This follows the DAN (Document Attention Network) evaluation protocol.

    Use DAN WER = Only use if you explicitly compare with DAN paper
'''

class HTRScores:
    def __init__(   self,
                    device,
                    args_path_charset,
                    args_dir_img,
                    args_dir_gt_xml,
                    args_path_model_crnn,
                    args_bb_sorting,
                    args_model_config_path,
                    args_add_squeeze_excitation=0,
                    args_height_max_line=128,
                    args_width_max_line=1024,
                    args_pad_left_line=64,
                    args_pad_right_line=64,
                    args_ext_img=".jpg"
                ):
        self.device = device
        self.args_dir_img = args_dir_img
        self.args_dir_gt_xml = args_dir_gt_xml
        self.args_pad_left_line = args_pad_left_line
        self.args_pad_right_line = args_pad_right_line
        self.args_ext_img = args_ext_img
        self.args_bb_sorting = args_bb_sorting

        config_values = {}

        with open(args_model_config_path, "r") as fp:
            config_values = json.load(fp)

        # Alphabet
        charset = CharsetToken([args_path_charset], use_blank=True)
        self.char_list = charset.get_charset_list()
        self.char_dict = charset.get_charset_dictionary()

        # CRNN
        # cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]
        # head_cfg = (256, 3)  # (hidden , num_layers)
        # width_divisor = 8
        cnn_cfg = config_values["cnn_cfg"]
        head_cfg = config_values["head_cfg"]  # (hidden dimension, num_layers blstm)
        width_divisor = config_values["width_divisor"]

        model_reco = CRNN(cnn_cfg,
                        head_cfg,
                        charset.get_nb_char(),
                        add_squeeze_excitation=args_add_squeeze_excitation)

        # Data
        self.fixed_size_img_line = (args_height_max_line, args_width_max_line)
        width_with_pad = args_width_max_line + args_pad_left_line + args_pad_right_line
        self.x_reduced_len = floor(width_with_pad / width_divisor)

        if os.path.isfile(args_path_model_crnn):
            load_pretrained_model(args_path_model_crnn, model_reco, device)

        print(f"Transferring model to {str(device)}...")
        model_reco = model_reco.to(device)

        number_parameters_reco = sum(p.numel() for p in model_reco.parameters() if p.requires_grad)
        print(f"Recognition model has {number_parameters_reco:,} trainable parameters.")

        model_reco.eval()
        self.model_reco = model_reco
    
    def __call__(self, file_id, lines_bb_pred):

        all_gt_txt = ""
        all_pred_txt = ""
        cer_all_norm_nb_letter = 0
        nb_total_letter = 0
        wer_all_norm_nb_words = 0
        nb_total_words = 0
        wer_all_norm_nb_words_dan = 0
        nb_total_words_dan = 0

        # Check if gt exist
        path_gt = os.path.join(self.args_dir_gt_xml, file_id + ".xml")

        if not os.path.isfile(path_gt):
            print(path_gt)
            print("gt associated doesn't exist.")
        
        path_img = os.path.join(self.args_dir_img, file_id + "." + self.args_ext_img)

        if not os.path.isfile(path_img):
            print(path_img)
            print("img associated doesn't exist.")

        # CRNN train with grayscale
        img_page = io.imread(path_img, as_gray=True)
        max_v = np.max(img_page)

        # Color image are converted to grayscale -> value [0 ; 1]
        # Grayscale image are not converted -> value [0 ; 255]
        if max_v <= 1:
            img_page *= 255.0


        # Sort
        bb_xyxy, labels = read_xml_page_gt(path_gt)

        if self.args_bb_sorting == "IoU-matching":
            labels_sorted = labels
            gt_indices, pred_indices = get_indexes_sort_bb([bb_xyxy, lines_bb_pred], self.args_bb_sorting)
            i_s = pred_indices
        else:
            i_g = get_indexes_sort_bb(bb_xyxy, self.args_bb_sorting)
            labels_sorted = [labels[i] for i in i_g]
            i_s = get_indexes_sort_bb(lines_bb_pred, self.args_bb_sorting)

        prediction_sorted = [lines_bb_pred[i] for i in i_s]
        batch_img_line = []
        
        # print(img_page.shape)
        for one_l in prediction_sorted:
            x1 = int(one_l[0])
            y1 = int(one_l[1])
            x2 = int(one_l[2])
            y2 = int(one_l[3])

            if x2 - x1 <= 0 or y2 - y1 <= 0:
                print("Incoherent size -> continue")
                continue
            
            img_line = img_page[y1:y2, x1:x2]

            if img_line is None or img_line.size == 0:
                print("Empty crop -> continue")
                continue

            # Recognition line with CRNN
            img_line = preprocess_img_line(img_line, self.fixed_size_img_line, self.args_pad_left_line, self.args_pad_right_line)
            img_tensor = torch.as_tensor(img_line, dtype=torch.float32)
            img_tensor = img_tensor.unsqueeze(0)  # Add channel dim
            batch_img_line.append(img_tensor)

        if len(batch_img_line) != 0:
            # no valid predicted lines on this page  

            batch_img_line = torch.stack(batch_img_line)
            batch_img_line = batch_img_line.to(self.device)

            pred_page_txt = []

            # Make recognition line level in a batch
            y_pred, _, _ = self.model_reco(batch_img_line)
            output, aux_output = y_pred

            # Main head
            output_log = torch.nn.functional.log_softmax(output, dim=-1)

            # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
            output_log = output_log.transpose(0, 1)

            top = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:self.x_reduced_len] for j, lp in enumerate(output_log)]
            predictions_text = [ctc_best_path_one(p, self.char_list, self.char_dict[BLANK_STR_TOKEN]) for p in top]
            predictions_text = [t.strip() for t in  predictions_text]  # Remove text padding

            # Batch of one element
            for one_line_pred in predictions_text:
                pred_page_txt.append(one_line_pred)


            # Evaluate
            pred_page_txt = " ".join(pred_page_txt) # Concatenate
            pred_page_txt = pred_page_txt.replace("  ", " ") # Remove double space
        else:
            pred_page_txt = "" # No valid predicted line, empty page level prediction
        all_pred_txt += pred_page_txt + " "


        gt_page_txt = " ".join(labels_sorted) # Concatenate
        gt_page_txt = gt_page_txt.replace("  ", " ") # Remove double space
        all_gt_txt += gt_page_txt + " "

        # Evaluation
        nb_char_gt = len(gt_page_txt)
        nb_total_letter += nb_char_gt

        # print("gt_page_txt :", gt_page_txt)
        # print("pred_page_txt :", pred_page_txt)
        cer = editdistance.eval(gt_page_txt, pred_page_txt)
        cer_all_norm_nb_letter += cer
        if nb_char_gt > 0:
            cer /= nb_char_gt
        # print("cer :", cer)

        gt_page_split = gt_page_txt.split(" ")
        pred_page_split = pred_page_txt.split(" ")
        nb_word_gt = len(gt_page_split)
        nb_total_words += nb_word_gt

        wer = editdistance.eval(gt_page_split, pred_page_split)
        wer_all_norm_nb_words += wer
        wer /= nb_word_gt


        # Other variant of WER from DAN git
        gt_page_txt_dan = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ',
                            gt_page_txt)  # punctuation processed as word
        pred_page_txt_dan = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ',
                            pred_page_txt)  # punctuation processed as word

        gt_page_split = gt_page_txt_dan.split(" ")
        pred_page_split = pred_page_txt_dan.split(" ")

        nb_word_gt_2 = len(gt_page_split)
        wer_2 = editdistance.eval(gt_page_split, pred_page_split)

        wer_all_norm_nb_words_dan += wer_2
        nb_total_words_dan += nb_word_gt_2

        wer_2 /= nb_word_gt_2

        # print("Gt:")
        # print(gt_page_txt)
        # print("Pred:")
        # print(pred_page_txt)
        # print(f"CER: {100 * cer:.2f}% ")
        # print(f"WER: {100 * wer:.2f}% ")
        # print(f"WER DAN: {100 * wer_2:.2f}% ")
        # print()
        
        # return :
        # cer: page leve CER
        # wer: page level WER
        # cer_all_norm_nb_letter: sum of edit distance over all letters (for all pages)
        # nb_total_letter: total number of letters (for all pages)
        # wer_all_norm_nb_words: sum of edit distance over all words (for all pages)
        # nb_total_words: total number of words (for all pages)

        # for DAN WER variant, uncomment the below line and comment the return line after
        # return cer, wer_2, cer_all_norm_nb_letter, nb_total_letter, wer_all_norm_nb_words_dan, nb_total_words_dan

        return cer, wer, cer_all_norm_nb_letter, nb_total_letter, wer_all_norm_nb_words, nb_total_words



