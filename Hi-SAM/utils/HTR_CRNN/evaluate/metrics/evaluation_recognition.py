from enum import Enum

import editdistance

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


# class ProcessWER(Enum):
#     NO = 1
#     VAN = 2
#     KEEP_ONLY_LETTER = 3
#     DAS_V2 = 4
#
#     def __str__(self):
#         return self.name


def nb_chars_from_list(list_gt):
    return sum([len(t) for t in list_gt])


def nb_words_from_list(list_gt, use_tokenizer):
    len_ = 0
    for gt in list_gt:
        if use_tokenizer:
            gt = word_tokenize(gt)
        else:
            gt = gt.split(" ")
        len_ += len(gt)
    return len_


def edit_wer_from_list(truth, pred, use_tokenizer):
    edit = 0
    for pred, gt in zip(pred, truth):
        if use_tokenizer:
            gt = word_tokenize(gt)
            pred = word_tokenize(pred)
        else:
            gt = gt.split(" ")
            pred = pred.split(" ")
        edit += editdistance.eval(gt, pred)
    return edit
