import glob
import json


def create_charset(all_labels_json, path_save):
    """
    """
    # files = glob.glob(dir_data + '/**/*.txt', recursive=True)

    full_text = ""

    with open(all_labels_json, "r") as fp:
        all_labels = json.load(fp)

        for key_id, text in all_labels.items():
            full_text += text

    charset = set(full_text)
    charset = sorted(charset)

    with open(path_save, 'w', encoding="utf-8") as file:
        for one_char in charset:
            file.write(one_char)
            file.write("\n")

#
# def merge_charsets(list_path_charset, path_save_merge, convert_uppercase):
#     charset_dictionary = {}
#     char_number = 0
#
#     for path_c in list_path_charset:
#         with open(path_c, mode='r', encoding="utf-8") as f:
#             for line in f.readlines():
#                 if len(line) > 0:
#                     c = line[:-1]
#
#                     if convert_uppercase:
#                         c = c.upper()
#
#                     if c not in charset_dictionary:
#                         charset_dictionary[c] = char_number
#                         char_number += 1
#
#     myKeys = list(charset_dictionary.keys())
#     myKeys.sort()
#     sorted_dict = {i: charset_dictionary[i] for i in myKeys}
#     print("charset_dict:")
#     print(sorted_dict)
#
#     with open(path_save_merge, 'w', encoding="utf-8") as file:
#         for one_class in sorted_dict:
#             file.write(one_class)
#             file.write("\n")


if __name__ == '__main__':
    label_line_read = "C:/Users/simcor/dev/data/READ/2016/convert_line_polygon_train/all_label.json"
    charset_read = "C:/Users/simcor/dev/data/READ/2016/charset_read_2016.txt"

    create_charset(label_line_read, charset_read)

    # c_read = "C:/Users/simcor/dev/data/READ/2018/charset_read_2018.txt"
    # c_norhand = "C:/Users/simcor/dev/data/NordHand/v1/split/charset.txt"
    # c_simancas = "C:/Users/simcor/dev/data/HTR-Simancas/charset.txt"
    #
    # save_merge = "C:/Users/simcor/dev/data/charset_upper_read_norhand_simancas.txt"
    #
    # merge_charsets([c_read, c_norhand, c_simancas], save_merge, convert_uppercase="True")

