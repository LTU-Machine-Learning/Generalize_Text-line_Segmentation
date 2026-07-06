import json


def print_stats_text_datasets_lines(json_label):
    full_text = ""

    nb_total_letter = 0
    nb_total_words = 0

    min_len_txt = 999999
    max_len_txt = 0
    mean_len_txt = 0

    min_nb_words_txt = 999999
    max_nb_words_txt = 0
    mean_nb_words_txt = 0

    nb_items = 0

    dict_class_nb = {}
    dict_line_length = {}

    # for one_file_label in files:
    with open(json_label, "r") as fp:
        all_labels = json.load(fp)

        for key_id, label in all_labels.items():
            nb_total_letter += len(label)
            label_word = label.split(" ")

            nb_total_words += len(label_word)

            full_text += label

            for one_char in label:
                if one_char in dict_class_nb:
                    dict_class_nb[one_char] += 1
                else:
                    dict_class_nb[one_char] = 1

            len_txt = len(label)
            if len_txt < min_len_txt:
                min_len_txt = len_txt
            if len_txt > max_len_txt:
                max_len_txt = len_txt

            mean_len_txt += len_txt

            if len_txt in dict_line_length:
                dict_line_length[len_txt] += 1
            else:
                dict_line_length[len_txt] = 1

            nb_words = len(label_word)

            if nb_words < min_nb_words_txt:
                min_nb_words_txt = nb_words
            if nb_words > max_nb_words_txt:
                max_nb_words_txt = nb_words

            mean_nb_words_txt += nb_words

            nb_items += 1

    mean_len_txt /= nb_items
    mean_nb_words_txt /= nb_items

    print("min_len_txt :" + str(min_len_txt))
    print("max_len_txt :" + str(max_len_txt))
    print("mean_w :" + str(mean_len_txt))

    print("min_nb_words_txt :" + str(min_nb_words_txt))
    print("max_nb_words_txt :" + str(max_nb_words_txt))
    print("mean_nb_words_txt :" + str(mean_nb_words_txt))

    myKeys = list(dict_line_length.keys())
    myKeys.sort()
    sorted_dict = {i: dict_line_length[i] for i in myKeys}
    print("dict_line_length:")
    print(sorted_dict)

    print("dict_class_nb :")
    print(dict_class_nb)
    print("nb_total_letter :" + str(nb_total_letter))
    print("nb_total_words :" + str(nb_total_words))


if __name__ == "__main__":
    json_label = "C:/Users/simcor/dev/data/READ/2016/convert_line_polygon_train/all_label.json"
    print_stats_text_datasets_lines(json_label)
