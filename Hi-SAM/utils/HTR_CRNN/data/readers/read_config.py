import json


def read_json_config_line_htr(path_file):
    train_info = []
    val_info = []
    test_info = []
    charsets_path = []
    all_labels_files = []
    dir_wandb = ""

    with open(path_file, "r") as fp:
        config_values = json.load(fp)

        print(config_values)

        if "train_data" in config_values:
            for key, value in config_values["train_data"].items():
                train_info.append([key, value["img_dir"], value["extension_img"]])
        if "val_data" in config_values:
            for key, value in config_values["val_data"].items():
                val_info.append([key, value["img_dir"], value["extension_img"]])
        if "test_data" in config_values:
            for key, value in config_values["test_data"].items():
                test_info.append([key, value["img_dir"], value["extension_img"]])

        if "charset_files" in config_values:
            for key, value in config_values["charset_files"].items():
                charsets_path.append(value)

        if "all_labels_files" in config_values:
            for key, value in config_values["all_labels_files"].items():
                all_labels_files.append(value)

        dir_wandb = config_values["dir_wandb"]

    return train_info, val_info, test_info, charsets_path, all_labels_files, dir_wandb


# To reformat -> define one common config format
def read_json_confi_page(path_file):
    train_info = []
    val_info = []
    test_info = []
    charsets_path = []
    dir_wandb = ""

    with open(path_file, "r") as fp:
        config_values = json.load(fp)

        print(config_values)

        if "train_data" in config_values:
            for key, value in config_values["train_data"].items():
                train_info.append([key, value["img_dir"], value["label_dir"], value["extension_img"]])
        if "val_data" in config_values:
            for key, value in config_values["val_data"].items():
                val_info.append([key, value["img_dir"], value["label_dir"], value["extension_img"]])
        if "test_data" in config_values:
            for key, value in config_values["test_data"].items():
                test_info.append([key, value["img_dir"], value["label_dir"], value["extension_img"]])

        if "charset_files" in config_values:
            for key, value in config_values["charset_files"].items():
                charsets_path.append(value)

        # if "charset_files" in config_values:
        dir_wandb = config_values["dir_wandb"]

    return train_info, val_info, test_info, charsets_path, dir_wandb


def read_labels_files(list_files):
    dict_all = {}

    for one_f in list_files:
        with open(one_f, "r") as fp:
            all_labels = json.load(fp)

            dict_all = dict_all | all_labels

    return dict_all
