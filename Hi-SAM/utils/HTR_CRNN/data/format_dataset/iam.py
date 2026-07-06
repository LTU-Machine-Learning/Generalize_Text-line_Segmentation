#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#  This software is a computer program written in Python whose purpose is
#  to automatically recognize text from text-block images following a single-column layout.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

import os
import shutil
import xml.etree.ElementTree as ET
import tarfile, zipfile
import pickle
import numpy as np
from PIL import Image


# from: https://github.com/FactoDeepLearning/VAN/blob/master/Datasets/format_datasets.py


def format_IAM_line():
    """
    Format the IAM dataset at line level with the commonly used split (6,482 for train, 976 for validation and 2,915 for test)
    """
    source_folder = "raw/IAM"
    target_folder = "formatted/IAM_lines"
    tar_filename = "lines.tgz"
    line_folder_path = os.path.join(target_folder, "lines")

    tar_path = os.path.join(source_folder, tar_filename)
    if not os.path.isfile(tar_path):
        print("error - {} not found".format(tar_path))
        exit(-1)

    os.makedirs(target_folder, exist_ok=True)
    tar = tarfile.open(tar_path)
    tar.extractall(line_folder_path)
    tar.close()

    set_names = ["train", "valid", "test"]
    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict()
    }
    charset = set()

    for set_name in set_names:
        id = 0
        current_folder = os.path.join(target_folder, set_name)
        os.makedirs(current_folder, exist_ok=True)
        xml_path = os.path.join(source_folder, "{}.xml".format(set_name))
        xml_root = ET.parse(xml_path).getroot()
        for page in xml_root:
            name = page.attrib.get("FileName").split("/")[-1].split(".")[0]
            img_fold_path = os.path.join(line_folder_path, name.split("-")[0], name)
            img_paths = [os.path.join(img_fold_path, p) for p in sorted(os.listdir(img_fold_path))]
            for i, line in enumerate(page[2]):
                label = line.attrib.get("Value")
                img_name = "{}_{}.png".format(set_name, id)
                gt[set_name][img_name] = {
                    "text": label,
                }
                charset = charset.union(set(label))
                new_path = os.path.join(current_folder, img_name)
                os.replace(img_paths[i], new_path)
                id += 1

    shutil.rmtree(line_folder_path)
    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
        pickle.dump({
            "ground_truth": gt,
            "charset": sorted(list(charset)),
        }, f)


def format_IAM_paragraph(source_folder, info_paragraph, target_folder):
    """
    Format the IAM dataset at paragraph level with the commonly used split (747 for train, 116 for validation and 336 for test)
    """
    # source_folder = "raw/IAM"
    # target_folder = "formatted/IAM_paragraph"
    img_folder_path = os.path.join(target_folder, "images")

    os.makedirs(target_folder, exist_ok=True)

    tar_filenames = ["formsA-D.tgz", "formsE-H.tgz", "formsI-Z.tgz"]
    tar_paths = [os.path.join(source_folder, name) for name in tar_filenames]
    for tar_path in tar_paths:
        if not os.path.isfile(tar_path):
            print("error - {} not found".format(tar_path))
            exit(-1)
        tar = tarfile.open(tar_path)
        tar.extractall(img_folder_path)
        tar.close()

    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict()
    }
    charset = set()

    for set_name in ["train", "valid", "test"]:
        new_folder = os.path.join(target_folder, set_name)
        os.makedirs(new_folder, exist_ok=True)

        img_folder = os.path.join(new_folder, "images")
        os.makedirs(img_folder, exist_ok=True)
        yolo_label_folder = os.path.join(new_folder, "labels_bb")
        os.makedirs(yolo_label_folder, exist_ok=True)
        txt_page_label_folder = os.path.join(new_folder, "labels_txt_page")
        os.makedirs(txt_page_label_folder, exist_ok=True)
        page_label_folder = os.path.join(new_folder, "page_label")
        os.makedirs(page_label_folder, exist_ok=True)

        # files in https://github.com/FactoDeepLearning/VAN/tree/master/Datasets/raw/IAM
        xml_path = os.path.join(info_paragraph, "{}.xml".format(set_name))
        xml_root = ET.parse(xml_path).getroot()
        for page in xml_root:
            name = page.attrib.get("FileName").split("/")[-1].split(".")[0]
            img_path = os.path.join(img_folder_path, name + ".png")
            # new_name = "{}_{}.png".format(set_name, len(os.listdir(img_folder_path)))
            new_img_path = os.path.join(img_folder, name + ".png")

            root_result = ET.Element('root')

            lines = []
            full_text = ""
            for section in page:
                if section.tag != "Paragraph":
                    continue
                p_left, p_right = int(section.attrib.get("Left")), int(section.attrib.get("Right"))
                p_bottom, p_top = int(section.attrib.get("Bottom")), int(section.attrib.get("Top"))
                for i, line in enumerate(section):
                    words = []
                    for word in line:
                        words.append({
                            "text": word.attrib.get("Value"),
                            "left": int(word.attrib.get("Left")) - p_left,
                            "bottom": int(word.attrib.get("Bottom")) - p_top,
                            "right": int(word.attrib.get("Right")) - p_left,
                            "top": int(word.attrib.get("Top")) - p_top
                        })
                    lines.append({
                        "text": line.attrib.get("Value"),
                        "left": int(line.attrib.get("Left")) - p_left,
                        "bottom": int(line.attrib.get("Bottom")) - p_top,
                        "right": int(line.attrib.get("Right")) - p_left,
                        "top": int(line.attrib.get("Top")) - p_top,
                        "words": words
                    })
                    full_text = "{}{}\n".format(full_text, lines[-1]["text"])

                paragraph = {
                    "text": full_text[:-1],
                    "lines": lines
                }
                # gt[set_name][new_name] = paragraph
                gt[set_name][name] = paragraph
                charset = charset.union(set(full_text))

                with Image.open(img_path) as pil_img:
                    img = np.array(pil_img)
                img = img[p_top:p_bottom + 1, p_left:p_right + 1]
                Image.fromarray(img).save(new_img_path)

                height = img.shape[0]
                width = img.shape[1]

                # new_name = "{}_{}.txt".format(set_name, len(os.listdir(yolo_label_folder)))
                new_yolo_gt_path = os.path.join(yolo_label_folder, name + ".txt")
                with open(new_yolo_gt_path, 'w', encoding="utf-8") as file:
                    for one_line in lines:
                        left = one_line["left"]
                        right = one_line["right"]
                        bottom = one_line["bottom"]
                        top = one_line["top"]

                        x_c = ((left + right) / 2) / width
                        y_c = ((bottom + top) / 2) / height
                        width_line = (right - left) / width
                        height_line = (bottom - top) / height

                        # Yolo format: class x_center y_center width height
                        file.write(str(0))  # class line
                        file.write(" ")
                        file.write(str(x_c))
                        file.write(" ")
                        file.write(str(y_c))
                        file.write(" ")
                        file.write(str(width_line))
                        file.write(" ")
                        file.write(str(height_line))
                        file.write("\n")

                        # xml page format
                        points = str(int(one_line["left"])) + "," + str(int(one_line["top"])) + " "
                        points += str(int(one_line["right"])) + "," + str(int(one_line["top"])) + " "
                        points += str(int(one_line["right"])) + "," + str(int(one_line["bottom"])) + " "
                        points += str(int(one_line["left"])) + "," + str(int(one_line["bottom"]))

                        tl = ET.SubElement(root_result, 'TextLine')
                        coords = ET.SubElement(tl, 'Coords', points=points)
                        text_e = ET.SubElement(tl, 'TextEquiv')
                        t_unicode = ET.SubElement(text_e, 'Unicode')
                        t_unicode.text = one_line["text"]

                path_result = os.path.join(page_label_folder, name + ".xml")
                tree = ET.ElementTree(root_result)
                ET.indent(tree, space="\t", level=0)
                tree.write(path_result, encoding="utf-8")

    shutil.rmtree(img_folder_path)
    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
        pickle.dump({
            "ground_truth": gt,
            "charset": sorted(list(charset)),
        }, f)


if __name__ == "__main__":
    # format_IAM_line()

    source_folder = "C:/Users/simcor/dev/data/IAM/origin"
    info_paragraph = "C:/Users/simcor/dev/data/IAM/van_info_paragraph"
    target_folder = "C:/Users/simcor/dev/data/IAM/paragraph_new_3"
    format_IAM_paragraph(source_folder, info_paragraph, target_folder)
