import json
import os
import glob

import xml.etree.ElementTree as ET

import cv2


def convert_db_page_bb_min_max(dir_xml, dir_img, dir_save):
    """
    """

    # Prepare directories save
    os.makedirs(dir_save, exist_ok=True)

    path_dir_img_page = os.path.join(dir_save, "image_page")
    path_dir_img_line = os.path.join(dir_save, "image_line")
    # path_dir_label_line = os.path.join(dir_save, "label_line")
    path_dir_label_page = os.path.join(dir_save, "label_page")
    path_dir_label_yolo = os.path.join(dir_save, "label_yolo")

    os.makedirs(path_dir_img_page, exist_ok=True)
    os.makedirs(path_dir_img_line, exist_ok=True)
    # os.makedirs(path_dir_label_line, exist_ok=True)
    os.makedirs(path_dir_label_page, exist_ok=True)
    os.makedirs(path_dir_label_yolo, exist_ok=True)

    files_xml = glob.glob(dir_xml + '/**/*.xml', recursive=True)

    label_line = {}

    for one_file in files_xml:
        # Get id file
        split_name = os.path.split(one_file)
        split_name = split_name[1].split(sep=".")  # Filename and extension
        id_file = split_name[0]

        img_path = os.path.join(dir_img, id_file + ".jpg")

        img = cv2.imread(img_path)    # dim: height, width, channel
        height_page = img.shape[0]
        width_page = img.shape[1]

        # Save img page
        path_img_page = os.path.join(path_dir_img_page, id_file + ".jpg")
        cv2.imwrite(path_img_page, img)

        xml_root = ET.parse(one_file).getroot()

        index_line = 0
        text_page = ""
        yolo_line_coordinate = []
        for node in xml_root.iter():

            if "TextLine" in node.tag:
                for node_c in node.iter():
                    if "Coords" in node_c.tag:
                        polygon_pts = node_c.attrib["points"]

                        polygon_pts = polygon_pts.split(sep=" ")

                        nb_points = len(polygon_pts)

                        x_min = 999999
                        x_max = 0
                        y_min = 999999
                        y_max = 0

                        # Draw line polygon
                        for i in range(nb_points):
                            coordinate_p_i = polygon_pts[i]  # str 380,282
                            coordinate_p_i = coordinate_p_i.split(sep=",")

                            # Min max x
                            if int(coordinate_p_i[0]) < x_min:
                                x_min = int(coordinate_p_i[0])
                            if int(coordinate_p_i[0]) > x_max:
                                x_max = int(coordinate_p_i[0])
                            # Min max y
                            if int(coordinate_p_i[1]) < y_min:
                                y_min = int(coordinate_p_i[1])
                            if int(coordinate_p_i[1]) > y_max:
                                y_max = int(coordinate_p_i[1])

                        center_x = (x_min + x_max) / 2.0
                        center_y = (y_min + y_max) / 2.0

                        center_x /= width_page
                        center_y /= height_page

                        width_line = (x_max - x_min) / width_page
                        height_line = (y_max - y_min) / height_page

                        yolo_line_coordinate.append([center_x, center_y, width_line, height_line])

                        # Extract line img
                        # dim img: heigh, width, channel
                        img_line = img[y_min:y_max, x_min:x_max, :]
                        path_img_line = os.path.join(path_dir_img_line, id_file + "_" + str(index_line) + ".jpg")
                        cv2.imwrite(path_img_line, img_line)

                    if "TextEquiv" in node_c.tag:
                        # print(node_c)
                        for node_txt in node_c.iter():
                            if "Unicode" in node_txt.tag:
                                txt_gt_line = node_txt.text
                                # # V1: OK
                                if txt_gt_line is not None:
                                    # file.write(txt_gt_line)
                                    label_line[id_file + "_" + str(index_line)] = txt_gt_line
                                    text_page += txt_gt_line
                                    text_page += "\n"
                                else:
                                    print("No gt text")
                                    # file.write(" ")
                                    label_line[id_file + "_" + str(index_line)] = " "
                                # # V2 -> cf. DAN
                                # id_line = node.attrib["id"]
                                # if txt_gt_line is None and id_line not in ["line_a5f4ab4e-2ea0-4c65-840c-4a89b04bd477",
                                #                                            "line_e1288df8-8a0d-40df-be91-4b4a332027ec",
                                #                                            "line_455330f3-9e27-4340-ae86-9d6c448dc091",
                                #                                            "line_ecbbccee-e8c2-495d-ac47-0aff93f3d9ac",
                                #                                            "line_e918616d-64f8-43d2-869c-f687726212be",
                                #                                            "line_ebd8f850-1da5-45b1-b59c-9349497ecc8e",
                                #                                            "line_816fb2ce-06b0-4e00-bb28-10c8b9c367f2"]:
                                #     # print("ignored null line{}".format(page_dict["img_path"]))
                                #     print("ignored null line in: " + id_file)
                                #     print("index_line: " + str(index_line))
                                #     continue
                                # if id_line == "line_816fb2ce-06b0-4e00-bb28-10c8b9c367f2":
                                #     txt_gt_line = "16"
                                # elif id_line == "line_a5f4ab4e-2ea0-4c65-840c-4a89b04bd477":
                                #     txt_gt_line = "108"
                                # elif id_line == "line_e1288df8-8a0d-40df-be91-4b4a332027ec":
                                #     txt_gt_line = "196"
                                # elif id_line == "line_455330f3-9e27-4340-ae86-9d6c448dc091":
                                #     txt_gt_line = "199"
                                # elif id_line == "line_ecbbccee-e8c2-495d-ac47-0aff93f3d9ac":
                                #     txt_gt_line = "202"
                                # elif id_line == "line_e918616d-64f8-43d2-869c-f687726212be":
                                #     txt_gt_line = "214"
                                # elif id_line == "line_ebd8f850-1da5-45b1-b59c-9349497ecc8e":
                                #     txt_gt_line = "216"
                                #
                                # label_line[id_file + "_" + str(index_line)] = txt_gt_line
                                # text_page += txt_gt_line
                                # text_page += "\n"

                index_line += 1

        # Label text page
        path_label_page = os.path.join(path_dir_label_page, id_file + ".txt")
        with open(path_label_page, 'w', encoding="utf-8") as file:
            if text_page is not None:
                file.write(text_page)
            else:
                print("No gt")
                file.write(" ")

        # 1 class line
        path_gt_yolo = os.path.join(path_dir_label_yolo, id_file + ".txt")
        with open(path_gt_yolo, 'w', encoding="utf-8") as file:
            for one_line in yolo_line_coordinate:
                # Yolo format: class x_center y_center width height
                file.write(str(0))  # class line
                file.write(" ")
                file.write(str(one_line[0]))
                file.write(" ")
                file.write(str(one_line[1]))
                file.write(" ")
                file.write(str(one_line[2]))
                file.write(" ")
                file.write(str(one_line[3]))
                file.write("\n")

        # Label text line level
        path_save_all_label = os.path.join(dir_save, "all_label.json")
        json_object = json.dumps(label_line, indent=4)

        with open(path_save_all_label, "w") as outfile:
            outfile.write(json_object)


if __name__ == "__main__":
    # Read 2016
    # # Training Validation  Test
    # path_db_xml = "C:/Users/simcor/dev/data/READ/2016/origin/Test/page/"  # page/"
    # path_db_img = "C:/Users/simcor/dev/data/READ/2016/origin/Test/"  # Images/"
    #
    # path_db_save = "C:/Users/simcor/dev/data/READ/2016/convert_line_polygon_test_delete"
    # #
    # convert_db_page_bb_min_max(path_db_xml, path_db_img, path_db_save)

    path_db_xml = "C:/Users/simcor/dev/data/READ/2016/origin/Training/page/page/"
    path_db_img = "C:/Users/simcor/dev/data/READ/2016/origin/Training/Images/"

    path_db_save = "C:/Users/simcor/dev/data/READ/2016/convert_line_polygon_train_delete"
    convert_db_page_bb_min_max(path_db_xml, path_db_img, path_db_save)

    # path_db_xml = "C:/Users/simcor/dev/data/READ/2016/origin/Validation/page/page/"
    # path_db_img = "C:/Users/simcor/dev/data/READ/2016/origin/Validation/Images/"
    #
    # path_db_save = "C:/Users/simcor/dev/data/READ/2016/convert_line_polygon_val_to_delete"
    # convert_db_page_bb_min_max(path_db_xml, path_db_img, path_db_save)

    print("End")


