def read_yolo_gt(path_file):
    gt = []

    with open(path_file, encoding="utf-8") as file:
        all_lines = file.readlines()

        for one_l in all_lines:
            # Yolo format: class x_center y_center width height
            one_l = one_l.replace("\n", "")

            one_l_split = one_l.split(sep=" ")

            if len(one_l_split) == 5:
                class_id = int(one_l_split[0])
                # Value normalized
                x_center = float(one_l_split[1])
                y_center = float(one_l_split[2])
                width = float(one_l_split[3])
                height = float(one_l_split[4])

                gt.append([class_id, x_center, y_center, width, height])

    return gt
