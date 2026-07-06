def read_line_bb_xyxy_txt(path_file):
    line_seg = []

    with open(path_file, encoding="utf-8") as file:
        all_lines = file.readlines()

        for one_l in all_lines:
            one_l = one_l.replace("\n", "")

            one_l_split = one_l.split(sep=" ")

            if len(one_l_split) == 4:
                x1 = float(one_l_split[0])
                y1 = float(one_l_split[1])
                x2 = float(one_l_split[2])
                y2 = float(one_l_split[3])

                line_seg.append([x1, y1, x2, y2])

    return line_seg