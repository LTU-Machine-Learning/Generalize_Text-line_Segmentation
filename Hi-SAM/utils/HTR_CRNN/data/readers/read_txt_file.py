def read_page_txt_gt(path_file):
    content_str = ""

    # encoding="utf-8 Use for letter with diacritics
    with open(path_file, encoding="utf-8") as file:  # mode="r
        all_lines = file.readlines()

        for one_l in all_lines:
            one_l = one_l.replace("\n", " ")

            content_str += one_l

    return content_str
