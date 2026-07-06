import glob

from skimage import io


def print_stats_img_datasets(dir_data, extension="png"):
    if isinstance(extension, list):
        files = []

        for one_e in extension:
            one_ext_file = glob.glob(dir_data + '/**/*.' + one_e, recursive=True)
            files.extend(one_ext_file)
    else:
        files = glob.glob(dir_data + '/**/*.' + extension, recursive=True)

    min_w = 999999
    max_w = 0
    mean_w = 0

    dict_width = {}

    min_h = 999999
    max_h = 0
    mean_h = 0

    dict_height = {}

    nb_items = 0

    for one_file in files:
        nb_items += 1
        img = io.imread(one_file)

        if len(img.shape) == 2:
            h, w = img.shape
        elif len(img.shape) == 3:
            h, w, _ = img.shape
        else:
            print("Format image not supported -> continue")

        if h < min_h:
            min_h = h
        if h > max_h:
            max_h = h

        if h in dict_height:
            dict_height[h] += 1
        else:
            dict_height[h] = 1

        mean_h += h

        if w < min_w:
            min_w = w
        if w > max_w:
            max_w = w

        if w in dict_width:
            dict_width[w] += 1
        else:
            dict_width[w] = 1

        mean_w += w

    mean_w /= nb_items
    mean_h /= nb_items

    myKeys = list(dict_width.keys())
    myKeys.sort()
    sorted_dict_w = {i: dict_width[i] for i in myKeys}

    print("dict_width :")
    print(sorted_dict_w)
    print("min_w :" + str(min_w))
    print("max_w :" + str(max_w))
    print("mean_w :" + str(mean_w))
    print()
    myKeys = list(dict_height.keys())
    myKeys.sort()
    sorted_dict_h = {i: dict_height[i] for i in myKeys}

    print("dict_height :")
    print(sorted_dict_h)
    print("min_h :" + str(min_h))
    print("max_h :" + str(max_h))
    print("mean_h :" + str(mean_h))


if __name__ == '__main__':
    dir_data = "C:/Users/simcor/dev/data/READ/2016/convert_line_polygon_train/image_line"
    print_stats_img_datasets(dir_data, extension="jpg")


