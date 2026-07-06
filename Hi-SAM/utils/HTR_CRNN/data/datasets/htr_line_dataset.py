import glob
import os

import numpy as np
import torch
from skimage import io
from skimage.transform import rescale
from torch.utils.data import Dataset


from utils.HTR_CRNN.data.image.preprocess_img import image_resize_das, centered_das
from utils.HTR_CRNN.data.text.text_to_index import transcript_text_to_index


class HTRLineDataset(Dataset):
    """
    dataset for multiple line datasets
    if one dataset -> list of one data_info

    load image and label for htr
    """

    def __init__(self,
                 data_info,
                 fixed_size,
                 width_divisor,
                 pad_left,
                 pad_right,
                 char_dict,
                 transforms: list = None,
                 load_img_as_grayscale: int = 1,
                 apply_noise: int = 0,
                 is_trainset=False,
                 ratio_upscale: float = 2.0,
                 all_labels={}):
        """
        """

        self.name_db = ""
        self.image_paths = []

        self.labels_str = []  # text version
        self.labels_ind = []  # index version
        self.id_item = []

        self.fixed_size = fixed_size
        self.transforms = transforms
        self.pad_left = pad_left
        self.pad_right = pad_right

        self.width_divisor = width_divisor

        self.apply_noise = apply_noise
        self.is_trainset = is_trainset

        self.load_img_as_grayscale = load_img_as_grayscale

        self.ratio_upscale = ratio_upscale

        for one_db in data_info:
            # one db: name_db, img_dir, extension_img
            print(one_db[0])
            if len(self.name_db) > 0:
                self.name_db += " "  # add space
            self.name_db += one_db[0]
            dir_img = one_db[1]
            # dir_label = one_db[2]
            ext_img = one_db[2]

            # self.dirs_label.append(dir_label)

            counter_nb_sample = 0

            if ext_img == "pngjpg":
                files_img = glob.glob(dir_img + '/**/*.png', recursive=True)

                files_img.extend(glob.glob(dir_img + '/**/*.jpg', recursive=True))

            else:
                files_img = glob.glob(dir_img + '/**/*.' + ext_img, recursive=True)

            for one_file in files_img:
                # Get id file
                split_name = os.path.split(one_file)
                split_name = split_name[1].split(sep=".")  # Filename and extension
                id_file = split_name[0]

                # 1 file with all label

                if id_file in all_labels:
                    label_str = all_labels[id_file]
                    # Add space padding
                    label_str = " " + label_str + " "
                else:
                    print(one_file)
                    print("label text associated doesn't exist. Data not loaded")
                    continue

                label_ind = transcript_text_to_index(char_dict, label_str)  # text_read.transcript_txt_to_index(label_str)

                if label_str == " " or label_str == "  " or label_str == "   ":
                    print(one_file)
                    print("label text empty. Data not loaded")
                    continue

                self.labels_str.append(label_str)
                self.labels_ind.append(label_ind)

                self.id_item.append(id_file)

                self.image_paths.append(one_file)

                counter_nb_sample += 1

            print("Nb samples: " + str(counter_nb_sample))

    def __len__(self):
        """
        Returns the number of images in the dataset
        Returns
        -------
        length: int
            number of images in the dataset
        """

        return len(self.image_paths)

    def preprocess_img_grayscale(self, path_img):
        img = io.imread(path_img, as_gray=True)  # , plugin='pil' add plugin for .tif image -> deprecated

        if self.ratio_upscale != 1:
            img = rescale(img, scale=self.ratio_upscale, anti_aliasing=False)  # convert to 0 - 1

        # Binarized img
        if img.dtype == bool:
            img = img.astype(int)
            img *= 255

        # Color image -> grayscale -> value between 0 and 1
        if img.dtype == float:
            img *= 255.0

        # grayscale image -> uint value 0 - 255
        # img_tensor_tmp = torch.as_tensor(img, dtype=torch.float32)
        # img_tensor_tmp /= 255.0
        # save_image(img_tensor_tmp, 'C:/Users/simcor/dev/logs/img_origin.png')

        # Invert color and normalized
        img = 1 - img.astype(np.float32) / 255.0

        # Resize (Best practices paper) and pad
        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]

        if self.is_trainset:
            nwidth = int(np.random.uniform(.75, 1.25) * img.shape[1])
            nheight = int((np.random.uniform(.9, 1.1) * img.shape[0] / img.shape[1]) * nwidth)
        else:
            nheight, nwidth = img.shape[0], img.shape[1]

        nheight, nwidth = max(4, min(fheight-16, nheight)), max(8, min(fwidth-32, nwidth))
        img = image_resize_das(img, height=int(1.0 * nheight), width=int(1.0 * nwidth))
        img = centered_das(img, (fheight, fwidth), border_value=0.0)

        img = np.pad(img, ((0, 0), (self.pad_left, self.pad_right)), 'constant', constant_values=0)

        # Augmentation
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        img_tensor = torch.as_tensor(img, dtype=torch.float32)

        # cf. DAS 2022
        if self.apply_noise == 1:
            if np.random.rand() < .33:
                img_tensor += torch.rand(img_tensor.size())

        img_tensor = img_tensor.unsqueeze(0)  # Add channel dim

        return img_tensor

    # def preprocess_img_color(self, path_img):
    #     # dimension: height, width, channel
    #     img = io.imread(path_img, as_gray=False)  # , plugin='pil' add plugin for .tif image -> deprecated
    #
    #     img = img.astype(np.float32) / 255.0
    #
    #     # Case grayscale image
    #     if len(img.shape) == 2:
    #         img = np.expand_dims(img, axis=-1)
    #         img = img.repeat(3, axis=-1)
    #     # # Binarize img
    #     # if img.dtype == bool:
    #     #     img = img.astype(int)
    #     #     img *= 255
    #     #
    #     # # Color image -> grayscale -> value between 0 and 1
    #     # if img.dtype == float:
    #     #     img *= 255.0
    #     #
    #     # # grayscale image -> uint value 0 - 255
    #     # # save_image(img, 'C:/Users/simcor/dev/logs/img1.png')
    #
    #     # Resize and pad
    #     img = 1 - img  #.astype(np.float32) / 255.0
    #
    #     fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
    #
    #     if self.resize_config == ResizeInputPolicy.ICDAR_2025:
    #         # Resize small image
    #         nheight, nwidth = img.shape[0], img.shape[1]
    #
    #         fheight_min = int(self.ratio_small * fheight)
    #         fwidth_min = int(self.ratio_small * fwidth)
    #
    #         if nheight <= fheight_min:
    #             img = image_resize_height(img, fheight_min)
    #             # img_tensor_tmp = torch.as_tensor(img, dtype=torch.float32)
    #             # save_image(img_tensor_tmp, 'C:/Users/simcor/dev/logs/img1_resize_min_h_img.png')
    #
    #         nheight, nwidth = img.shape[0], img.shape[1]
    #         if nwidth <= fwidth_min:
    #             img = image_resize_width(img, fwidth_min)
    #             # img_tensor_tmp = torch.as_tensor(img, dtype=torch.float32)
    #             # save_image(img_tensor_tmp, 'C:/Users/simcor/dev/logs/img1_resize_min_w_img.png')
    #
    #         # Resize bigger image
    #         nheight, nwidth = img.shape[0], img.shape[1]
    #         if nheight > fheight or nwidth > fwidth:
    #             img = image_resize_larger_height_width_keep_aspect_ratio(img, fheight, fwidth)
    #
    #         # img_tensor_tmp = torch.as_tensor(img, dtype=torch.float32)
    #         # save_image(img_tensor_tmp, 'C:/Users/simcor/dev/logs/img1_resize.png')
    #
    #         img = centered_das_color(img, (fheight, fwidth), border_value=0.0)
    #
    #         img = np.pad(img, ((4, 4), (self.pad_left, self.pad_right), (0, 0)), 'constant', constant_values=0)
    #
    #         # img_tensor_tmp = torch.as_tensor(img, dtype=torch.float32)
    #         # save_image(img_tensor_tmp, 'C:/Users/simcor/dev/logs/img1_resize_end.png')
    #     else:
    #         print("Preprocess image not implemented")
    #         # # default is DAS resize policy
    #         # if self.is_trainset:
    #         #     nwidth = int(np.random.uniform(.75, 1.25) * img.shape[1])
    #         #     nheight = int((np.random.uniform(.9, 1.1) * img.shape[0] / img.shape[1]) * nwidth)
    #         # else:
    #         #     nheight, nwidth = img.shape[0], img.shape[1]
    #         #
    #         # nheight, nwidth = max(4, min(fheight-16, nheight)), max(8, min(fwidth-32, nwidth))
    #         # img = image_resize_das(img, height=int(1.0 * nheight), width=int(1.0 * nwidth))
    #         #
    #         # img = centered_das(img, (fheight, fwidth), border_value=0.0)
    #         #
    #         # img = np.pad(img, ((0, 0), (self.pad_left, self.pad_right)), 'constant', constant_values=0)
    #
    #     # H, W, C
    #     # img_tensor_tmp = torch.as_tensor(img, dtype=torch.float32)
    #     # img_tensor_tmp = img_tensor_tmp.permute(2, 0, 1)  # C, H, W
    #     # save_image(img_tensor_tmp, 'C:/Users/simcor/dev/logs/img1_resize.png')
    #
    #     # Augmentation
    #     if self.transforms is not None:
    #         img = self.transforms(image=img)['image']
    #
    #     # imgs_shape = img.shape
    #     # w_reduce = np.floor(imgs_shape[1] / self.width_divisor).astype(int)
    #
    #     img_tensor = torch.as_tensor(img, dtype=torch.float32)
    #     img_tensor = img_tensor.permute(2, 0, 1)  # C, H, W
    #
    #     # cf. DAS 2022
    #     if self.apply_noise == 1:
    #         if np.random.rand() < .33:
    #             img_tensor += torch.rand(img_tensor.size())
    #
    #     # save_image(img_tensor_tmp, 'C:/Users/simcor/dev/logs/img1_resize_end.png')
    #
    #     return img_tensor

    def __getitem__(self, idx):
        """
        """
        path_img = self.image_paths[idx]

        if self.load_img_as_grayscale == 1:
            img_tensor = self.preprocess_img_grayscale(path_img)
        else:
            img_tensor = self.preprocess_img_color(path_img)

        imgs_shape = img_tensor.shape
        w_reduce = np.floor(imgs_shape[2] / self.width_divisor).astype(int)

        sample = {
            "ids": self.id_item[idx],

            "label_str": self.labels_str[idx],
            "label_ind": self.labels_ind[idx],

            "img": img_tensor,
            "w_reduce": w_reduce
        }

        return sample
