from skimage.transform import resize
import numpy as np


def centered_das(word_img, tsize, centering=(.5, .5), border_value=None):

    height = tsize[0]
    width = tsize[1]

    xs, ys, xe, ye = 0, 0, width, height
    diff_h = height-word_img.shape[0]
    if diff_h >= 0:
        pv = int(centering[0] * diff_h)
        padh = (pv, diff_h-pv)
    else:
        diff_h = abs(diff_h)
        ys, ye = diff_h/2, word_img.shape[0] - (diff_h - diff_h/2)
        padh = (0, 0)
    diff_w = width - word_img.shape[1]
    if diff_w >= 0:
        pv = int(centering[1] * diff_w)
        padw = (pv, diff_w - pv)
    else:
        diff_w = abs(diff_w)
        xs, xe = diff_w / 2, word_img.shape[1] - (diff_w - diff_w / 2)
        padw = (0, 0)

    if border_value is None:
        border_value = np.median(word_img)
    word_img = np.pad(word_img[ys:ye, xs:xe], (padh, padw), 'constant', constant_values=border_value)
    return word_img


def image_resize_das(img, height=None, width=None):
    if height is not None and width is None:
        scale = float(height) / float(img.shape[0])
        width = int(scale*img.shape[1])

    if width is not None and height is None:
        scale = float(width) / float(img.shape[1])
        height = int(scale*img.shape[0])

    # if img.shape[0] >0 and img.shape[1] > 0:
    img = resize(image=img, output_shape=(height, width)).astype(np.float32)
        
    return img


def preprocess_img_line(img_in, fixed_size, pad_left, pad_right):
    # image value should be [0, 255]

    # Resize and pad
    img = 1 - img_in.astype(np.float32) / 255.0

    fheight, fwidth = fixed_size[0], fixed_size[1]

    # default is DAS resize policy
    nheight, nwidth = img.shape[0], img.shape[1]

    nheight, nwidth = max(4, min(fheight-16, nheight)), max(8, min(fwidth-32, nwidth))
    img = image_resize_das(img, height=int(1.0 * nheight), width=int(1.0 * nwidth))

    img = centered_das(img, (fheight, fwidth), border_value=0.0)

    img = np.pad(img, ((0, 0), (pad_left, pad_right)), 'constant', constant_values=0)

    return img



