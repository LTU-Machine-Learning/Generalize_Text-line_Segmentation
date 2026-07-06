<h1 align="center">Hi-SAM model for Text Line Segmentation</h1> 

<p>The original work is from:</p>


> [Hi-SAM: Marrying Segment Anything Model for Hierarchical Text Segmentation](https://arxiv.org/abs/2401.17904).
>
>  [IEEE TPAMI]
This was copied from https://github.com/ymy-k/Hi-SAM which is the official repository for Hi-SAM, a unified hierarchical text segmentation model. Refer to yheir paper for more details.


## :What we added
- A pipeline for creating binary page image mask for Hi-SAM training.
- mAP, F1 and CER as metrics to measure performance
- Updated code to run with torchrun (for multi GPU training)


## :sparkles: Highlight

![overview](.asset/overview.jpg)

- **Hierarchical Text Segmentation.** Hi-SAM unifies text segmentation across stroke, word, text-line, and paragraph levels. Hi-SAM also achieves layout analysis as a by-product.

- **Automatic and Interactive.** Hi-SAM supports both automatic mask generation and interactive promptable mode. Given a single-point prompt, Hi-SAM provides word, text-line, and paragraph masks.

- **High-quality Pixel-level Text (Stroke) Segmentation & Labeling Assistant.** High-quality pixel-level text (stroke) segmentation by introducing mask feature of 1024×1024 resolution with minimal modification in SAM's original mask decoder. Our contributed stroke level annotations for [HierText](https://github.com/google-research-datasets/hiertext) can be downloaded following [data_preparation.md](datasets/data_preparation.md). Some examples are displayed here:

![example](samples/comparison_figure-Hi-SAM.png)


## :bulb: Overview of Hi-SAM



**Recommended**: `Linux` `Python 3.11` `Pytorch 2.5 `CUDA 12.5`



# Training Data Preparation

## Checkpoints

Please download the following pre-train weights in-order to train the model.

|Model|Weights|
|:------:|:------:|
|Hi-SAM-H|[OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcoxoNjp1IG7xitzrg?e=0z4QhJ)|
|Mask-Decoder| [OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgctig8BXzlCQaPm1ng?e=6XOCid)|
|ViT Encoder|[SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

### Hi-SAM require following 3 input the for model training

- Images
- xml Ground Truth (GT) polygon coordinates
- Binarize images


However datasets does not come with binarize images by default. Therefore, we need to create them. Luckily, Hi-SAM itself has a banalization module (Pixel-level Text (Stroke) Segmentation). But it failed to cover the whole line when the document is heavily impacted by bleed-trough text. Therefore, use 3 stage approach.
- Generate binary masks from Hi-SAM
- Generate binary mask by Otsu thresholding. We apply this only to green channel (in RGB images) as it is least impacted by the bleed-through text.
- Combine both.
- Delete content not covered by the GT polygons.

![Binarization](samples/binarization.png)

<b>Generate binary masks from Hi-SAM</b>
```
python demo_hisam.py \
--pretrained_path ../../logs/Hi-SAM_Doc/pretrained_checkpoint \
--checkpoint ../../logs/Hi-SAM_Doc/pretrained_checkpoint/hi_sam_h.pth \
--model-type vit_h \
--input ../../logs/Hi-SAM_Doc/input_images/Bentham \
--output ../../logs/Hi-SAM_Doc/out \
--patch_mode 
```

<p><b>Generate Otsu masks and combine with Hi-SAM generated masks</b></p>
<p>Please follow the process in the following Notebook file.</p>

```
image_binarization.ipynb
```

:star: **`Note`:** 

1. For faster downloading and saving storage, **above checkpoints do not contain the parameters in SAM's ViT image encoder**. Please follow [segment-anything](https://github.com/facebookresearch/segment-anything) to achieve `sam_vit_b_01ec64.pth`, `sam_vit_l_0b3195.pth`, `sam_vit_h_4b8939.pth` and put them in `pretrained_checkpoint/` for loading the frozen parameters in ViT image encoder.
2. **To train Hi-SAM in yourself, in addition to download the SAM weights, please also download the isolated mask decoder weights and put them in `pretrained_checkpoint/` for initializing H-Decoder (or you can separate the mask decoder part from SAM weights in yourself).** [vit_b_maskdecoder.pth](https://1drv.ms/u/s!AimBgYV7JjTlgcth1ceH68P-vOF87g?e=lK2bIL) & [vit_l_maskdecoder.pth](https://1drv.ms/u/s!AimBgYV7JjTlgctjx03utTjx31EexA?e=HG7zZD) & [vit_h_maskdecoder.pth](https://1drv.ms/u/s!AimBgYV7JjTlgctig8BXzlCQaPm1ng?e=6XOCid) from [segment-anything](https://github.com/facebookresearch/segment-anything), [vit_s_maskdecoder.pth](https://1drv.ms/u/s!AimBgYV7JjTlgctkk7xz198vz5TOhQ?e=sCfhYm) from [EfficientSAM](https://github.com/yformer/EfficientSAM). For example, if you want to train Hi-SAM-L, it looks like this in `pretrained_checkpoint/`:

```
|- pretrained_checkpoint
|  |- sam_vit_l_0b3195.pth
|  └  vit_l_maskdecoder.pth
```


## :arrow_forward: Usage

### **1. Visualization Demo**

**1.1 Pixel-level Text (Stroke) Segmentation (for SAM-TS & Hi-SAM):**

```
python demo_hisam.py --checkpoint pretrained_checkpoint/sam_tss_l_hiertext.pth --model-type vit_l --input demo/2e0cb33320757201.jpg --output demo/
```

- `--checkpoint`: the model path.
- `--model-type`: the backbone type. Use `vit_b` for ViT-Base backbone,  `vit_l` for ViT-Large,  `vit_h` for ViT-Huge. Use `vit_s` for ViT-S.
- `--input`: the input image path.
- `--output`: the output image path or folder.

To achieve better quality on small texts using sliding window, run the following script:

```
python demo_hisam.py --checkpoint pretrained_checkpoint/sam_tss_l_hiertext.pth --model-type vit_l --input demo/2e0cb33320757201.jpg --output demo/2e0cb33320757201_sliding.png --patch_mode
```

- `--patch_mode`: enabling sliding window inference. The default patch size is 512×512, the stride is 384 (for HierText). You can adjust the setting for better result on your data.

**1.2 Word, Text-line, and Paragraph Segmentation (for Hi-SAM)**

Run the following script for promptable segmentation on demo/img293.jpg:

```
python demo_hisam.py --checkpoint pretrained_checkpoint/hi_sam_l.pth --model-type vit_l --input demo/img293.jpg --output demo/ --hier_det
```

- `--hier_det`: enabling hierarchical segmentation. Hi-SAM predicts a word mask, a text-line mask, and a paragraph mask for each single-point prompt. See demo_hisam.py for the point position and details.

### **2. Evaluation**


### **3. Training**


## 💗 Acknowledgement

- [segment-anything](https://github.com/facebookresearch/segment-anything), [EfficientSAM](https://github.com/yformer/EfficientSAM)
- [HierText](https://github.com/google-research-datasets/hiertext), [Total-Text](https://github.com/cs-chan/Total-Text-Dataset), [TextSeg](https://github.com/SHI-Labs/Rethinking-Text-Segmentation)
- The codebase is partially from [sam-hq](https://github.com/SysCV/sam-hq)


## :black_nib: Citation

If you find Hi-SAM helpful in your research, please consider giving this repository a :star: and citing:

```bibtex
@article{ye2025hi,
  title={Hi-SAM: Marrying Segment Anything Model for Hierarchical Text Segmentation},
  author={Ye, Maoyuan and Zhang, Jing and Liu, Juhua and Liu, Chenyu and Yin, Baocai and Liu, Cong and Du, Bo and Tao, Dacheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  volume={47},
  number={03},
  pages={1431--1447},
}
```
