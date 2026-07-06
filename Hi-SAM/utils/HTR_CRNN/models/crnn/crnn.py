import torch.nn as nn

from utils.HTR_CRNN.models.cnn.cnn import CNN
from utils.HTR_CRNN.models.heads.crnn_heads import CTCHead


# From https://github.com/georgeretsi/HTR-best-practices/
class CRNN(nn.Module):
    def __init__(self, cnn_cfg, head_cfg, nclasses,
                 dropout_last_fc_crnn=0.2, dropout_lstm=0.2, load_img_as_grayscale=1, add_squeeze_excitation=1):
        super(CRNN, self).__init__()

        self.features = CNN(cnn_cfg, load_img_as_grayscale, add_squeeze_excitation)
        hidden = cnn_cfg[-1][-1]

        self.top = CTCHead(hidden, head_cfg, nclasses, dropout_last_fc_crnn, dropout_lstm)

        self.nclasses = nclasses

    def forward(self, x):

        y = self.features(x)  # Output dimension: batch size, 256, 1,  nb frames

        before_blstm = y

        y, after_blstm = self.top(y)

        return y, before_blstm, after_blstm

