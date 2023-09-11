from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from fibseg.utils import fibseg_model_utils
import torch.nn.functional as F

# =========================================================================================
# Encoder Decoder Model
# Vaanathi Sundaresan
# 05-08-2023
# =========================================================================================


class EncDec(nn.Module):
    """
    TrUE-Net model definition
    """
    def __init__(self, n_channels, n_classes, init_channels, plane='axial', bilinear=False):
        super(EncDec, self).__init__()
        self.n_channels = n_channels
        self.init_channels = init_channels
        self.n_classes = n_classes
        self.n_layers = 3
        self.bilinear = True

        self.inpconv = fibseg_model_utils.OutConv(n_channels, 3, name="inpconv_")
        if plane == 'axial' or plane == 'tc':
            self.convfirst = fibseg_model_utils.DoubleConv(3, init_channels, 3, name="convfirst_")
        else:
            self.convfirst = fibseg_model_utils.DoubleConv(3, init_channels, 5, name="convfirst_")
        self.down1 = fibseg_model_utils.Down(init_channels, init_channels*2, 3, name="down1_")
        self.down2 = fibseg_model_utils.Down(init_channels*2, init_channels*4, 3, name="down2_")
        self.down3 = fibseg_model_utils.Down(init_channels*4, init_channels*8, 3, name="down3_")
        self.up3 = fibseg_model_utils.Up(init_channels*8, init_channels*4, 2, "up3_", bilinear)
        self.up2 = fibseg_model_utils.Up(init_channels*4, init_channels*2, 2, "up2_", bilinear)
        self.up1 = fibseg_model_utils.Up(init_channels*2, init_channels, 2, "up1_", bilinear)
        self.outconv = fibseg_model_utils.OutConv(init_channels, n_classes, name="outconv_")
        self.outconvrecon = fibseg_model_utils.OutConv(init_channels, n_channels, name="outconvrec_")

    def forward(self, x):
        xi = self.inpconv(x)
        x1 = self.convfirst(xi)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.outconv(x)
        recon = self.outconvrecon(x)
        return logits, recon


class EncDecClass(nn.Module):
    '''
    TrUE-Net model definition
    '''
    def __init__(self, n_channels, n_classes, init_channels, feat_channels, plane='axial', bilinear=False):
        super(EncDecClass, self).__init__()
        self.n_channels = n_channels
        self.init_channels = init_channels
        self.n_classes = n_classes
        self.n_layers = 3
        self.bilinear = True
        self.feat_channels = feat_channels

        self.inpconv = fibseg_model_utils.OutConv(n_channels, 3, name="inpconv_")
        if plane == 'axial' or plane == 'tc':
            self.convfirst = fibseg_model_utils.DoubleConv(3, init_channels, 3, name="convfirst_")
        else:
            self.convfirst = fibseg_model_utils.DoubleConv(3, init_channels, 5, name="convfirst_")
        self.down1 = fibseg_model_utils.Down(init_channels, init_channels*2, 3, name="down1_")
        self.down2 = fibseg_model_utils.Down(init_channels*2, init_channels*4, 3, name="down2_")
        self.down3 = fibseg_model_utils.Down(init_channels*4, init_channels*8, 3, name="down3_")
        self.down4 = fibseg_model_utils.Down(init_channels * 8, init_channels * 8, 3, name="down3_")
        self.down5 = fibseg_model_utils.Down(init_channels * 8, init_channels * 8, 3, name="down3_")
        self.up3 = fibseg_model_utils.Up(init_channels*8, init_channels*4, 2, "up3_", bilinear)
        self.up2 = fibseg_model_utils.Up(init_channels*4, init_channels*2, 2, "up2_", bilinear)
        self.up1 = fibseg_model_utils.Up(init_channels*2, init_channels, 2, "up1_", bilinear)
        self.outconv = fibseg_model_utils.OutConv(init_channels, n_classes, name="outconv_")
        self.outconvbflin = fibseg_model_utils.OutConv(init_channels * 8, init_channels, name="outconv_")
        # self.outconvrecon = fibseg_model_utils.OutConv(init_channels, n_channels, name="outconvrec_")
        self.fc1 = fibseg_model_utils.FullConn(init_channels * 4 * 4, init_channels * 4, name="fc1")
        self.fc2 = fibseg_model_utils.FullConn(init_channels * 4, self.feat_channels, name="fc1")

    def forward(self, x, in_ch):
        xi = self.inpconv(x)
        x1 = self.convfirst(xi)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.outconv(x)
        x = self.down5(self.down4(x4))
        x = self.outconvbflin(x)
        x = x.view(-1, in_ch * 16)
        x = F.normalize(self.fc2(self.fc1(x)), dim=1)
        return logits, x


