# Copyright (c) Opendatalab. All rights reserved.
import torch.nn.functional as F
from torch import nn


class CTCHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=6625,
        fc_decay=0.0004,
        mid_channels=None,
        return_feats=False,
        use_guide=False,
        **kwargs
    ):
        super(CTCHead, self).__init__()
        self.use_guide = use_guide
        if use_guide:
            self.conv1 = nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=5,
                padding=2,
                groups=in_channels,
                bias=False,
            )
            self.norm1 = nn.BatchNorm1d(in_channels)
            self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
            self.norm2 = nn.BatchNorm1d(in_channels)
            self.act_fn = nn.Hardswish()
        if mid_channels is None:
            self.fc = nn.Linear(
                in_channels,
                out_channels,
                bias=True,
            )
        else:
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels,
                bias=True,
            )
            self.fc2 = nn.Linear(
                mid_channels,
                out_channels,
                bias=True,
            )

        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    def forward(self, x, labels=None):
        if self.use_guide:
            x = x.transpose(1, 2)
            x = self.act_fn(self.norm1(self.conv1(x)))
            x = self.act_fn(self.norm2(self.conv2(x)))
            x = x.transpose(1, 2)
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts

        if not self.training:
            predicts = F.softmax(predicts, dim=2)
            result = predicts

        return result
