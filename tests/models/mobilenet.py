#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from math import floor
import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm


class MobileNet(nn.Module):
    def __init__(self, channel_multiplier=1.0, min_channels=8):
        super().__init__()

        if channel_multiplier <= 0:
            raise ValueError("channel_multiplier must be >= 0")

        def conv_bn_relu(n_ifm, n_ofm, kernel_size, stride=1, padding=0, groups=1):
            return [
                nn.Conv2d(
                    n_ifm,
                    n_ofm,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(n_ofm),
                nn.ReLU(inplace=True),
            ]

        def depthwise_conv(n_ifm, n_ofm, stride):
            return nn.Sequential(
                *conv_bn_relu(n_ifm, n_ifm, 3, stride=stride, padding=1, groups=n_ifm),
                *conv_bn_relu(n_ifm, n_ofm, 1, stride=1),
            )

        base_channels = [32, 64, 128, 256, 512, 1024]
        self.channels = [
            max(floor(n * channel_multiplier), min_channels) for n in base_channels
        ]

        self.model = nn.Sequential(
            nn.Sequential(*conv_bn_relu(3, self.channels[0], 3, stride=2, padding=1)),
            depthwise_conv(self.channels[0], self.channels[1], 1),
            depthwise_conv(self.channels[1], self.channels[2], 2),
            depthwise_conv(self.channels[2], self.channels[2], 1),
            depthwise_conv(self.channels[2], self.channels[3], 2),
            depthwise_conv(self.channels[3], self.channels[3], 1),
            depthwise_conv(self.channels[3], self.channels[4], 2),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[5], 2),
            depthwise_conv(self.channels[5], self.channels[5], 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(self.channels[5], 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, x.size(1))
        x = self.fc(x)
        return x


def get_mobilenet_untrained(dataset: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset in ["imagenet", "imagenet_memory"]:
        return MobileNet()
    raise NotImplementedError(f"MobileNet (untrained) is not available for {dataset}")


def get_mobilenet_trained(dataset: str, checkpoint_dir: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset in ["imagenet", "imagenet_memory"]:
        model = MobileNet()
        checkpoint_path = os.path.join(checkpoint_dir, "mobilenet.pth")
        state_trained = torch.load(checkpoint_path, map_location=torch.device("cpu"))[
            "state_dict"
        ]
        new_state_trained = model.state_dict()
        for k in state_trained:
            key = k[7:]
            if key in new_state_trained:
                new_state_trained[key] = state_trained[k].view(
                    new_state_trained[key].size()
                )
            else:
                print("Missing key", key)
        model.load_state_dict(new_state_trained, strict=False)
        tqdm.write(f"Loaded MobileNet weights from {checkpoint_path}")
        return model
    raise NotImplementedError(f"MobileNet (trained) is not available for {dataset}")
