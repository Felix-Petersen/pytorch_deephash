import torch.nn as nn
from torchvision import models

import algonet

# alexnet_model = models.alexnet(pretrained=True)
alexnet_model = models.alexnet(pretrained=False)
# print(list(alexnet_model.features.children()))
# [
# Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
# ReLU(inplace),
# +92.330%  2
# MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
# Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
# ReLU(inplace),
# +92.490%  5
# MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
# Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
# ReLU(inplace),
# Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
# ReLU(inplace),
# Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
# ReLU(inplace),
# +scc top center
# MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
# ]


class AlexNetPlusLatent(nn.Module):
    def __init__(self, bits):
        super(AlexNetPlusLatent, self).__init__()
        self.bits = bits
        self.algonet = algonet.FiniteDifferences(padding='zeros')
        features = list(alexnet_model.features.children())
        # features.insert(5, self.algonet)
        self.features = nn.Sequential(*features)
        self.remain = nn.Sequential(*list(alexnet_model.classifier.children())[:-1])
        self.Linear1 = nn.Linear(4096, self.bits)
        self.sigmoid = nn.Sigmoid()
        self.Linear2 = nn.Linear(self.bits, 10)
    def forward(self, x):
        # x = self.algonet(x)
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.remain(x)
        x = self.Linear1(x)
        features = self.sigmoid(x)
        result = self.Linear2(features)
        print('###features.insert(, self.algonet)')
        return features, result



# Without:
# /usr/local/bin/python3.6 "/Users/felixpetersen/Library/Mobile Documents/com~apple~CloudDocs/PFS/AlgoNet/experiments/pytorch_deephash/train.py"
# Downloading: "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth" to /Users/felixpetersen/.torch/models/alexnet-owt-4df8aa71.pth
# 100%|██████████| 244418560/244418560 [00:20<00:00, 11732692.59it/s]
# Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
# Failed download. Trying https -> http instead. Downloading http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
# Files already downloaded and verified
#
# Epoch: 1
# 0 391 Loss: 2.393 | Acc: 10.938% (14/128)
# 1 391 Loss: 2.392 | Acc: 11.328% (29/256)
# 2 391 Loss: 2.370 | Acc: 9.896% (38/384)
# 3 391 Loss: 2.360 | Acc: 9.961% (51/512)
# 4 391 Loss: 2.345 | Acc: 9.531% (61/640)
# 5 391 Loss: 2.327 | Acc: 11.068% (85/768)
# 6 391 Loss: 2.313 | Acc: 12.946% (116/896)
# 7 391 Loss: 2.302 | Acc: 13.867% (142/1024)
# 8 391 Loss: 2.289 | Acc: 14.410% (166/1152)
# 9 391 Loss: 2.277 | Acc: 15.391% (197/1280)
# 10 391 Loss: 2.264 | Acc: 16.690% (235/1408)
# 11 391 Loss: 2.247 | Acc: 18.945% (291/1536)
# 12 391 Loss: 2.234 | Acc: 20.433% (340/1664)
# 13 391 Loss: 2.217 | Acc: 22.600% (405/1792)
# 14 391 Loss: 2.204 | Acc: 23.333% (448/1920)
# 15 391 Loss: 2.185 | Acc: 24.658% (505/2048)
# 16 391 Loss: 2.167 | Acc: 25.551% (556/2176)
# 17 391 Loss: 2.153 | Acc: 26.476% (610/2304)
# 18 391 Loss: 2.137 | Acc: 27.590% (671/2432)
# 19 391 Loss: 2.121 | Acc: 28.633% (733/2560)
# 20 391 Loss: 2.107 | Acc: 29.427% (791/2688)
# 21 391 Loss: 2.092 | Acc: 30.327% (854/2816)
# 22 391 Loss: 2.074 | Acc: 31.318% (922/2944)
# 23 391 Loss: 2.060 | Acc: 32.031% (984/3072)
# 24 391 Loss: 2.047 | Acc: 33.062% (1058/3200)
# 25 391 Loss: 2.031 | Acc: 34.075% (1134/3328)
# 26 391 Loss: 2.017 | Acc: 34.838% (1204/3456)
# 27 391 Loss: 2.003 | Acc: 35.547% (1274/3584)
# 28 391 Loss: 1.992 | Acc: 35.830% (1330/3712)
# 29 391 Loss: 1.980 | Acc: 36.250% (1392/3840)
# 30 391 Loss: 1.967 | Acc: 36.971% (1467/3968)
# 31 391 Loss: 1.955 | Acc: 37.646% (1542/4096)
# 32 391 Loss: 1.944 | Acc: 38.329% (1619/4224)
# 33 391 Loss: 1.932 | Acc: 38.810% (1689/4352)
# 34 391 Loss: 1.922 | Acc: 39.375% (1764/4480)
# 35 391 Loss: 1.912 | Acc: 39.909% (1839/4608)
# 36 391 Loss: 1.899 | Acc: 40.498% (1918/4736)
# 37 391 Loss: 1.891 | Acc: 40.851% (1987/4864)
# 38 391 Loss: 1.880 | Acc: 41.206% (2057/4992)
# 39 391 Loss: 1.874 | Acc: 41.504% (2125/5120)
# 40 391 Loss: 1.867 | Acc: 41.578% (2182/5248)
# 41 391 Loss: 1.859 | Acc: 41.722% (2243/5376)
# 42 391 Loss: 1.851 | Acc: 42.060% (2315/5504)
# 43 391 Loss: 1.842 | Acc: 42.188% (2376/5632)
# 44 391 Loss: 1.832 | Acc: 42.674% (2458/5760)
# 45 391 Loss: 1.824 | Acc: 43.037% (2534/5888)
