import torch.nn as nn
from torchvision import models


def get_model(num_classes: int = 2) -> nn.Module:
    # 学習済みのResNet18モデルを読み込む
    model = models.resnet18(weights="DEFAULT")

    # 全結合層を付け替える
    # 元々の出口に入ってくる信号の数（in_features）を調べる
    num_ftrs = model.fc.in_features

    # その数を受け取って、num_classes（今回は2個）の答えを出す新しい層に置き換える
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
