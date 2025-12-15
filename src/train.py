import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import dataloaders, dataset_sizes
from src.model import get_model

# デバイスの設定 (使えるならmps、だめならcpu)
device = torch.device("mps" if torch.cuda.is_available() else "cpu")


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = 25
) -> nn.Module:
    since = time.time()

    # 最も精度の高かったモデルの重みを保存しておく変数
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # 各エポックで Train と Val のフェーズを繰り返す
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # データローダーからバッチを取り出す
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 勾配をリセット (掃除)
                optimizer.zero_grad()

                # 順伝播
                # 学習時のみ、勾配の履歴を記録する
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 学習フェーズなら、バックプロパゲーションと更新を行う
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # 最も良かった重みをロードして返す
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    # モデルのセットアップ
    model_ft = get_model(num_classes=2)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=5)
    torch.save(model_ft.state_dict(), "model_ants_bees.pth")
