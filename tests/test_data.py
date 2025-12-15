import torch
from src.data_loader import dataloaders

def test_dataloader_shape():
    """
    データローダーが正しいバッチサイズと画像サイズを返すか検証
    """
    # --- 1. Arrange & 2. Act ---
    # train用のデータローダーから、最初の1バッチだけを取り出す
    inputs, classes = next(iter(dataloaders['train']))

    # --- 3. Assert ---
    # バッチサイズなどの期待値
    expected_batch_size = 4
    expected_channels = 3
    expected_height = 224
    expected_width = 224

    # 画像データの形をチェック: [4, 3, 224, 224] になっているはず
    assert inputs.shape == (expected_batch_size, expected_channels, expected_height, expected_width)
    
    # ラベル（正解データ）の形をチェック: [4] になっているはず
    assert classes.shape == (expected_batch_size,)
    
    print("DataLoader shape test passed!")