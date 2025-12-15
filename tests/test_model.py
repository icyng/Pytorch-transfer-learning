import torch
from src.model import get_model

def test_model_output_shape():
    """
    モデルの出力サイズが(バッチサイズ, クラス数)になっているか検証する
    """
    
    # --- 1. Arrange (準備) ---
    # テスト対象のモデルを準備（2クラス分類用）
    num_classes = 2
    model = get_model(num_classes=num_classes)
    
    # モデルへの入力となるダミーデータを作成
    # (バッチサイズ=1, チャンネル=3, 高さ=224, 幅=224) のランダムな数値
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    # --- 2. Act (実行) ---
    # 実際にモデルにデータを通す
    output = model(dummy_input)

    # --- 3. Assert (検証) ---
    # 出力の形が [1, 2] になっているか確認
    # (ここが [1, 1000] のままだと、モデルの改変に失敗していることになる)
    assert output.shape == (batch_size, num_classes)
    
    print("Model output shape test passed!")