# Advanced Artificial Intelligence Applications I 期末レポート

#### 1\. 課題概要

本課題では、公開されている機械学習コードに対し、講義で学んだモダンな開発手法（パッケージ管理、静的解析、テスト駆動開発など）を適用し、再現性とコード品質の向上を図る

#### 2\. 対象ソース

  * **元コード:** PyTorch Tutorial: Transfer Learning for Computer Vision
  * **URL:** `https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html`

#### 3\. 主な修正・改善内容

講義で扱った以下のベストプラクティスを適用

1.  **プロジェクト構造の再構築:**

      * 単一のスクリプトであったコードを、`src` ディレクトリ構成（`data_loader.py`, `model.py`, `train.py`）に分割し、責務を明確化

2.  **再現性のある環境構築 (uv):**

      * `uv` を導入し、`pyproject.toml` による依存関係の管理を行った、これにより `uv sync` コマンド一つで誰でも同一の実行環境を再現可能

3.  **コード品質の向上 (Ruff & Type Hints):**

      * 静的解析ツール `Ruff` を導入し、PEP 8 に準拠したフォーマットを適用
      * 主要な関数（`train_model`, `get_model` 等）に型ヒント（Type Hints）を付与し、可読性と保守性を向上

4.  **テストの実装 (pytest):**

      * `pytest` を導入し、以下の単体テストを実装することでコード変更時の安全性を担保
          * `test_model.py`: モデルの出力形状がクラス数と一致するか検証
          * `test_data.py`: データローダーが期待通りのバッチサイズと形状でデータを供給するか検証