# 競艇AI予測システム

競艇（ボートレース）の2連単予測を行うAIシステム。
期待値ベースの賭け戦略で回収率向上を目指す。

## プロジェクト構成

```
boatrace-ai/
├── config/
│   └── settings.py          # 設定ファイル
├── data/
│   ├── raw/                  # 生データ（LZH, TXT）
│   │   ├── results/          # 競走成績
│   │   └── programs/         # 番組表
│   └── processed/            # 処理済みデータ（CSV）
├── src/
│   ├── data_collection/      # データ取得
│   │   ├── downloader.py     # ダウンローダー
│   │   └── extractor.py      # LZH解凍
│   ├── preprocessing/        # 前処理
│   │   └── parser.py         # パーサー
│   ├── models/               # 予測モデル（Phase 2）
│   └── api/                  # 推論API（Phase 4）
├── notebooks/                # Jupyter探索用
├── requirements.txt
└── README.md
```

## セットアップ

```bash
# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージインストール
pip install -r requirements.txt
```

## 使い方

### Phase 1: データ取得

```bash
# 1. データダウンロード（2023-2024年、約2年分）
python src/data_collection/downloader.py

# 2. LZH解凍
python src/data_collection/extractor.py

# 3. CSVに変換
python src/preprocessing/parser.py
```

## データソース

- 競走成績: http://www1.mbrace.or.jp/od2/K/
- 番組表: http://www1.mbrace.or.jp/od2/B/
- 選手データ: https://www.boatrace.jp/owpc/pc/extra/data/download.html

## 開発フェーズ

- [x] Phase 1: データ収集 & 探索
- [ ] Phase 2: モデル構築
- [ ] Phase 3: バックテスト
- [ ] Phase 4: 推論API & UI

## 戦略

**期待値ベース**
```
期待値 = 予測的中確率 × オッズ
期待値 > 1.0 の買い目のみ購入
```

## 拡張予定

- 券種拡張（3連単、3連複）
- 目標金額逆算機能
- ケリー基準によるベット最適化
