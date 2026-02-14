標高データから道路網を生成するAIモデルです。
標高データから計算された勾配データを第2の入力として与えることで、
地形の険しさを考慮した道路生成を行います。

■ フォルダ構成
pix2pix_map/
├── dataset/
│   └── train/
│       ├── height/   <-- 標高画像を配置 (例: area01_height.png)
│       └── road/     <-- 対応する道路画像を配置 (例: area01_road.png)
├── result/           <-- 生成された画像が保存されます
├── params/           <-- 学習済みモデル(.pth)が保存されます
└── pix2pix_map.py    <-- 実行用コード

■ 実行方法
1. 必要なライブラリのインストール:
   pip install torch torchvision opencv-python tqdm numpy pillow

2. 学習の開始:
   python pix2pix_map.py --mode train --root_dir ./dataset/train
