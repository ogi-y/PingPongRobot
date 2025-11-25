# Pythonパッケージのインストール手順（自己責任でお願いします）
1. パッケージをシステムにインストール
~~~(bash)
sudo pip install ultralytics deepface --break-system-packages
~~~
2. 重要：numpyのダウングレード
~~~(bash)
sudo pip install "numpy<2" --break-system-packages
~~~

# その他必要な準備
1. テンプレートの作成 
  - 対象：`src/cpp_pingpong/src/controller.cpp`
  - 内容：レベルとパラメータの定義
2. パラメータの微調整
  - 対象：`src/cpp_pingpong/src/ballistics.cpp`（77 ~ 88行目）
  - 内容：距離と対応する仰角、発射パワーの設定