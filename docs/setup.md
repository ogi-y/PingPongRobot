# Pythonパッケージのインストール手順（自己責任でお願いします）
1. パッケージをシステムにインストール
~~~(bash)
sudo pip install ultralytics deepface --break-system-packages
~~~
2. 重要：numpyのダウングレード
~~~(bash)
sudo pip install "numpy<2" --break-system-packages
~~~