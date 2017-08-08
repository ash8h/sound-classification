# Sound Classification with tenforflow/keras

[Urban Sound Classification with Neural Networks in Tensorflow](
http://www.kdnuggets.com/2016/09/urban-sound-classification-neural-networks-tensorflow.html)
を動かそうと思ったが、サンプルコードで想定しているpythonとtenforflowのバージョンが古くて最新の環境で動かないので、コードをいろいろいじって動くようにしてみた.

## Environment
- Anaconda + ipython + python3.6
- tenforflow 1.2 + keras
- LibROSA: オーディオ信号処理のためのpythonライブラリ

## Dataset

[URBANSOUND8K DATASET](https://serv.cusp.nyu.edu/projects/urbansounddataset/)使用。街中での音を集めたデータセット. 4秒以下の音データがラベル付きで8,732個入っている. ラベルは以下の10種類. オーディオデータのフォーマットは wav.

- air_conditioner
- car_horn
- children_playing
- dog_bark
- drilling
- enginge_idling
- gun_shot
- jackhammer
- siren
- street_music

## Source Code

ipythonで編集/動作確認している. `.py`ファイルは`.ipynb`から自動生成.

- [urbansound8k_loader.ipynb](/urbansound8k_loader.ipynb) ([.py](/urbansound8k_loader.py)) : ローカルに保存したURBANSOUND8Kデータセットを読み込むためのヘルパーライブラリ. 毎回wavから読み込むととても時間がかかるので、一度読み込んだデータを特徴量抽出して`npy`形式で保存し、次回以降は`.npy`から読み込めるようにした.
- [sound_classification_tensorflow.ipynb](/sound_classification_tensorflown.ipynb) ([.py](/sound_classification_tensorflow.py)) : tenforflowを使ったsound classification
- [sound_classification.ipyn](/sound_classification.ipynb) ([.py](/sound_classification.py)) : Kerasを使った sound classification


