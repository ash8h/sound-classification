
# coding: utf-8

# In[2]:

get_ipython().magic("config InlineBackend.figure_format='retina'")
# audioファイルを読み込んで可視化
import librosa, librosa.display
import matplotlib.pyplot as plt
get_ipython().magic("time sound, fs = librosa.audio.load('UrbanSound8K/audio/fold1/101415-3-0-2.wav')")
plt.figure()
librosa.display.waveplot(sound, sr=fs)
plt.show()


# In[ ]:



