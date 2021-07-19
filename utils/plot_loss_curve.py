# coding: utf-8
import json

import matplotlib as mpl
import matplotlib.pyplot as plt

with open('../log/train_log.json', encoding='utf-8') as f:
    loss_data = json.load(f)

train_losses = loss_data['train_losses']

mpl.rc_file('../resource/style/matlab.mplstyle')
plt.plot(range(1, len(train_losses)+1), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss curve')
plt.show()
