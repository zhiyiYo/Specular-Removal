# coding:utf-8
import matplotlib as mpl
import matplotlib.pyplot as plt
from specular_removal import TrainPipeline


train_config = {
    'lr': 0.01,
    'epochs': 10,
    'test_freq': 4,
    'step_size': 4,
    'train_dataset_dir': 'data/SHIQ_data/train',
    'test_dataset_dir': 'data/SHIQ_data/test',
    'train_batch_size': 24,
    'test_batch_size': 24,
    'model_dir': 'model',
    'use_gpu': True
}
train_pipeline = TrainPipeline(**train_config)
train_losses, test_losses = train_pipeline.train()
train_pipeline.save()

mpl.rc_file('resource/style/matlab.mplstyle')
plt.plot(range(1, train_config["epochs"] + 1), train_losses)
plt.title('Loss curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()