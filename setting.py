import numpy as np
import torch

class args():
	def __init__(self):
		self.n_classes = 6

		self.pretrained = False        # 是否支持预训练模型
		self.aug = False               # 是否支持数据增强； 增强有很多种，但是不是所有的都合理(旋转+噪音..)
		self.class_weights = False     # 是否使用权重来平衡 类的不平衡

		self.dataset_name = 'F3'       # 数据的名字,data/train/时为空, F3， NZPM
		self.img_height = 256          # F3=256, NZPM=960
		self.img_width = 512	       # F3=512， NZPM=512
		self.split_train_val = True    # 第一次使用True, 以后的测试，因为有第一次的生成索引，所以设定为False

		self.channel = 1
		self.clip = 0.1               # 梯度剪切的范围，如果是0表示不使用

		self.section_inter = 8
		self.channels = 1             # 图像的通道数 
		self.lambda_pixel = 101       # 像素损失权重，这个值太大，L1损失是否压制了判别损失。 101都能到100epoch

		self.per_val = 0.1     # 训练集中，用于验证的数量	
		self.lr = 0.0002
		self.b1 = 0.5          # adam 的参数
		self.b2 = 0.999        # adam 的参数
		self.batch_size = 2

		self.resume = None     # 从新启动模型的路径,checkpoint的路径

		self.save_interval = 100        # 保存训练结果的batch间隔,xxxx舍弃xxxxx
		self.save_interval_val = 2      # 保存测试结果的batch间隔,xxxxs舍弃xxxx
		self.val_epoch_interval = 1     # 间隔1 epoch测试一次，xxxxxxxx舍弃xxxx


		self.arch = 'U'                 # 训练结构的选择 
		self.epoch = 0                  # 默认开始训练的epoch
		self.n_epochs = 100   		    # 50或100
		self.save_model_interval = 20   # 保存模型的间隔,10
		self.sample_interval = 400      # 训练时保存的图片的间隔-400
