'''
	1、模型的输出是没有激活的(B,n_class,H,W)
	2、训练时使用损失函数是cross_entropy, 其包含了softmax激活和交叉熵损失
	3、label(real_B) 是分割结果（B,1,H,W）,数据类别是LongTensor
	4、生成器生成的fake_B(B,n_class,H,W) 和 对应的real_A（B,1,H,W） 需要叠加输入判别器中，
	   所以需要提前处理fake_B成（B,1,H,w）成分割类别形式

因为训练判别器时，是隔代获取数据，对于生成模型和判别模型。--------------所以只能等大小。*************
'''

import os
import numpy as np 
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets

from model.modelG import *
from model.utils import *
from model.loss import *

from dataset import *
from setting import *

import torch.nn as nn
import torch.nn.functional as F 
import torch

opt = args()
dataset_name = opt.dataset_name
os.makedirs("images/%s" % dataset_name, exist_ok=True)
os.makedirs("save_train_models/%s" % dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = False
device = 'cpu'
weight = torch.tensor(np.array([2,1,3,2,2,1])[np.newaxis,:,np.newaxis])  # 类间权

# 如果已经有分割，则不需要再次分割，不然无法进行对比实验。
if opt.split_train_val:
	split_train_val(opt, per_val=opt.per_val)
	print('===new split data===')
else:
	print('===data split over===')


# ----------------------------------------------------------------------------------
criterion_GAN = torch.nn.MSELoss()  # patch的均方根子块真假误差

# 像素损失
# criterion_pixelwise = torch.nn.L1Loss()  # 像素拟合的误差
# criterion_pixelwise = cross_entropy      # 交叉熵
criterion_pixelwise = Focal_Loss()         # 这是通过train_U筛选出来的最优损失函数，
# criterion_pixelwise = DiceLoss()         # 
# criterion_pixelwise = Focal_Loss_z(weight)
#-----------------------------------------------------------------------------------

patch = (1, opt.img_height//2**4, opt.img_width//2**4)  # 计算判别器的输出 (1,16,32)

generator = GeneratorUNet(in_channels=opt.channels, out_channels=6)  # 输出6个类别的非归一化概率
discriminator = Discriminator(in_channels=1)

if cuda:
	generator = generator.cuda()
	discriminator = discriminator.cuda()
	criterion_GAN.cuda()
	criterion_pixelwise.cuda()

# -------------------------------------------------初始化模型
if opt.epoch != 0:
	generator.load_state_dict(torch.load('saved_models/%s/generator_%d.pth' % (dataset_name, opt.epoch)))
	discriminator.load_state_dict(torch.load('saved_models/%s/discriminator_%d.pth' % (dataset_name, opt.epoch)))
else:
	generator.apply(weights_init_normal)
	discriminator.apply(weights_init_normal)


# 需要设计学习率下降的样式
optimizers_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1,opt.b2))
optimizers_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


# --------------------------------------------------准备数据; 
if opt.aug:
	data_aug = Compose([RandomRotate(10), RandomHorizonTallyFlip(), AddNiose()])
else:
	data_aug = None

train_set = section_loader(split='train', dataset_name=dataset_name, is_transform=True, augmentations=data_aug)
val_set = section_loader(split='val', dataset_name=dataset_name, is_transform=True,)

shuffle = False  # 字定义sampler,shuffle必须为False
with open(pjoin('data', dataset_name, 'splits', 'section_train.txt'), 'r') as f:
	train_list = f.read().splitlines()
with open(pjoin('data', dataset_name, 'splits', 'section_val.txt'), 'r') as f:
	val_list = f.read().splitlines()

# 一个batch中不能有inline和xline
class CustomSamplerTrain(torch.utils.data.Sampler):
	def __iter__(self):
		# 1、先选择剖面
		char = ['i' if np.random.randint(2) == 1 else 'x']
		# 2、再获取所有对应剖面的索引，随机排列
		self.indices = [idx for (idx, name) in enumerate(train_list) if char[0] in name]
		return (self.indices[i] for i in torch.randperm(len(self.indices)))
	def __len__(self):
		return len(train_list)

class CustomSamplerVal(torch.utils.data.Sampler):
	def __iter__(self):
		char = ['i' if np.random.randint(2) == 1 else 'x']
		self.indices = [idx for (idx, name) in enumerate(val_list) if char[0] in name]
		return (self.indices[i] for i in torch.randperm(len(self.indices)))
	def __len__(self):
		return len(val_list)

trainloader = data.DataLoader(train_set, batch_size=opt.batch_size, num_workers=0,shuffle=shuffle, 
	                          sampler=CustomSamplerTrain(train_list),drop_last=True)
valloader = data.DataLoader(val_set, batch_size=opt.batch_size, num_workers=0,shuffle=shuffle,
	                          sampler=CustomSamplerVal(val_list),drop_last=True)

# 非整数输入到网络需要floattensor
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_images(batches_done):
	'''
		1,保存validation数据集的生成结果，用于显示训练效果
		2,因为val_dataloader设置的batch是10，
		3,normalize=True将（0，255）的图片存储起来; normalize=False代表只能将（0，1）的图片存储起来
		4，nrow 默认分割第0维，(batch/nrow, nrow)的形式组合保存
	'''
	imgs, label = next(iter(valloader))
	real_A = imgs.type(Tensor)
	# real_B = label.type(Tensor) # -----------******
	real_B = label.type(LongTensor)
	fake_B = generator(real_A)
	fake_B = F.softmax(fake_B, dim=1).data.max(1)[1].unsqueeze(1) #-----------******
	img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)  # (B, 3, 3*H, W)，从下往上叠
	save_image(img_sample, "images/%s/%s.png" % (dataset_name, batches_done), nrow=5, normalize=True)



if __name__ == '__main__':
	'''
		A（剖面） ---> B（分割）
	'''
	G_loss = []
	D_loss = []

	prev_time = time.time()
	for epoch in range(opt.epoch, opt.n_epochs):
		for i, (images, labels) in enumerate(trainloader):
			images_original, labels_original = images, labels
			real_A = images.type(Tensor)
			real_B = labels.type(LongTensor)
			# real_B = labels.type(Tensor) # ----------********
			# print(real_A.shape)  # (2,1,256,301); (2,1,256,201)两种剖面
			valid = Tensor(np.ones((real_A.size(0), *patch)))  # (B,1,PH,PW)
			fake = Tensor(np.zeros((real_A.size(0), *patch)))

			'''
				训练的五步：
					1、前向传播
					2、计算损失  ； 计算损失前需要完成前向传播
					3、梯度清零  ； 可以放到第一步
					4、计算梯度  ； 计算梯度前，原来梯度需要清零
					5、更新参数
			'''

			# 前向传播
			fake_B = generator(real_A) #====前向传播没有随机z的输入，随机的引用使用dropout====？？
			fake_B_1 = F.softmax(fake_B, dim=1).data.max(1)[1].unsqueeze(1)  # ------********
			# pred_fake = discriminator(fake_B, real_A)
			pred_fake = discriminator(fake_B_1, real_A)


			#-------------------------
			#    Train G
			#-------------------------

			# 计算损失
			loss_GAN = criterion_GAN(pred_fake, valid)       # 希望判定为真
			loss_pixel = criterion_pixelwise(fake_B, real_B) # 像素损失； 
			# 这里不用detch的原因是损失是从网络的末端传来的G-D整个图都存在，只有梯度传到G，才能更新G的参数
			loss_G = loss_GAN + opt.lambda_pixel*loss_pixel

			G_loss.append(loss_G.item())
			optimizers_G.zero_grad()
			loss_G.backward()
			optimizers_G.step()


			# -------------------------------
			#    Train D
			#--------------------------------

			pred_real = discriminator(real_B, real_A)           # y真+c真
			loss_real = criterion_GAN(pred_real, valid)         # 应该判为真

			# pred_fake = discriminator(fake_B.detach(), real_A)  # y生+c真
			pred_fake = discriminator(fake_B_1.detach(), real_A)  # y生+c真
			# pred_fake = discriminator(fake_B, real_A) 
			# 报错的原因是因为训练G时构建的G-D图经过backward已经销毁，fake_B来自G，梯度回传找不到G的计算图，所以需要detch和生成它的图脱钩。
			loss_fake = criterion_GAN(pred_fake, fake)          # 应判定为假
			# ------------有没有别的组合
			loss_D = 0.5 * (loss_real +loss_fake)

			D_loss.append(loss_D.item())
			optimizers_D.zero_grad()
			loss_D.backward()
			optimizers_D.step()

			# ----------------------------------------------------------------------------------------------------------------
			print("Epoch=%d; batch=%d; pix_L=%f; G_L=%f; D_L=%f" % (epoch,i,loss_pixel.item(),loss_G.item(), loss_D.item()))
			# 写入文件方便成图

			batches_done = epoch * len(trainloader) + i
			if batches_done % opt.sample_interval == 0:
				sample_images(batches_done)

		if opt.save_model_interval != -1 and epoch % opt.save_model_interval == 0:
			torch.save(generator, "save_train_models/%s/generator_U_F_D_%d.pth" % (dataset_name, epoch))
			torch.save(discriminator, "save_train_models/%s/discriminator_U_F_D_%d.pth" % (dataset_name, epoch))

	np.save('result/D_loss.npy', np.array(D_loss))
	np.save('result/G_loss.npy', np.array(G_loss))

'''
问题：
	1、GAN训练时，detch, item 的使用
		答案： 原始训练GAN的方式是先训练判别器，再训练生成器。但先训练生成器，再训练判别器可能更方便
			在训练G时，梯度需要传过D-G，然后更新G的参数，因为使用一次backward,计算图会销毁。
			再训练D时，梯度只需传过D，但当使用G生成的fake_B时用到计算图G，而G已经被销毁，所以会出错，这里使用fake_B.detch()完成“脱钩”
		答案：item是直接从计算图中copy个tensor出来

	2、判别损失使用均方差的形式是否等价交叉熵; WGAN和这种的区别
		答案：
'''




