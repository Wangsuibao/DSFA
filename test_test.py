
'''
这是的分割样式
	1、模型的输出是没有激活的(B,n_class,H,W)
	2、训练时使用损失函数是cross_entropy, 其包含了softmax激活和交叉熵损失
	3、label是分割结果（B,1,H,W）,数据类别是LongTensor

测试的目标是从训练集合中分离出的。 test占比15/16。测数据来直接存split的文件

'''

import pickle
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

from model import *
from dataset import *
from setting import *
from model.utils import *
from model.loss import *
from model.metrics import runningScore

import torch.nn as nn
import torch.nn.functional as F 
import torch

opt = args()
dataset_name = opt.dataset_name
os.makedirs("images/%s" % dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = False
device = 'cpu'


if opt.split_train_val:
	split_train_val(opt, per_val=opt.per_val)
	print('===new split data===')
else:
	print('===data split over===')


running_metrics_overall = runningScore(6)

end_epoch = 20
shiyan = opt.arch

# ---------------------------- 给定需要测试的，训练好的分割模型
test_name = 'generator_U_F_D_20.pth'

print('test_name: ', test_name)
model_path = 'save_train_models/%s/%s' % (dataset_name, test_name)
generator = torch.load(model_path)
generator.eval()

# 读取数据体和test对应的索引。 dataset_name
seismic_path = 'data/%s/train/train_seismic.npy'%dataset_name
label_path = 'data/%s/train/train_labels.npy'%dataset_name
test_file_path = 'data/%s/splits/section_test.txt'%dataset_name


seismic_data = np.load(seismic_path)
label_data = np.load(label_path)

file_list = tuple(open(test_file_path, 'r'))
file_list = [id_.rstrip() for id_ in file_list]  # ['x_61',...]

# file_list = file_list[0:20]


# 非整数输入到网络需要floattensor
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def save_test_fig(data, name, section, model_name=0, dataset_name='F3'):
	'''
		保存一个测试样例
	'''
	# px, py = width*dpi, height*dpi, dpi越高，图像素越高
	if dataset_name == 'F3':
		fig = plt.figure(figsize=(20, 10), dpi=100)  # (256, 512) == 10:20
	if dataset_name == 'NZPM':
		fig = plt.figure(figsize=(8, 15), dpi=100)   # (960, 512) == 15:8
	plt.imshow(data)
	plt.axis('off')  # 不要坐标轴
	plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)  # 不要四周空白
	plt.savefig('test_fig/%s_%s_%s.png' % (name, section, str(model_name)))
	plt.close()

def sample_images(file_list):
	'''
		1,保存validation数据集的生成结果，用于显示训练效果
		2,因为val_dataloader设置的batch是10，
		3,normalize=True将（0，255）的图片存储起来; normalize=False代表只能将（0，1）的图片存储起来
		4，nrow 默认分割第0维，(batch/nrow, nrow)的形式组合保存
	'''
	for section_name in file_list:
		direction, number = section_name.split(sep='_')
		if direction == 'i':
			im = seismic_data[int(number),:,:]
			lbl = label_data[int(number),:,:]
		elif direction == 'x':
			im = seismic_data[:,int(number),:]
			lbl = label_data[:,int(number),:]

		# 数据转化格式
		seismic_mean = 0.000941
		im -= seismic_mean
		im, lbl = im.T, lbl.T
		# pad(255,256)-->(256,256),上部加一行0
		im = np.pad(im,((1,0),(0,0)),'constant', constant_values=0.)
		lbl = np.pad(lbl,((1,0),(0,0)),'constant', constant_values=0)
		im, lbl = im[np.newaxis, np.newaxis,:,:], lbl[np.newaxis, np.newaxis,:,:]
		im = torch.from_numpy(im)
		im = im.float()
		lbl = torch.from_numpy(lbl)
		lbl = lbl.long()
		imgs, label = im, lbl
		# print(imgs.shape)

		real_A = imgs.type(Tensor)
		real_B = label.type(LongTensor)

		if shiyan in ['cs', 'Uu', 'GU', 'Res', 'z', 'SFA']:
			fake_B, _ = generator(real_A)
		else:
			fake_B = generator(real_A)
		fake_Bo = F.softmax(fake_B, dim=1).data.max(1)[1].unsqueeze(1) #---模型最后输出的是非激活（B,n_class,H,W）


		# 测试精度保存----------------------------------------------------
		pred = fake_B.detach().max(1)[1].numpy()  # (B,H,W)
		gt = real_B.numpy()                        # (B,1,H,W)
		running_metrics_overall.update(gt, pred)   # 累记混淆矩阵


		# 测试结果保存------------------------------------------------------------------------------------------
		img_sample = torch.cat((real_A.data, fake_Bo.data, real_B.data), -2)  # (B, 3, 3*H, W)，从下往上叠
		HH = img_sample.size()[2]//3

		# ---每张图片独立显示----
		datap = img_sample[0,0,HH:HH*2,:]
		# datap[datap<5]=0
		# datap[datap==5]=1
		save_test_fig(img_sample[0,0,0:HH,:], 'real_A', section_name, shiyan, dataset_name=dataset_name)
		save_test_fig(datap, 'fake_B', section_name, shiyan, dataset_name=dataset_name)
		save_test_fig(img_sample[0,0,HH*2:HH*3,:], 'real_B', section_name, shiyan, dataset_name=dataset_name)


	score = running_metrics_overall.get_scores()
	print(score)

	'''
	读取：
	with open("myDictionary.pkl", "rb") as tf:
	    new_dict = pickle.load(tf)	
	'''
	with open('result/test_Score/test_score_%s.pkl' % shiyan, 'wb') as tf:
		pickle.dump(score, tf)



if __name__ == '__main__':
	'''
		A（剖面） ---> B（分割）
		正常应该使用test数据集
	'''
	sample_images(file_list)



