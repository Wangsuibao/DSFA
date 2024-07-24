'''
数据导入. 读取的数据体大小是（512, 512, 256）, 训练数据使用的是剖面，所以大小是（512,256）
数据有缺失值， data为255， label也为255---------------------是pad导致，删除--
正常数据：data[-1,1]; label[0-nclass)

数据处理方式说明
	1、F3数据集，读取的数据体大小是（512, 512, 255）, 训练数据使用的是剖面。所以大小是（512,256）
		数据标签是 0-5
		2D数据的大小是（512,256）

	2、NZPM数据集，原始数据大小是（590,782,1006）， 截取数据大小为（512,512,959） 可以满足5次下采样
		数据标签是 1-6
		2D数据的大小是（512,960）

	数据全部保存成npy
	**** 计算损失时，类别默认是从0开始，所以NZPM数据的label需要处理成0-5
	**** 统一在在高度缺少1处理, 因为F3原始数据缺1，所以NZPM数据也处理成缺1
'''

import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils import data


class section_loader(data.Dataset):
	'''
		section 级别的地震剖面相分类
		输出数据的维度是：振幅的值data:(channel,H,W)浮点数, 属于那个类别labels:(1,H,W)-0-n_class的整数
		
		# data 的类型, data:  tensor.Float ---(1,H,W)
		# label的类型，label: tensor.Long ----(1,H,W)
	'''
	def __init__(self, split='train', dataset_name='F3', is_transform=True, augmentations=None):
		'''
			地震数据体和label体是读取到内存中
			
		'''
		# self.root = 'data/F3/'
		self.root = 'data/%s/'%dataset_name
		self.split = split    # 数据集的选择 
		self.is_transform = is_transform
		self.augmentations = augmentations
		self.n_classes = 6    # 分割类别数量
		self.mean = 0.000941  # 训练数据的平均值
		self.sections = collections.defaultdict(list)

		if 'test' not in self.split:
			# train and val 都使用这个数据
			self.seismic = np.load(pjoin(self.root,'train','train_seismic.npy'))
			self.labels = np.load(pjoin(self.root, 'train', 'train_labels.npy'))
		elif 'test' in self.split:
			# 测试数据使用单独的test地震数据。
			self.seismic = np.load(pjoin(self.root, 'test_once', '%s_seismic.npy'%split))
			self.labels = np.load(pjoin(self.root, 'test_once', '%s_labels.npy'%split))
		else: 
			raise ValueError('Unknown split.')

		# --------------------------------------------------
		'''
			根据数据索引从大的数据集中获取数据--
			inline:i, xline:x, 左上角索引
			x_60    ： 第60 xline 剖面上
			i_321   ： 第321 inline 剖面上
			self.sections :  {'train':[], 'train_val':[], 'val':[]}的格式
		'''
		if 'test' not in self.split:
			for split in ['train', 'val', 'train_val']:
				path = pjoin(self.root, 'splits', 'section_' + split + '.txt')
				file_list = tuple(open(path, 'r'))
				file_list = [id_.rstrip() for id_ in file_list]
				self.sections[split] = file_list

		elif 'test' in self.split:
			path = pjoin(self.root, 'splits', 'section_' + split + '.txt')
			patch_list = tuple(open(path, 'r'))
			file_list = [id_.rstrip() for id_ in file_list]
			self.sections[split] = file_list
		else:
			raise ValueError('Unknown split')

	def __len__(self):
		return len(self.sections[self.split])

	def __getitem__(self, index):
		'''
			首先是随机获取索引，然后根据索引获取对应的剖面名称，然后根据剖面名称从地震体数据中获取剖面数据。
			获取剖面数据后，对数据剖面数据进行处理，归一化-转置-增加维度-转换成tensor。
		'''
		section_name = self.sections[self.split][index]  # 一个section的具体位置字符串
		direction, number = section_name.split(sep='_')  # x\i, 剖面位置

		if direction == 'i':
			im = self.seismic[int(number),:,:]
			lbl = self.labels[int(number),:,:]
		elif direction == 'x':
			im = self.seismic[:,int(number),:]
			lbl = self.labels[:,int(number),:]

		# 数据增强和变化
		if self.augmentations is not None:
			im, lbl = self.augmentations(im, lbl)

		if self.is_transform:
			# 1、减均值； 2、转置； 3、增加channel维度； 4、toTensor
			im, lbl = self.transform(im, lbl)

		# （1，H，W）, (1,H,W)
		return im, lbl

	def transform(self, im, lbl): 
		'''
			问题1： 为什么要减去均值
			1、减均值； 2、转置； 3、增加channel维度； 4、toTensor
			2、转置(512, 255)--> (255, 512)
		'''
		im -= self.mean
		im, lbl = im.T, lbl.T
		im = np.pad(im,((1,0),(0,0)),'constant',constant_values=0.)
		lbl = np.pad(lbl,((1,0),(0,0)),'constant',constant_values=0)
		im, lbl = im[np.newaxis,:,:], lbl[np.newaxis,:,:]
		im = torch.from_numpy(im)
		im = im.float()
		lbl = torch.from_numpy(lbl)
		lbl = lbl.long()

		return im, lbl

	def get_seismic_labels(self):
		'''
			label对应的颜色值（RGB）
		'''
		return np.asarray([ [69,117,180], [145,191,219], [224,243,248], 
			                [254,224,144], [252,141,89], [215,48,39]]) 

	def decode_segmap(self, label_mask, plot=False):
		'''
			模型输出变成彩色图
			label_mask : ndarray, (M,N), int, 每个空间位置的类别
			return： ndarray, 
		'''
		label_colours = self.get_seismic_labels()
		r = label_mask.copy()
		g = label_mask.copy()
		b = label_mask.copy()
		for l1 in range(0, self.n_classes):
			r[label_mask == l1] = label_colours[l1, 0]
			g[label_mask == l1] = label_colours[l1, 1]
			b[label_mask == l1] = label_colours[l1, 2]

		rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
		rgb[:,:,0] = r/255.0
		rgb[:,:,1] = g/255.0
		rgb[:,:,2] = b/255.0

		if plot:
			plt.imshow(rgb)
			plt.show()
		else:
			return rgb



if __name__ == '__main__':
	import matplotlib.pyplot as plt
	dataset = section_loader(split='train')

	a = dataset.seismic
	b = dataset.labels
	print(a.shape, b.shape)
	# 这样获取的数据是旋转90度的，需要转置T
	plt.imshow(a[0,:,:])
	plt.show()

	plt.imshow(b[0,:,:])
	plt.show()

	dataset.decode_segmap(b[0,:,:], True)


	# for i in range(10000):
	# 	a,b = dataset.__getitem__(i)
	# 	print(a.shape,b.shape)
	a,b = dataset.__getitem__(10)
	plt.imshow(a[0,:,:])
	plt.show()
	plt.imshow(b[0,:,:])
	plt.show()


