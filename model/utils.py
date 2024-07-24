
import numpy as np
import torch
import matplotlib.pyplot as plt
from os.path import join as pjoin
import itertools
from sklearn.model_selection import train_test_split

def weights_init_normal(m):
	'''
		模型的初始化，也可以使用训练好pre-train的初始化
	'''
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)

def split_train_val(args, per_val=0.1, inter_n=16):
	'''
		**************************等间隔，按1:16划分训练数据和test数据，在训练数据中再随机划分测试数据*********
		*************************所以应该输出三个数据划分的index,train,val,test*****************************
		训练集-验证集-测试集都出自data/train/。
		'splits/section_train.txt' : x_193; i_1; i_54; x_116  其中x表示xline, i表示inline
	'''

	loader_type = 'section'
	labels = np.load(pjoin('data', args.dataset_name, 'train', 'train_labels.npy'))

	i_list = list(range(labels.shape[0]))
	i_list = ['i_'+str(inline) for inline in i_list]
	x_list = list(range(labels.shape[1]))
	x_list = ['x_'+str(crossline) for crossline in x_list]
	list_train_val_test = i_list + x_list

	# 间隔inter_n获取训练数据，例如间隔16道获取一个地震剖面，这符合地震解释过程。
	i_list_ = list(range(0, labels.shape[0], inter_n))
	i_list_ = ['i_'+str(inline) for inline in i_list_]
	x_list_ = list(range(0, labels.shape[1], inter_n))
	x_list_ = ['x_'+str(crossline) for crossline in x_list_]
	list_train_val = i_list_ + x_list_

	list_test = list(set(list_train_val_test)-set(list_train_val))

	# create train and test splits: 按比例分割数据集
	list_train, list_val = train_test_split(list_train_val, test_size=per_val, shuffle=True)

	# write to files to disK:
	file_object = open(pjoin('data',args.dataset_name, 'splits', loader_type + '_train_val_test.txt'), 'w')
	file_object.write('\n'.join(list_train_val_test))  # 1、全部数据的索引
	file_object.close()

	file_object = open(pjoin('data',args.dataset_name, 'splits', loader_type + '_train_val.txt'), 'w')
	file_object.write('\n'.join(list_train_val))       # 2、训练+验证数据索引，占全部数据的1/16
	file_object.close()

	file_object = open(pjoin('data', args.dataset_name, 'splits', loader_type + '_train.txt'), 'w')
	file_object.write('\n'.join(list_train))            # 3、训练数据索引
	file_object.close()

	file_object = open(pjoin('data', args.dataset_name, 'splits', loader_type + '_val.txt'), 'w')
	file_object.write('\n'.join(list_val))              # 4、验证数据索引
	file_object.close()

	file_object = open(pjoin('data', args.dataset_name, 'splits', loader_type + '_test.txt'), 'w')
	file_object.write('\n'.join(list_test))              # 5、测试数据索引，占全部数据的15/16
	file_object.close()

def save_train_sample(data, save_path):
	'''
		data: list , [img, label, pred]
	'''
	fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
	ax1.imshow(data[0])
	ax2.imshow(data[1])
	ax3.imshow(data[2])
	plt.savefig(save_path)
	plt.close()

def show_loss(model_path):
	'''
		显示训练误差和验证误差
	'''
	train_loss = np.load(model_path + 'train_loss.npy')
	val_loss = np.load(model_path + 'val_loss.npy')
	plt.plot(train_loss,label='train_loss')
	plt.plot(val_loss,label='val_loss')
	plt.xlabel('Epoch')
	plt.ylabel('Cross Entropy Loss')
	plt.legend()
	plt.show()

def G_Xdata():
	'''
		剪切数据集，原始数据集维度：（401，701，255）
	'''
	seismic = np.load(pjoin('data','train','train_seismic.npy'))
	labels = np.load(pjoin('data', 'train', 'train_labels.npy'))

	seismic = seismic[100:301, 200:501,:]
	labels = labels[100:301, 200:501,:]
	print(seismic.shape)
	print(labels.shape)

	np.save('train_seismic.npy',seismic)
	np.save('train_labels.npy',labels)


# show_loss('data/')


#----------------------------------------------------------------------------------不使用的函数-----------------
def split_train_val_(args, per_val=0.1):
	# create inline and crossline sections for training and validation:
	loader_type = 'section'
	labels = np.load(pjoin('data', 'train', 'train_labels.npy'))
	i_list = list(range(labels.shape[0]))
	i_list = ['i_'+str(inline) for inline in i_list]

	x_list = list(range(labels.shape[1]))
	x_list = ['x_'+str(crossline) for crossline in x_list]

	list_train_val = i_list + x_list
	# list_train_val = i_list  # 因为inline和xline大小不一致
	# create train and test splits:
	list_train, list_val = train_test_split(list_train_val, test_size=per_val, shuffle=True)

	# write to files to disK:
	file_object = open(pjoin('data', 'splits', loader_type + '_train_val.txt'), 'w')
	file_object.write('\n'.join(list_train_val))
	file_object.close()

	file_object = open(pjoin('data', 'splits', loader_type + '_train.txt'), 'w')
	file_object.write('\n'.join(list_train))
	file_object.close()

	file_object = open(pjoin('data', 'splits', loader_type + '_val.txt'), 'w')
	file_object.write('\n'.join(list_val))
	file_object.close()

def split_test_patch(args, data_name='test1'):
	'''
		建立inline核crossline的patches 为测试
		分割后的数据索引格式，i_288表示第288inline, (96,96)表示patch的起点索引
		i_288_96_96
		x_117_192_0	
	'''
	vert_stride = 52 # 垂向需要密集采样
	loader_type = 'section'
	data_file = data_name + '_labels.npy' 
	labels = np.load(pjoin('data', 'test_once', data_file))
	i_list = list(range(labels.shape[0]))
	i_list = ['i_'+str(inline) for inline in i_list]

	x_list = list(range(labels.shape[1]))
	x_list = ['x_'+str(crossline) for crossline in x_list]

	list_train_val = i_list + x_list

	# write to files to disK:
	file_object = open(pjoin('data', 'splits', loader_type + '_%s.txt'%data_name), 'w')
	file_object.write('\n'.join(list_train_val))
	file_object.close()
