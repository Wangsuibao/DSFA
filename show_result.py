'''
先使用测试集合测试，生成精度文件，test_score
显示 混淆矩阵。
'''

import pickle
import numpy as np
import pandas as pd
from os.path import join as pjoin
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from setting import args
args = args()


def read_score(path_file,score_name):
	with open(path_file+score_name, "rb") as tf:
		score = pickle.load(tf)
	return score

def show_CM(score):
	cm = score['confusion_matrix']
	cm_nor = cm/cm.sum(axis=1)
	# print(cm_nor)
	# class_names = ['upper_ns', 'middle_ns', 'lower_ns', 'rijnland_chalk', 'scruff', 'zechstein']  # F3
	class_names = ['basement', 'mud_A', 'mass_TD', 'mud_B', 'valley', 'submarine']  # NZPM
	df=pd.DataFrame(cm_nor, index=class_names, columns=class_names) 

	fig = plt.figure(figsize=(8, 6))
	sns.heatmap(df,annot=True, linewidths=1, annot_kws={'size':12})
	# sns.heatmap(df, annot=True, linewidths=1)
	# plt.xlabel('Predicted seismic face')
	# plt.ylabel('True seismic face')
	# plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=1,hspace=None)
	plt.tight_layout()
	plt.savefig('result/CM_%s.png'%model_name)
	plt.close()

def show_jingdu(path_file):
	'''
		----测试结果的精度一起显示----
	'Pixel Acc: ': acc,                                  # 总正确率，flot
	'Class Accuracy: ': acc_cls,                         # 每个类别的正确率, ndarray(n_class,)
	'Mean Class Acc: ': mean_acc_cls,                    # 平均正确率, flot
	'Freq Weighted IoU: ': fwavacc,                      # 带权重的每个类别IOU,ndarray(n_class,)
	'Mean IoU: ': mean_iu,                               # 平均IOU
	'''
	PA = []
	MCA = []
	MIOU = []
	FWIOU = []
	AC = []
	# model_name = ['cs', 'GU', 'Res', 'Uu']
	model_name = ['GU', 'Res', 'cs', 'U']
	for i in model_name:
		score_name = 'test_score_%s.pkl' % str(i)
		score = read_score(path_file, score_name)[0]

		# print(score)
		PA.append(score['Pixel Acc: '])
		MCA.append(score['Mean Class Acc: '])
		MIOU.append(score['Mean IoU: '])
		FWIOU.append(score['Freq Weighted IoU: '])
		AC.append(score['Class Accuracy: '])
	print('model name list: ', model_name)
	print('PA: ', PA)
	print('MCA: ',MCA)
	print('MIOU: ', MIOU)
	print('FWIOU: ', FWIOU)
	print('AC: ', AC)

def show_loss(model_path, name=''):
	'''
		显示训练误差和验证误差。
	'''
	train_loss = np.load(model_path + 'train_loss_%s.npy'%name)
	min_loss = train_loss[-10:].mean()
	# val_loss = np.load(model_path + 'val_loss%s.npy'%name)
	plt.plot(train_loss,label='train_loss')
	# plt.plot(val_loss,label='val_loss')
	plt.xlabel('Epoch')
	plt.ylabel('Cross Entropy Loss')
	plt.title('the train loss min: %f%s'%(min_loss, name))
	plt.legend()
	plt.savefig('result/train_loss_%s.png'%name)
	plt.close()


model_name = args.arch  # ['cs', 'Uu', 'GU', 'Res', 'z',   'UrS', 'U', 'UaF', 'SFA']	
path_file = 'result/test_Score/'
score_name = 'test_score_%s.pkl' % str(model_name)


# 显示混淆矩阵---------------------------------------
score = read_score(path_file, score_name)
# print(score)
score = score[0]
show_CM(score)

# 一起打印多个函数的精度------------------------------
show_jingdu(path_file)


# 训练误差图------------------------------------------
# show_loss('result/','SFA')  # 'UrS', 'U', 'UaF', 'SFA'

