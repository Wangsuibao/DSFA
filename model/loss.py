'''
计算损失函数
	分割损失（地震相分割损失）
'''

import torch
import torch.nn.functional as F 
import torch.nn as nn
import numpy as np

def cross_entropy(input, target, weight=None, ignore_index=255):
	'''
		1、cross_entropy:首先实施softmax, 然后再计算不同类别熵， 所以网络的最后输出不能使用--激活函数
		2、input: 模型输出（B,n_class,H,W）; float
		3、target: labels (B,1, H,W) ;      Long

		#-------------------------------------
		cross_entropy(input, target)
		input: (B，n_class, H，W)
		target: (B，H，W)

		#-------------------------------------
		原始数据有无效值，data中为255， label中也为255，如何避免这种情况对训练的影响
	''' 

	target = torch.squeeze(target, dim=1)  # (B，1，H，W)-->(B，H，W)
	# ignore_index: 
	loss = F.cross_entropy(input, target, weight, reduction='sum', ignore_index=255)
	return loss


#---------------------------------------------------------------------------------
# 包含所有图像分割的损失函数
#----------------------------------------------------------------------------------
def make_one_hot(labels, classes):
	'''
		* make_one_hot(target.unique(dim=1), classes=output.size()[1])  编码好像不对--？？

		1、torch.FloatTensor(), 如果里面传入数组，则直接处理成张量，如果传入的是独立的数字，则会处理成张量的维度。
		2、torch.tensor(), 不能传入独立的数字，必须传入数组的形式
		3、scatter_(dim, index, src), 其中index必须是Long型，
		4、输入的labels的维度是（B,h,w）, 输出的target的维度是（B,n,h,w）
		** 完成对label的one-hot 编码
	'''
	one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
	target = one_hot.scatter_(1, labels.data, 1)

	return target


# 1、Dice Loss
class DiceLoss(nn.Module):
	'''
			2*(AB)交集(点乘)  + 1（拉普拉斯平滑）
		1 - ————————————————————————————————————
			  |A| + |B| (元素个数) + 1（拉普拉斯平滑）
	'''
	def __init__(self, smooth=1, ignore_index=255):
		super(DiceLoss, self).__init__()
		self.ignore_index = ignore_index  # 有些缺失值，无效值，被赋予255
		self.smooth = smooth

	def forward(self, output, target):
		'''
			output: (B，n_class, H,W)  # 每个channel一个类别的概率
			target: (B，H，W)  # 类别的形式是0-n_class的整数
		'''
		if self.ignore_index not in range(target.min(), target.max()):
			if (target == self.ignore_index).sum() > 0:
				# 说明有无效值
				target[target==self.ignore_index] = target.min()

		target = make_one_hot(target.unique(dim=1), classes=output.size()[1])
		output = F.softmax(output, dim=1)

		output_flat = output.contiguous().view(-1)
		target_flat = target.contiguous().view(-1)

		intersection = (output_flat*target_flat).sum()  # 交集统计
		loss = 1-(2.*intersection + self.smooth) / (output_flat.sum() + target_flat.sum() + self.smooth)

		return loss


# focal loss
class Focal_Loss(nn.Module):
	'''
		借助交叉熵求解focal loss

		loss = Focal_Loss()
		input = torch.randn(2,3,5, 5, requires_grad=True)         # （B,c,H,W）
		target = torch.empty(2,5,5, dtype=torch.long).random_(3)  #  (B，H，W) 整数
		output = loss(input, target)
		print(output)
	'''
	def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
		super(Focal_Loss, self).__init__()
		self.gamma = gamma
		self.size_average = size_average

		self.CE_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index, weight=alpha)

	def forward(self, output, target):
		'''
		'''
		target = torch.squeeze(target, dim=1)  # (B，1，H，W)-->(B，H，W)
		logpt = self.CE_loss(output, target)    # 1、 -1*log(p)*y;  (B，H，W), 先做了channel上的求和
		pt = torch.exp(-logpt)
		loss = ((1-pt)**self.gamma) * logpt     # 2、（1-p）**gamma * [-1*log(p)*y]

		if self.size_average:
			return loss.mean()
		return loss.sum()


class Focal_Loss_z(nn.Module):
	'''
		实现Focal loss，问题是会出现nan在计算log时
		1、log(p)*y * -1
		2、（1-p）**gamma * log(p)*y * -1
		3、Weight*（1-p）**gamma * log(p)*y * -1
	'''
	def __init__(self, weight, gamma=2):
		super(Focal_Loss_z,self).__init__()
		self.gamma = gamma    # 一般是2
		self.weight = weight  # 可以是每个像素的权重，也可以是每个类别的权重，数据的维度是（1,C,1）

	def forward(self, preds, labels):
		"""
		preds  :  softmax输出结果; tensor.softmax(dim=1), 维度是（B,C,H,W）
		labels :  真实值,维度是（B,C,H,W），需要onehot 编码
		"""
		eps=1e-7
		labels = make_one_hot(labels,6)
		preds = F.softmax(preds,dim=1)

		y_pred = preds.view((preds.size()[0],preds.size()[1],-1))  # (B,C,H,W)->(B,C,H*W)
		target = labels.view(y_pred.size())  # (B,C,H,W)->(B,C,H*W)

		# y_pred 是softmax后的概率
		# target 是one_hot 编码后label
		ce = -1.*torch.log(y_pred+eps)*target        # 交叉熵权重, 这一步产生nan---------------
		floss = torch.pow((1-y_pred), self.gamma)*ce  # 交叉熵基础上增加难易系数（1-p)**gamma
		floss = torch.mul(floss, self.weight)         # 在channel上乘权重weight

		# 不同类别（C上）间求和
		floss = torch.nansum(floss, dim=1)
		# 不同样本（B和像素上）间求均值
		floss = torch.nanmean(floss)
		return floss