'''
pix2pix GAN模型 = Unet + patchGAN

图像的输入格式是，不定大小的（B,1,H,W），有个要求是，尺寸可以整数下采样4次。

归一化层和dropout层不在网络的开头和最后使用
'''

import torch.nn as nn
import torch.nn.functional as F 
import torch

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

#----------------------
#  G : U-net
#-----------------------

class UNetDown(nn.Module):
	'''
		conv-BN-drop-ReLU; 4*4filter; stride=2; pad=1
		1: drop的目标是加入随机性，替换生成网络中输入的随机向量z.
		2: filter=4*4，stride=2, pad=1 的下采样1/2卷积
	'''
	def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
		super(UNetDown, self).__init__()
		# (in_channels, out_channels, kernel_size, stride, padding, dilation(空洞卷积), groups, bias)
		layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
		if normalize:
			layers.append(nn.InstanceNorm2d(out_size))
		layers.append(nn.LeakyReLU(0.2))

		if dropout:
			layers.append(nn.Dropout(dropout))
		self.model = nn.Sequential(*layers)

	def forward(self, x):
		return self.model(x)

class UNetUp(nn.Module):
	'''
		对传递方向先卷积，后直接cat跳跃链接的down信息
	'''
	def __init__(self, in_size, out_size, dropout=0.0):
		super(UNetUp, self).__init__()
		layers = [
			nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),  # 反卷积-先pad再卷积
			nn.InstanceNorm2d(out_size),
			nn.ReLU(inplace=True)]
		if dropout:
			layers.append(nn.Dropout(dropout))
		self.model = nn.Sequential(*layers)

	def forward(self, x, skip_input):
		'''
			先上采样卷积，再cat
		'''
		x = self.model(x)
		x = torch.cat((x, skip_input), 1)  # 以cat的形式，把down的对应输出在channel上叠加
		return x

class GeneratorUNet(nn.Module):
	def __init__(self, in_channels=1, out_channels=6):
		# 生成一个分类结果，所以这里outchannel=n_class
		super(GeneratorUNet, self).__init__()
		self.down1 = UNetDown(in_channels, 64, normalize=False)
		self.down2 = UNetDown(64, 128, dropout=0.5)
		self.down3 = UNetDown(128, 256, dropout=0.5)
		self.down4 = UNetDown(256, 512, normalize=False, dropout=0.5)  # 不用BN

		self.up1 = UNetUp(512, 256, dropout=0.5)               # 有跳跃传递的256
		self.up2 = UNetUp(512, 128, dropout=0.5)               # 跳跃输入128
		self.up3 = UNetUp(256, 64)                             # 跳跃输入64

		# 地震分割模型的输出
		self.final = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.ZeroPad2d((1,0,1,0)),
			nn.Conv2d(128, 64, 4, padding=1),
			nn.Conv2d(64, out_channels, 1)
			# 生成的结果是类别，，分布在0-6之间，所以输出非激活形式（使用softmax激活）。（直接relu拟合类别也可以）
			)

	def forward(self, x):
		'''
			下采样8次，上采样8次，则输入是否需要是64的倍数。如(640,256)或
		'''
		d1 = self.down1(x)   # 64
		d2 = self.down2(d1)  # 128 
		d3 = self.down3(d2)  # 256
		d4 = self.down4(d3)  # 512

		# print()
		u1 = self.up1(d4, d3)  # 256u + 256d
		u2 = self.up2(u1, d2)  # 128u + 128d
		u3 = self.up3(u2, d1)  # 64u + 64d
		# u7的输出channels=128

		return self.final(u3)


# -------------------------------------------------------
#   D - PatchGAN
# -------------------------------------------------------
class UNetUp_noskip(nn.Module):
	'''
		对传递方向先卷积，后直接cat跳跃链接的down信息
	'''
	def __init__(self, in_size, out_size, dropout=0.0):
		super(UNetUp_noskip, self).__init__()
		layers = [
			nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),  # 反卷积-先pad再卷积
			nn.InstanceNorm2d(out_size),
			nn.ReLU(inplace=True)]
		if dropout:
			layers.append(nn.Dropout(dropout))
		self.model = nn.Sequential(*layers)

	def forward(self, x):
		'''
			先上采样卷积，再cat
		'''
		x = self.model(x)
		return x

class GeneratorUNet_noskip(nn.Module):
	def __init__(self, in_channels=1, out_channels=6):
		# 生成一个分类结果，所以这里outchannel=n_class
		super(GeneratorUNet_noskip, self).__init__()
		self.down1 = UNetDown(in_channels, 64, normalize=False)
		self.down2 = UNetDown(64, 128, dropout=0.5)
		self.down3 = UNetDown(128, 256, dropout=0.5)
		self.down4 = UNetDown(256, 512, normalize=False, dropout=0.5)  # 不用BN

		self.up1 = UNetUp_noskip(512, 256, dropout=0.5)               # 有跳跃传递的256
		self.up2 = UNetUp_noskip(256, 128, dropout=0.5)               # 跳跃输入128
		self.up3 = UNetUp_noskip(128, 64)                             # 跳跃输入64

		# 地震分割模型的输出
		self.final = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.ZeroPad2d((1,0,1,0)),
			nn.Conv2d(64, 64, 4, padding=1),
			nn.Conv2d(64, out_channels, 1)
			# 生成的结果是类别，，分布在0-6之间，所以输出非激活形式（使用softmax激活）。（直接relu拟合类别也可以）
			)

	def forward(self, x):
		'''
			下采样8次，上采样8次，则输入是否需要是64的倍数。如(640,256)或
		'''
		d1 = self.down1(x)   # 64
		d2 = self.down2(d1)  # 128 
		d3 = self.down3(d2)  # 256
		d4 = self.down4(d3)  # 512

		u1 = self.up1(d4)  #
		u2 = self.up2(u1)  #
		u3 = self.up3(u2)  #
		# u7的输出channels=128

		return self.final(u3)

# -------------------------------------------------------
#   D - PatchGAN
# -------------------------------------------------------

class Discriminator(nn.Module):
	def __init__(self, in_channels=3):
		super(Discriminator, self).__init__()

		def discriminator_block(in_filters, out_filters, normalization=True):
			'''
				返回每个下采样的D-block：[C_B_R]
			'''
			layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
			if normalization:
				layers.append(nn.InstanceNorm2d(out_filters))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		self.model = nn.Sequential(
			*discriminator_block(in_channels*2, 64, normalization=False),
			*discriminator_block(64, 128),
			*discriminator_block(128, 256),
			*discriminator_block(256, 512),
			nn.ZeroPad2d((1,0,1,0)),   # 2D图的（左右上下）填充个数
			nn.Conv2d(512, 1, 4, padding=1, bias=False))  # 判别器拟合的是推土机距离，所以不使用激活函数

	def forward(self, img_A, img_B):
		'''
			条件图和生成图cat后输入D
		'''
		img_input = torch.cat((img_A, img_B), 1)
		return self.model(img_input)



# if __name__ == '__main__':
# 	from torchsummary import summary
# 	G = GeneratorUNet()
# 	summary(G, (1,512,512))