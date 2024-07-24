'''
	收集模型重新训练，还是直接使用其论文中精度？
	1、Unet作为基础模型，原始Unet的基本特征是：
		(1)、3*3conv + BatchNorm + Relu  ; 两层卷积
		(2)、MaxPool2d(2)        channel: 64-1024  ; down: 2**4
		(3)、ConvTranspose2d(2)  channel: 1024-64  ; up  : 2**4
		(4)、cat((up,dw), dim=1)

'''

# -------------------------------------经典Unet------------------------------

class DoubleConv(nn.Module):
	'''
		双same卷积： 卷积+BN+Relu + 卷积+BN+Relu
					(in_c, out_c) - (out_c, out_c) ; 卷积块（3*3）； padding：1
		** 尺寸不变，channel增加2倍，是在第一个卷积层完成 **
	'''
	def __init__(self, in_ch, out_ch):
		super(DoubleConv,self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace = True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace = True)
			)

	def forward(self,input):
		return self.conv(input)


class Unet(nn.Module):
	'''
	****有skip链接的encoder-decoder结构，Unet。 没有地震参数的patch输出*****

		卷积:使用(C+B+R)*2
		下采样：使用MaxPool2d(kernel_size=2, stride=kernel_szie, padding=0)  当kernel=2时，表示下采样1/2
				             kernel_size:滑动窗口尺寸,可以是数组。
				             stride：窗口滑动步，默认等于kernel_size。
				             padding：边界填充,默认是0，不进行填充。
		上采样：使用ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True)
							 in_channels:  输入数据的通道数
							 out_channels: 输出数据的通道数
							 kernel_size:  卷积核的大小，可以是整数，可以是数组
							 stride：      卷积步长，本质上是对图像的0填充方式，如果stride=2则图像大小先增加2倍。
							 padding：     图像外边界填充
		上采样的不同类型：
			1、Upsample 上采样，直接插值，插值后接Conv
			2、ConvTranspose2d(in_, out_, kernel_size=2, stride=2)  # 图像大小增加2倍

	'''
	def __init__(self, in_ch=1, out_ch=6):
		super(Unet,self).__init__()

		# 1、encoder
		self.in_ch = in_ch
		self.out_ch = out_ch
		self.conv1 = DoubleConv(self.in_ch, 64)
		self.pool1 = nn.MaxPool2d(2)
		self.conv2 = DoubleConv(64, 128)
		self.pool2 = nn.MaxPool2d(2)
		self.conv3 = DoubleConv(128, 256)
		self.pool3 = nn.MaxPool2d(2)
		self.conv4 = DoubleConv(256, 512)
		self.pool4 = nn.MaxPool2d(2)
		self.conv5 = DoubleConv(512, 1024)

		# 3、decoder
		self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
		self.conv6 = DoubleConv(1024, 512)
		self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
		self.conv7 = DoubleConv(512, 256)
		self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
		self.conv8 = DoubleConv(256, 128)
		self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
		self.conv9 = DoubleConv(128, 64)

		# 4、地震相分割输出
		self.conv10 = nn.Conv2d(64, self.out_ch, 1)  # 类别拟合,1*1卷积

	def forward(self,x):
		'''
			网络数据传递的方式：
		(n,m)     1/2       1/4       1/8       1/16 @   1/8              1/4              1/2             (n,m)  
			------------------------2倍增------------->  /2    *2    /2    /2    *2    /2  /2    *2    /2   /2    *2    /2
		3   64    64  128   128 256   256 512   512 1024 512  1024   512  256   512   256  128   256   128  64    128   64   3         
			CC1-|-P1--CC2-|-P2--CC3-|-P3--CC4-|-P4--CC5--UP6--|cat|--CC6--UP7--|cat|--CC7--UP8--|cat|--CC8--UP9--|cat|--CC9--CC10--sigmoid()
						                      |---------------|cat|
								    |------------------------------------------|cat|
						  |---------------------------------------------------------------------|cat|
				|------------------------------------------------------------------------------------------------|cat|
		'''
		c1 = self.conv1(x)
		p1 = self.pool1(c1)
		c2 = self.conv2(p1)
		p2 = self.pool2(c2)
		c3 = self.conv3(p2)
		p3 = self.pool3(c3)
		c4 = self.conv4(p3)
		p4 = self.pool4(c4)
		c5 = self.conv5(p4)

		# 上采样--cat--卷积
		up6 = self.up6(c5)
		merge6 = torch.cat((up6,c4),dim=1)
		c6 = self.conv6(merge6)

		up7 = self.up7(c6)
		merge7 = torch.cat((up7,c3),dim=1)
		c7 = self.conv7(merge7)

		up8 = self.up8(c7)
		merge8 = torch.cat((up8,c2),dim=1)
		c8 = self.conv8(merge8)

		up9 = self.up9(c8)
		merge9 = torch.cat((up9,c1),dim=1)
		c9 = self.conv9(merge9)

		out = self.conv10(c9)  # 分割输出===============

		# 根据损失函数需要确定模型最后的输出是否需要激活

		return out