''' 

=============================================================================================== 
Discriminator-based Stratigraphic Sequence Semantic Augmentation Seismic Facies Analysis

Sui-Bao Wang
a. School of Earth Sciences, Northeast Petroleum University, Daqing 163318, China;
b. National Key Laboratory of Continental Shale Oil, Northeast Petroleum University, Daqing, Heilongjiang 163318, China

E-mail: wangsuibao@stu.nepu.edu.cn;


===============================================================================================
Module introduction：
	1、train_U.py test object:
		U_loss: ['cross_entropy', 'DiceLoss', 'Focal_Loss']
		U_skip: ['noskip', 'skip']
		
	2、train_U_D.py test object:
		The loss function is determined as Focal_Loss and the connection mode as skip in train_U.py. 
		This training primarily demonstrates the improvement brought by the discriminator to the model.

	3、test_test.py
		The test target is separated from the training set. The test accounts for 15/16. 
		The test data is stored in the split file and the test accuracy is calculated.

	4、show_result.py
		Visualizes the confusion matrix.
		
	5、dataset.py
		Obtains the dataset.
		
	6、data:
		Stores the raw dataset.
		data/F3/train/train_labels.npy and data/F3/train/train_seismic.npy
		download link: https://zenodo.org/record/3755060/files/data.zip

		
	7、model:
		Stores the model-related code.
		
	8、save_train_models:
		Stores the trained models.
		
	9、test_fig:
		Stores seismic profiles and seismic facies profiles during the testing process.
	10、setting.py
		Hyperparameter setting


===============================================================================================
baseline model use Unet
	1、generator = modelG.GeneratorUNet                 ： Unet
	2、generator = modelG.GeneratorUNet_noskip          ： Unet w/o skip
	3、generator = modelG.GeneratorUNet                 ： Unet w/ D
	   discriminator = modelG.Discriminator             :  Discriminator

The geological body identification module focuses on a specific seismic facies within a particular stratigraphic interval in seismic data, 
treating other seismic facies as background. Compared to seismic facies identification, 
geological body identification merges the facies classification results of each seismic profile to form a single body. 
It then proceeds with seismic geomorphology analysis.


===============================================================================================
We thank dGB Earth Sciences for providing free and public datasets from the F3 block. 
data download link: https://zenodo.org/record/3755060/files/data.zip
We also thank the Society of Exploration Geophysicists (SEG) for providing free and public datasets from Parihaka New Zealand.
data download link: https://public.3.basecamp.com/p/JyT276MM7krjYrMoLqLQ6xST

