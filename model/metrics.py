'''
评估预测精度,模型最后使用，不在训练过程中使用
混淆矩阵： 对角线上的是分类正确的类别
'''
import numpy as np 

class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        '''
        	计算混淆矩阵
            self._fast_hist(lt.flatten(), lp.flatten(), self.classes)
            输入是一维，int(0-n_class)
        '''
        mask = (label_true >= 0) & (label_true < n_class)
        # minlength: 统计的最大的长度
        xx = n_class * label_true[mask].astype(int) + label_pred[mask]  # n_class*true + pred
        hist = np.bincount(xx, minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        '''
            叠加batch 中的每个样本统计的混淆矩阵，输出一个batch的统计结果
            label: (B，1，H，W) ； 具体的类别，整数0-n_class
            pred: (B，H，W)     ; 具体的类别，整数0-n_class
        '''
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """
        	计算各种评估参数
        	return：
        		{'Pixel Acc: ': acc,                                 # 总正确率，flot

                'Class Accuracy: ': acc_cls,                         # 每个类别的正确率, ndarray(n_class,)
                'Mean Class Acc: ': mean_acc_cls,                    # 平均正确率, flot

                'class IOU': cls_iu                                  # 每个类别的iou, dict{0:float,..}
                'Freq Weighted IoU: ': fwavacc,                      # 带权重的每个类别IOU,ndarray(n_class,)
                'Mean IoU: ': mean_iu,                               # 平均IOU, flot

                'confusion_matrix': self.confusion_matrix},          # 混淆矩阵

        """
        hist = self.confusion_matrix
        # 1、PA: 所有正确的/总数  , 标量
        acc = np.diag(hist).sum() / hist.sum()

        # 2、每个类别的正确率, **** 一维列表
        acc_cls = np.diag(hist) / hist.sum(axis=1)

        # 3、平均类别正确率，每个类别正确率的均值， 标量
        mean_acc_cls = np.nanmean(acc_cls)  # 求值时忽略NAN

        # 4、每个类别真实和预测的IOU，*** 一维列表。
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

        # 5、平均IOU，多个类别IOU均值
        mean_iu = np.nanmean(iu)

        # 6、带类别权重的IOU，【每个类别的频次*对应的IOU，求和】，防止少数量类别的影响。
        freq = hist.sum(axis=1) / hist.sum() # fraction of the pixels that come from each class
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        # IOU，每个类别的IOU， **** 字典的形式
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Pixel Acc: ': acc,                                  # 总正确率，flot
                
                'Class Accuracy: ': acc_cls,                         # 每个类别的正确率, ndarray(n_class,)
                'Mean Class Acc: ': mean_acc_cls,                    # 平均正确率, flot
                
                'class IOU': cls_iu,                                  # 每个类的IOU, dict{0:float,1:float...}
                'Freq Weighted IoU: ': fwavacc,                      # 带权重的每个类别IOU,ndarray(n_class,)，求和
                'Mean IoU: ': mean_iu,                               # 平均IOU, flot
                'confusion_matrix': self.confusion_matrix},          # 混淆矩阵

    def reset(self):
    	# 混淆矩阵初始化
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))



