# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

confusion = np.array(([48, 0, 0, 2],
                      [1, 49, 0, 0],
                      [0, 1, 49, 0],
                      [1, 0, 0, 49]
                      ))
classes = ['MC', 'OX', 'QC', 'WD']


def calculae_lable_prediction(confMatrix):
    '''
    计算每一个类别的预测精度:该类被预测正确的数除以该类的总数
    '''
    l = len(confMatrix)
    for i in range(l):
        label_total_sum = confMatrix.sum(axis=1)[i]
        label_correct_sum = confMatrix[i][i]
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
        print('精确率:' + classes[i] + ":" + str(prediction) + '%')


def calculate_label_recall(confMatrix):
    l = len(confMatrix)
    for i in range(l):
        label_total_sum = confMatrix.sum(axis=0)[i]
        label_correct_sum = confMatrix[i][i]
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
        print('召回率:' + classes[i] + ":" + str(prediction) + '%')


calculae_lable_prediction(confusion)
calculate_label_recall(confusion)
