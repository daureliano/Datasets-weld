# -- coding: utf-8 --
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import itertools
import numpy as np

plt.figure(figsize=(6.4, 4.8))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.savefig('C:/Users/DELL/Desktop/cm.png', dpi=1440)
    # plt.savefig('/tmp/pycharm_project_553/hunju.png', dpi=1440)
    plt.savefig('D:/python/pytorch-image-models-master/hunju.png', dpi=1440)
    plt.show()


if __name__ == '__main__':
    # cm = np.array([[60, 0, 0, 0, 0, 0],
    #       [0, 59, 0, 1, 0, 0],
    #       [0, 0, 60, 0, 0, 0],
    #       [0, 0, 0, 60, 0, 0],
    #       [0, 0, 0, 0, 60, 0],
    #       [0, 0, 0, 0, 0, 60]])
    # classes = (
    #     'Cr'
    #     , 'In'
    #     , 'Pa'
    #     , 'PS'
    #     , 'RS'
    #     , 'Sc'
    #     )
    cm = np.array([[48, 0, 0, 2],
                   [1, 49, 0, 0],
                   [0, 1, 49, 0],
                   [1, 0, 0, 49]])
    classes = (
        'MC'
        , 'OX'
        , 'QC'
        , 'WD'
    )
    plot_confusion_matrix(cm, classes)
