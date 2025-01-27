import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import os


def get_all_images(folder,subfolders):
    imgs = []
    for sub in subfolders:
        for filepath in os.listdir(folder + '/' + sub):
            imgs.append(cv2.imread(folder + '/' + sub + '/' + filepath))
    return np.array(imgs)

def plotConfMatrix(confMatrix, labels, plotpath):
    leftmargin = 1.5 # inches
    rightmargin = 1.5 # inches
    categorysize = 1.5 # inches
    figwidth = leftmargin + rightmargin + (len(labels) * categorysize)
    fig = plt.figure(figsize=(figwidth, figwidth))
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    fig.subplots_adjust(left=leftmargin/figwidth, right=1-rightmargin/figwidth, top=0.94, bottom=0.1)
    cax = ax.matshow(confMatrix)

    for i in range(confMatrix.shape[0]):
      for j in range(confMatrix.shape[1]):
        c = confMatrix[j,i]
        if c > 0.6:
            color = 'black'
        else:
            color = 'w'
        ax.text(i, j, str(np.round(c,2)),color=color, va='center', ha='center')

    plt.title('Confusion matrix')
    fig.colorbar(cax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels)-0.5, 0-0.5)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(plotpath)

def confusion_matrix_metrics(confusion_matrix):
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    FPR = [0 if math.isnan(x) else x for x in FPR]
    # False negative rate
    FNR = FN/(TP+FN)
    FNR = [0 if math.isnan(x) else x for x in FNR]
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    #return FP, FN, TP, TN, TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC
    return FP, FN, TP, TN, TNR, FPR, FNR

# def create_model(model_name, model_dim, optimizer, loss_func, summary=True):
#     return model

# def save_training_configs(): #epochs, optimizer, image pre-processing, model-artchitecture
#     pass