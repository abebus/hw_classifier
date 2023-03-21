from dataset.sportsmans_height import Sportsmanheight
from model.simple_classifier import Classifier
import numpy as np
import pandas as pd
from config.cfg import cfg
import plotly.graph_objects as go
import plotly.express as ex
import copy
from easydict import EasyDict


def TP_TN_FP_FN(confidence, gt, predictions):
    tp = np.sum(np.logical_and(predictions > confidence, gt == 1))
    tn = np.sum(np.logical_and(predictions <= confidence, gt == 0))
    fp = np.sum(np.logical_and(predictions > confidence, gt == 0))
    fn = np.sum(np.logical_and(predictions <= confidence, gt == 1))
    return dict(tp=tp, tn=tn, fp=fp, fn=fn)

def calculate_metrics(TP, TN, FP, FN):
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    type_1_error = FP / (TN + FP)
    type_2_error = FN / (TP + FN)
    F_score = 2 * precision * recall / (precision + recall)
    return dict(accuracy=accuracy,
                precision=precision,
                recall=recall,
                type_1_error=type_1_error,
                type_2_error=type_2_error,
                F_score=F_score)


def visualise(x, y, title, labels):
    fig = ex.area(
        x=x, y=y,
        title=title,
        labels=labels
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()


if __name__ == '__main__':
    dataset = Sportsmanheight()()
    predictions = Classifier()(dataset['height'])
    heights = dataset['height']
    gt = dataset['class']

    precision, recall = [], []
    fpr, tpr = [], []
    for confidence in predictions:
        bin_classification = TP_TN_FP_FN(confidence, gt, predictions)

        metrics = calculate_metrics(*bin_classification.values())

        precision.append(metrics['precision'])
        recall.append(metrics['recall'])

        fpr.append(bin_classification['fp'])
        tpr.append(bin_classification['tp'])
    ...
    precision.append(1)
    recall.append(0)

    pr_labels = dict(x='precision', y='recall')
    visualise(precision, recall, 'pr', pr_labels)

    roc_labels = dict(x='fpr', y='tpr')
    visualise(fpr, tpr, 'ROC', roc_labels)

