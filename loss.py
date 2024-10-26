import numpy as np
from sklearn.metrics import roc_auc_score

def dice_coefficient(y_true, y_pred):
    tp = np.sum((y_true == 255) & (y_pred == 255))
    fp = np.sum((y_true == 0) & (y_pred == 255))
    fn = np.sum((y_true == 255) & (y_pred == 0))
    return  2 * tp / (2 * tp + fp + fn)
def sensitivity(y_true, y_pred):
    tp = np.sum((y_true == 255) & (y_pred == 255))
    fn = np.sum((y_true == 255) & (y_pred == 0))
    return tp / (tp + fn)

def specificity(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 255))
    return tn / (tn + fp)

def accuracy(y_true, y_pred):
    tp = np.sum((y_true == 255) & (y_pred == 255))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 255))
    fn = np.sum((y_true == 255) & (y_pred == 0))
    # import pdb; pdb.set_trace()
    return (tp + tn) / (tp + tn + fp + fn)

def auc(y_true, y_pred):
    return roc_auc_score(y_true.flatten(), y_pred.flatten())

def evaluate_segmentation(ground_truth, segmented_image):
    dice = dice_coefficient(ground_truth, segmented_image)
    sens = sensitivity(ground_truth, segmented_image)
    spec = specificity(ground_truth, segmented_image)
    acc = accuracy(ground_truth, segmented_image)
    auc_score = auc(ground_truth, segmented_image)
    return dice, sens, spec, acc, auc_score