import numpy as np
from utils import evaluate


def get_train_miou(pred,seg_mask):
    refined_pseudo_label = pred.clone()
    refined_pseudo_label[pred == 255]
    if 255 in refined_pseudo_label:
        refined_pseudo_label[refined_pseudo_label == 255] = 0
        refined_pseudo_label[pred == 255]
    miou = evaluate.scores(seg_mask.cpu().numpy().astype(np.int16),refined_pseudo_label.cpu().numpy().astype(np.int16))['miou']
    return miou