import math
import torch
import torch.nn as nn
import torch.nn.functional as F

##############！！！！！！！！！！！！！！
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def process_matrix(tensor_matrix):
    min_value = torch.min(tensor_matrix)
    max_value = torch.max(tensor_matrix)

    margin = max_value * 0.1
    threshold_min = min_value + margin
    threshold_max = max_value - margin

    mask = (tensor_matrix > threshold_min) & (tensor_matrix < threshold_max)

    filtered_matrix = torch.zeros_like(tensor_matrix)
    filtered_matrix[mask] = tensor_matrix[mask]

    return filtered_matrix


#############

###############
import copy


def process_matrix_chu2(tensor_matrix,scale):
    min_value = torch.min(tensor_matrix)
    max_value = torch.max(tensor_matrix)
    margin = max_value * 0.2 #
    threshold_min = min_value + margin
    threshold_max = max_value - margin

    mask_get_middle =  (tensor_matrix < threshold_max)
    mask_get_max = (tensor_matrix > threshold_max)
    mask_get_min = (tensor_matrix < threshold_min)

    filtered_matrix1 = torch.zeros_like(tensor_matrix)
    filtered_matrix2 = copy.deepcopy(filtered_matrix1)
    filtered_matrix3 = copy.deepcopy(filtered_matrix1)
    filtered_matrix1[mask_get_middle] = tensor_matrix[mask_get_middle]
    filtered_matrix2[mask_get_max] = tensor_matrix[mask_get_max]
    filtered_matrix3[mask_get_min] = tensor_matrix[mask_get_min]
    filtered_matrix1[~mask_get_middle] = (tensor_matrix[~mask_get_middle] / 10)#*scale[2]
    filtered_matrix2[~mask_get_max] = tensor_matrix[~mask_get_max] / scale[0]
    filtered_matrix3[~mask_get_min] = (tensor_matrix[~mask_get_min] /10)#*scale[2]

    return filtered_matrix1, filtered_matrix2, filtered_matrix3


################


def transform_matrix_tensor_mask_mastrix(matrix):
    matrix = matrix.cuda()
    row_sums = matrix.sum(dim=1)
    transformed_matrix = torch.zeros_like(matrix)  # +1e-10
    transformed_matrix[row_sums > 1] = 1

    return transformed_matrix


def transform_matrix_tensor_mask(matrix):
    matrix = matrix.cuda()
    row_sums = matrix.sum(dim=1)
    transformed_vector = torch.zeros(matrix.size(0), device=matrix.device)
    transformed_vector[row_sums > 1] = 1

    return transformed_vector


def compute_sdm_per(scores, pid, logit_scale, epsilon=1e-8, tau=0.02, margin=0.1):
    """
     Similarity Distribution Matching
     """
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()


    GCL_mask = transform_matrix_tensor_mask(labels)
    mask = 1 - labels
    argc_i2t = ((scores / tau).exp() * labels / ((scores / tau).exp() * labels).sum(dim=1,
                                                                                     keepdim=True)).detach()  # detach()的意思是不参与梯度更新，只是用与前向传播使用
    argc_t2i = ((scores.t() / tau).exp() * labels / ((scores.t() / tau).exp() * labels).sum(dim=1,
                                                                                             keepdim=True)).detach()

    GCL_loss = (-  (argc_i2t * scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(
        max=10e35).log() + margin).clamp(min=0) \
               + (-  (argc_t2i * scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(
        max=10e35).log() + margin).clamp(min=0)
    GCL_loss = GCL_loss  # +0.0005
    GCL_loss_mask = GCL_mask * GCL_loss * 50
    #########################

    t2i_cosine_theta = scores
    i2t_cosine_theta = t2i_cosine_theta.t()
    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta


    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)
    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))


    loss = torch.sum(i2t_loss, dim=1) + torch.sum(t2i_loss, dim=1)
    return loss  # *GCL_mask_reverse+GCL_loss_mask




def compute_TRL_per(scores, pid, margin=0.2, tau=0.02):
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    argc_1 = ((scores / tau).exp() * labels / ((scores / tau).exp() * labels).sum(dim=1, keepdim=True)).detach()
    argc_2 = ((scores.t() / tau).exp() * labels / ((scores.t() / tau).exp() * labels).sum(dim=1,
                                                                                           keepdim=True)).detach()

    pos_1 = (argc_1 * scores).sum(1)
    pos_2 = (argc_2 * scores.t()).sum(1)

    neg_1 = (mask * scores).max(1)[0]
    neg_2 = (mask * scores.t()).max(1)[0]

    cost_1 = (margin + neg_1 - pos_1).clamp(min=0)
    cost_2 = (margin + neg_2 - pos_2).clamp(min=0)
    return cost_1 + cost_2


def compute_InfoNCE_per(scores, logit_scale):
    # cosine similarity as logits
    logits_per_image = logit_scale * scores
    logits_per_text = logits_per_image.t()

    p1 = F.softmax(logits_per_image, dim=1)
    p2 = F.softmax(logits_per_text, dim=1)

    loss = (- p1.diag().log() - p2.diag().log()) / 2
    return loss


def compute_GCL_perx(scores, pid, tau, margin):
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    argc_i2t = ((scores / tau).exp() * labels / ((scores / tau).exp() * labels).sum(dim=1, keepdim=True)).detach()
    argc_t2i = ((scores.t() / tau).exp() * labels / ((scores.t() / tau).exp() * labels).sum(dim=1,
                                                                                             keepdim=True)).detach()

    loss = (-  (argc_i2t * scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(
        max=10e35).log() + margin).clamp(min=0) \
           + (-  (argc_t2i * scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(
        max=10e35).log() + margin).clamp(min=0)

    return loss


import numpy as np


def compute_GCL_per(scores, pid, tau, margin,scale,argc):
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    argc_i2t = ((scores / tau).exp() * labels / ((scores / tau).exp() * labels).sum(dim=1,
                                                                                     keepdim=True)).detach()  # detach()的意思是不参与梯度更新，只是用与前向传播使用
    argc_t2i = ((scores.t() / tau).exp() * labels / ((scores.t() / tau).exp() * labels).sum(dim=1,
                                                                                             keepdim=True)).detach()

    scores_neg1, scores_neg2, scores_neg3 = process_matrix_chu2(scores,scale)
    loss_neg1 = (-  (argc_i2t * scores).sum(1) + tau * ((scores_neg1 / tau).exp() * mask).sum(1).clamp(max=10e35).log() + argc[1]).clamp(min=0) \
                + (-  (argc_t2i * scores.t()).sum(1) + tau * ((scores_neg1.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + argc[1]).clamp(min=0)
    loss_neg2 = (-  (argc_i2t * scores).sum(1) + tau * ((scores_neg2 / tau).exp() * mask).sum(1).clamp(max=10e35).log() + argc[2]).clamp(min=0) \
                + (-  (argc_t2i * scores.t()).sum(1) + tau * ((scores_neg2.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + argc[2]).clamp(min=0)
    loss_neg3 = (-  (argc_i2t * scores).sum(1) + tau * ((scores_neg3 / tau).exp() * mask).sum(1).clamp(max=10e35).log() + argc[0]).clamp(min=0) \
                + (-  (argc_t2i * scores.t()).sum(1) + tau * ((scores_neg3.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + argc[0]).clamp(min=0)

    return loss_neg1 + loss_neg2 + loss_neg3+ torch.tensor(np.random.uniform(0.00001, 0.000001, (batch_size, 1))).squeeze(1).to(scores.device)




def compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, pid, label_hat=None, tau=0.02, margin=0.1, loss_type='GCL',
                logit_scale=50,scale=None,argc=None):
    loss_bgm, _ = compute_per_loss(i_feats, t_feats, pid, tau, margin, loss_type, logit_scale,scale,argc)
    loss_tse, _ = compute_per_loss(i_tse_f, t_tse_f, pid, tau, margin, loss_type, logit_scale,scale,argc)

    loss_bgm = (label_hat * loss_bgm).sum()
    loss_tse = (label_hat * loss_tse).sum()

    if loss_type in ['GCL', 'TRL']:
        return loss_bgm, loss_tse
    else:
        return loss_bgm / label_hat.sum(), loss_tse / label_hat.sum()  # mean


def compute_per_loss(image_features, text_features, pid, tau=0.02, margin=0.2, loss_type='GCL', logit_scale=50,scale=None,argc=None):
    # # normalized features
    image_norm = image_features / image_features.norm(dim=-1,keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()

    if 'GCL' in loss_type:
        per_loss = compute_GCL_per(scores, pid, tau, margin=margin,scale=scale,argc=argc)
    elif 'TRL' in loss_type:
        per_loss = compute_TRL_per(scores, pid, tau=tau, margin=margin)
    elif 'InfoNCE' in loss_type:
        per_loss = compute_InfoNCE_per(scores, logit_scale)
    elif 'SDM' in loss_type:
        per_loss = compute_sdm_per(scores, pid, logit_scale)
    else:
        exit()

    return per_loss, scores.diag()