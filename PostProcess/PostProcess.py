"""
    后处理功能函数
    后处理相关的辅助函数都在此实现：
        1.预测标签解析
        2.序列中值滤波
        3.逐标签中值滤波
        4.标签合法化封装
"""
from typing import *
import torch
import numpy as np
from statistics import median_low

"""
    预测标签解析 raw model predict output --> signal mat(numpy.ndarray)
    raw model predict output --> Mat [4, frame_count]
    4D label vector [outside, nonsense, ileocecal, bbps0-3] (用-1表示各标签的无效标签)
"""


def parse_predict_label(predict_label: List[Tuple[torch.Tensor, torch.Tensor]]) -> np.ndarray:
    """
    :param predict_label: raw model predict output
    :return: label mat(numpy.ndarray)
    """
    # 前一个 Tensor 是预测概率，后一个 Tensor 是二值化后的标签，只需要后者
    # binary_label_mat: (len(predict_label), 7)
    # 7D label vector: [outside, nonsense, ileocecal, bbps0, bbps1, bbps2, bbps3]
    binary_label_mat: np.ndarray = np.array([label[1].squeeze().cpu().numpy() for label in predict_label]).astype(float)
    # 将 7D label vector 转换成 4D label vector，bbps0-3 合并成一个 bbps， 用-1表示各标签的无效标签
    bbps: np.ndarray = np.argmax(binary_label_mat[:, 3:7], axis=1)  # shape: (len(predict_label),)
    # 4D label vector [outside, nonsense, ileocecal, bbps]
    signal_mat: np.ndarray = np.concatenate((binary_label_mat[:, 0:3], bbps[:, np.newaxis]), axis=1)  # shape: (len(predict_label), 4)
    # 有outside，则其他标签均为无效标签；有nonsense，则ileocecal与bbps为无效标签
    signal_mat[signal_mat[:, 0] == 1, 1:] = -1
    signal_mat[signal_mat[:, 1] == 1, 2:] = -1
    signal_mat = signal_mat.T  # shape: (4, len(predict_label))
    return signal_mat


def extract_predict_logit_label_7(predict_label: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    :param predict_label: raw model predict output
    :return: logit_mat, label_mat
    """
    # 前一个 Tensor 是预测概率，后一个 Tensor 是二值化后的标签
    # logit_mat: (len(predict_label), 7)
    # label_mat: (len(predict_label), 7)
    # 7D label vector: [outside, nonsense, ileocecal, bbps0, bbps1, bbps2, bbps3]
    logit_list: List[List[float]] = [list(label[0].squeeze().cpu().numpy().astype(float)) for label in predict_label]
    label_list: List[List[float]] = [list(label[1].squeeze().cpu().numpy().astype(float)) for label in predict_label]
    return logit_list, label_list


"""
    序列中值滤波 sequence(numpy.ndarray) --> median-filtered sequence(numpy.ndarray)
    对一个1D序列进行中值滤波
    核规格 N 作为参数（大致相当于 thresh=N/2 的效力）
    对边缘进行Padding，Padding标签均为-1（无效值）
    以滑窗方式遍历，对于窗内数据，剔除无效标签（清洁度的-1），然后取下中位数作为滤波结果
    Seq [label0, label1, ...] --> Seq [label0, label1, ...]
"""


def median_filter_sequence(seq: np.ndarray, N: int) -> np.ndarray:
    """
    :param seq: sequence(numpy.ndarray)
    :param N: kernel size
    :return: median-filtered sequence(numpy.ndarray)
    """
    pad_size = N // 2
    if N % 2 == 0:  # 如果N是偶数，那么向右扩展一位窗口
        padded_seq = np.pad(seq, (pad_size, pad_size + 1), 'constant', constant_values=-1)
    else:  # 如果N是奇数，那么两边扩展相同的窗口
        padded_seq = np.pad(seq, (pad_size, pad_size), 'constant', constant_values=-1)

    filtered_seq = np.zeros_like(padded_seq)
    for i in range(pad_size, len(padded_seq) - pad_size):
        window = padded_seq[i - pad_size: i + pad_size + 1]
        valid_values = window[window != -1]
        if valid_values.size > 0:
            filtered_seq[i] = median_low(valid_values.tolist())
        else:
            filtered_seq[i] = -1  # 如果窗口内所有值都是无效值，则滤波结果也设为无效值

    if N % 2 == 0:  # 如果N是偶数，那么去掉右边多余的一位
        return filtered_seq[pad_size:-pad_size - 1]
    else:
        return filtered_seq[pad_size:-pad_size]


"""
    逐标签中值滤波 signal mat(numpy.ndarray) --> median-filtered mat(numpy.ndarray)
    核规格列表 List[int*4] 作为参数
    调用 序列中值滤波 对每个标签分别执行中值滤波
    Mat [4, frame_count] --> Mat [4, frame_count]
"""


def median_filter_signal_mat(signal_mat: np.ndarray, kernel_sizes: List[int]) -> np.ndarray:
    """
    :param signal_mat: signal mat(numpy.ndarray)
    :param kernel_sizes: kernel size list
    :return: median-filtered mat(numpy.ndarray)
    """
    median_filtered_mat = np.zeros_like(signal_mat)
    for i in range(4):
        median_filtered_mat[i] = median_filter_sequence(signal_mat[i], kernel_sizes[i])
    return median_filtered_mat


"""
    标签合法化封装 median-filtered sequence(numpy.ndarray) -> result sequence(numpy.ndarray)
    记 F = median-filtered mat（中值滤波后的）， S = signal mat（原始的）
    转置以上两个矩阵 F | S --> Mat [frame_count, 4]
    遍历 F 中的每个标签向量，处理唯一的不合法情况 [0, 0, 0, -1]（在体内但没有清洁度，假定为第c帧）:
        if S[c][3] != -1: F[c][3] = S[c][3] # 如果有原始清洁度，则直接取用原标签
        else: F[c][1] = 1 # 否则直接设置为坏帧
    最后执行和网络模型中一样的标签抑制过程
    Mat [4, frame_count] --> Mat [frame_count, 4]
"""


def legalize_label(median_filtered_mat: np.ndarray, signal_mat: np.ndarray) -> np.ndarray:
    """
    :param median_filtered_mat: median-filtered mat(numpy.ndarray)
    :param signal_mat: signal mat(numpy.ndarray)
    :return: result mat(numpy.ndarray)
    """
    # 转置以上两个矩阵 F | S --> Mat [frame_count, 4]
    F = median_filtered_mat.T
    S = signal_mat.T
    # 遍历 F 中的每个标签向量，处理唯一的不合法情况 [0, 0, 0, -1]（在体内但没有清洁度，假定为第c帧）:
    #     if S[c][3] != -1: F[c][3] = S[c][3] # 如果有原始清洁度，则直接取用原标签
    #     else: F[c][1] = 1 # 否则直接设置为坏帧
    for i in range(F.shape[0]):
        if F[i][3] == -1 and F[i][0] == 0:  # 如果是在体内且没有清洁度
            if S[i][3] != -1:  # 如果有原始清洁度，则直接取用原标签
                F[i][3] = S[i][3]
            else:
                F[i][1] = 1
    # 最后执行和网络模型中一样的标签抑制过程
    # Mat [4, frame_count] --> Mat [frame_count, 4]
    F[F[:, 0] == 1, 1:] = -1
    F[F[:, 1] == 1, 2:] = -1
    return F
