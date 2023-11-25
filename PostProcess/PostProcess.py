"""
    后处理功能函数
    后处理相关的辅助函数都在此实现：
        1.预测标签解析
        2.序列中值滤波
        3.逐标签中值滤波
        4.标签合法化封装
"""

"""
    预测标签解析 raw model predict output --> signal mat(numpy.ndarray)
    raw model predict output --> Mat [4, frame_count]
    4D label vector [outside, nonsense, ileocecal, bbps0-3] (用-1表示各标签的无效标签)
"""

"""
    序列中值滤波 sequence(numpy.ndarray) --> median-filtered sequence(numpy.ndarray)
    对一个1D序列进行中值滤波
    核规格 N 作为参数（大致相当于 thresh=N/2 的效力）
    对边缘进行Padding，Padding标签均为-1（无效值）
    以滑窗方式遍历，对于窗内数据，剔除无效标签（清洁度的-1），然后取下中位数作为滤波结果
    Seq [label0, label1, ...] --> Seq [label0, label1, ...]
"""

"""
    逐标签中值滤波 signal mat(numpy.ndarray) --> median-filtered mat(numpy.ndarray)
    核规格列表 List[int*4] 作为参数
    调用 序列中值滤波 对每个标签分别执行中值滤波
    Mat [4, frame_count] --> Mat [4, frame_count]
"""

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
