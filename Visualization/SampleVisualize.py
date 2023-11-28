"""
    可视化程序入口
    实现可视化管线：（对管线中的每一步封装一个单独函数）
        1.读取视频 --> 指定源视频路径，提取元信息
        2.拆帧 --> 指定存储全部帧的目录
        3.实例化模型进行预测 --> （只读，可选参数）指定预测标签Json存储路径（logit、label都记录在一个Json里），指定信号图存储路径（logit 7个、label 7个标签信号，14合1的总图）
        4.解析预测结果 --> 变换成后处理所需的数据格式
        5.后处理 --> （只读，可选参数）调用后处理方法生成结果，输出到Json文件（指定标签Json存储路径，只记录label），指定信号图存储路径（label 4个标签信号，4合1的总图）
        6.帧渲染 --> 指定存储渲染帧的目录，调用帧渲染方法生成结果，
        7.合成视频 --> 指定合成视频的目录，视频规格与源视频的元信息保持一致

    注：
        管线中输出的任何Json仅作日志使用，请勿做数据读取之用
        管线中的下一步处理直接从上一步的输出中获取数据，无需通过Json中转
"""
import argparse

"""
    3.模型预测日志标签Json格式：
    {
        'logit': [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] # [outside, nonsense, ileocecal, bbps0, bbps1, bbps2, bbps3],
            ...
        ],
        'label': [
            [0, 0, 1, 0, 0, 0, 1] # [outside, nonsense, ileocecal, bbps0, bbps1, bbps2, bbps3],
            ...
        ]
    }
"""

"""
    5.后处理日志标签Json格式：
    [
        [0, 0, 0, 3] # [outside, nonsense, ileocecal, bbps(-1 for nobbps)],
        ...
    ]
"""




if __name__ == '__main__':
    """
        实例化 ArgumentParser，接收下列参数：[*]表示可选参数
            源视频路径 List[str]：指定每个视频路径
            拆帧目录：List[str]：每个视频指定一个拆帧目录
            [模型预测日志标签Json存储路径]：List[str]：每个视频指定一个Json存储路径
            [模型预测日志信号图存储路径]：List[str]：每个视频指定一个信号图存储路径
            卷积核规格列表 List[int*4]：分别对应 [体外，坏帧，回盲部，清洁度] 的中值滤波1D卷积核规格
            [后处理日志标签Json存储路径] List[str]：每个视频指定一个Json存储路径
            [后处理日志信号图存储路径] List[str]：每个视频指定一个信号图存储路径
            渲染目录：List[str]：每个视频指定一个渲染帧目录
            视频输出路径 List[str]：每个视频指定一个输出路径
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, nargs='+', help='视频路径')
    parser.add_argument('frame_path', type=str, nargs='+', help='拆帧目录')
    parser.add_argument('--pred_json_path', type=str, nargs='+', help='模型预测日志标签Json存储路径')
    parser.add_argument('--pred_signal_path', type=str, nargs='+', help='模型预测日志信号图存储路径')
    parser.add_argument('kernel_sizes', type=int, nargs='+', help='卷积核规格列表')
    parser.add_argument('--post_json_path', type=str, nargs='+', help='后处理日志标签Json存储路径')
    parser.add_argument('--post_signal_path', type=str, nargs='+', help='后处理日志信号图存储路径')
    parser.add_argument('render_path', type=str, nargs='+', help='渲染目录')
    parser.add_argument('output_path', type=str, nargs='+', help='视频输出路径')
