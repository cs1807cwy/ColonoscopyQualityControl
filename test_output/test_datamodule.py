from Classifier.DataModule import *
from MultiLabelClassifier import *


def TestColonoscopySiteQualityDataModule_SiteQuality():
    print('TestColonoscopySiteQualityDataModule_SiteQuality')
    # Dict[数据子集名, Dict[{索引文件index|目录dir}, 路径]]
    image_index_dir: Dict[str, Dict[str, str]] = {
        'UIHIMG-ileocecal': {'index': '../Datasets/KIndex/UIHIMG/ileocecal/fold0.json',
                             'dir': '../Datasets/UIHIMG/ileocecal'},
        'UIHIMG-nofeature': {'index': '../Datasets/KIndex/UIHIMG/nofeature/fold0.json',
                             'dir': '../Datasets/UIHIMG/nofeature'},
        'UIHIMG-nonsense': {'index': '../Datasets/KIndex/UIHIMG/nonsense/fold0.json',
                            'dir': '../Datasets/UIHIMG/nonsense'},
        'UIHIMG-outside': {'index': '../Datasets/KIndex/UIHIMG/outside/fold0.json',
                           'dir': '../Datasets/UIHIMG/outside'},
        'Nerthus-0': {'index': '../Datasets/KIndex/Nerthus/0/fold0.json',
                      'dir': '../Datasets/Nerthus/0'},
        'Nerthus-1': {'index': '../Datasets/KIndex/Nerthus/1/fold0.json',
                      'dir': '../Datasets/Nerthus/1'},
        'Nerthus-2': {'index': '../Datasets/KIndex/Nerthus/2/fold0.json',
                      'dir': '../Datasets/Nerthus/2'},
        'Nerthus-3': {'index': '../Datasets/KIndex/Nerthus/3/fold0.json',
                      'dir': '../Datasets/Nerthus/3'},
    }
    # Dict[数据子集名, 标签]
    image_label: Dict[str, str] = {
        'UIHIMG-ileocecal': 'fine',
        'UIHIMG-nofeature': 'fine',
        'UIHIMG-nonsense': 'nonsense',
        'UIHIMG-outside': 'outside',
        'Nerthus-0': 'fine',
        'Nerthus-1': 'fine',
        'Nerthus-2': 'fine',
        'Nerthus-3': 'fine',
    }
    sample_weight: Union[None, int, float, Dict[str, Union[int, float]]] = {
        'UIHIMG-ileocecal': 1666,
        'UIHIMG-nofeature': 1666,
        'UIHIMG-nonsense': 5000,
        'UIHIMG-outside': 5000,
        'Nerthus-0': 417,
        'Nerthus-1': 417,
        'Nerthus-2': 417,
        'Nerthus-3': 417,
    }

    resize_shape: Tuple[int, int] = (306, 306)
    center_crop_shape: Tuple[int, int] = (256, 256)
    brightness_jitter: Union[float, Tuple[float, float]] = 0.8
    contrast_jitter: Union[float, Tuple[float, float]] = 0.8
    saturation_jitter: Union[float, Tuple[float, float]] = 0.8
    batch_size: int = 16
    num_workers: int = 0

    cqc_data_module: ColonoscopySiteQualityDataModule = ColonoscopySiteQualityDataModule(
        image_index_dir,
        image_label,
        sample_weight,
        resize_shape,
        center_crop_shape,
        brightness_jitter,
        contrast_jitter,
        saturation_jitter,
        batch_size,
        num_workers,
        True
    )
    cqc_data_module.setup('fit')

    # 统计表
    # Dict[
    #   标签,
    #   Dict{
    #       sample_count: 采样标签计数,
    #       item_count: 不重复的项目计数,
    #       content: Dict[
    #           原始标签,
    #           Dict{
    #               sample_count: 采样原始标签计数,
    #               item_count: 不重复的项目计数,
    #               content: Dict[图像文件名, 计数]
    #           }
    #       ]
    #   }
    # ]
    item_counter: Dict[str, Dict[str, Optional[int, Dict[str, Dict[str, Optional[int, Dict[str, int]]]]]]] = {}

    train_dataloader = cqc_data_module.train_dataloader()

    from tqdm import tqdm
    epochs: int = 21
    samples: int = epochs * cqc_data_module.size('train')
    with tqdm(total=samples) as pbar:
        pbar.set_description('Processing')
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(train_dataloader):
                # 依次为
                # label: 包装后的3类标签{outside|nonsense|fine}
                # origin_label: 原始标签{UIHIMG-ileocecal|Nerthus-0|...}
                # basename: 图像文件名
                label_code, label, origin_label, basename = batch
                # print(f'\tBatch {batch_idx}:' + str({'label': label, 'basename': basename}))
                for lc, lb, olb, bn in zip(label_code, label, origin_label, basename):
                    if item_counter.get(lb) is None:
                        item_counter[lb] = {
                            'sample_count': 1, 'item_count': 0,
                            'code': lc.tolist(),
                            'content': {olb: {'sample_count': 1, 'item_count': 0, 'content': {bn: 1}}}
                        }
                    elif item_counter[lb]['content'].get(olb) is None:
                        item_counter[lb]['sample_count'] += 1
                        item_counter[lb]['content'][olb] = {'sample_count': 1, 'item_count': 0, 'content': {bn: 1}}
                    elif item_counter[lb]['content'][olb]['content'].get(bn) is None:
                        item_counter[lb]['sample_count'] += 1
                        item_counter[lb]['content'][olb]['sample_count'] += 1
                        item_counter[lb]['content'][olb]['content'][bn] = 1
                    else:
                        item_counter[lb]['sample_count'] += 1
                        item_counter[lb]['content'][olb]['sample_count'] += 1
                        item_counter[lb]['content'][olb]['content'][bn] += 1
                pbar.update(len(label))

    # 计算不重复项数
    for lb in item_counter:
        for olb in item_counter[lb]['content']:
            itc = len(item_counter[lb]['content'][olb]['content'])
            item_counter[lb]['content'][olb]['item_count'] = itc
            item_counter[lb]['item_count'] += itc

    # 获取类别编码的方法调用
    print('[One-Hot Label Code for Reference]')
    print('[label to code]', cqc_data_module.get_label_code_dict('train'))
    print('[code to label]', cqc_data_module.get_code_label_dict('train'))

    import json
    os.makedirs('test_output', exist_ok=True)
    with open('test_output/SiteQuality_count_log.json', 'w') as count_file:
        json.dump(item_counter, count_file, indent=2)

    import matplotlib.pyplot as plt
    plt.rc('font', family='SimHei')  # 设置字体为黑体
    plt.rc('axes', unicode_minus=False)  # 解决坐标轴负号显示问题
    parameters = {'axes.labelsize': 30,
                  'axes.titlesize': 30,
                  'figure.titlesize': 30,
                  'xtick.labelsize': 20,
                  'ytick.labelsize': 20,
                  }
    plt.rcParams.update(parameters)
    colors = ['#00429d'] * 2 + ['#c76a9f'] * 7 + ['#d0e848'] * 2

    # 采样总数
    x_label: List[str] = []
    y_count: List[int] = []
    for lb, ct in item_counter.items():
        x_label.append(lb)
        y_count.append(item_counter[lb]['sample_count'])
        for olb, ct2 in item_counter[lb]['content'].items():
            x_label.append(olb)
            y_count.append(item_counter[lb]['content'][olb]['sample_count'])
    plt.figure(figsize=(25, 10))
    plt.title("[训练集SiteQuality]采样总数")
    plt.xlabel("采样计数")
    plt.ylabel("数据子集")
    bar_graph = plt.barh(x_label, y_count, height=0.5, color=colors)
    for rect in bar_graph:
        w = rect.get_width()
        plt.text(w + 1, rect.get_y() + rect.get_height() / 2, '%d' % int(w), ha='left', va='center', fontsize=20)
    plt.show()

    # 覆盖总数
    x_label: List[str] = []
    y_count: List[int] = []
    for lb, ct in item_counter.items():
        x_label.append(lb)
        y_count.append(item_counter[lb]['item_count'])
        for olb, ct2 in item_counter[lb]['content'].items():
            x_label.append(olb)
            y_count.append(item_counter[lb]['content'][olb]['item_count'])
    plt.figure(figsize=(25, 10))
    plt.title("[训练集SiteQuality]覆盖总数")
    plt.xlabel("覆盖计数")
    plt.ylabel("数据子集")
    bar_graph = plt.barh(x_label, y_count, height=0.5, color=colors)
    for rect in bar_graph:
        w = rect.get_width()
        plt.text(w + 1, rect.get_y() + rect.get_height() / 2, '%d' % int(w), ha='left', va='center', fontsize=20)
    plt.show()

    # 采样频数分布直方图
    for lb, ct in item_counter.items():
        x_label.append(lb)
        y_count.append(item_counter[lb]['item_count'])
        for olb, ct2 in item_counter[lb]['content'].items():
            x_label.append(olb)
            all_samples = list(item_counter[lb]['content'][olb]['content'].values())
            plt.figure(figsize=(15, 10))
            plt.title(f"[训练集SiteQuality][{olb}]采样频数分布直方图")
            plt.xlabel("采样计数区间")
            plt.ylabel("频数")
            bins_split = max(1, int(max(all_samples) - min(all_samples))) * 2
            nums, bins, patches = plt.hist(all_samples, bins=bins_split, edgecolor='k')
            plt.xticks(bins, bins)
            for num, beg, end in zip(nums, bins[:-1], bins[1:]):
                plt.text((beg + end) / 2, num, '%d' % num, ha='center', va='bottom', fontsize=20)
            plt.show()


def TestColonoscopySiteQualityDataModule_IleocecalDetect():
    print('TestColonoscopySiteQualityDataModule_IleocecalDetect')
    # Dict[数据子集名, Dict[{索引文件index|目录dir}, 路径]]
    image_index_dir: Dict[str, Dict[str, str]] = {
        'UIHIMG-ileocecal': {'index': '../Datasets/KIndex/UIHIMG/ileocecal/fold0.json',
                             'dir': '../Datasets/UIHIMG/ileocecal'},
        'UIHIMG-nofeature': {'index': '../Datasets/KIndex/UIHIMG/nofeature/fold0.json',
                             'dir': '../Datasets/UIHIMG/nofeature'}
    }
    # Dict[数据子集名, 标签]
    image_label: Dict[str, str] = {
        'UIHIMG-ileocecal': 'ileocecal',
        'UIHIMG-nofeature': 'nofeature',
    }
    sample_weight: Union[None, int, float, Dict[str, Union[int, float]]] = {
        'UIHIMG-ileocecal': 5163,
        'UIHIMG-nofeature': 5163,
    }

    resize_shape: Tuple[int, int] = (306, 306)
    center_crop_shape: Tuple[int, int] = (256, 256)
    brightness_jitter: Union[float, Tuple[float, float]] = 0.8
    contrast_jitter: Union[float, Tuple[float, float]] = 0.8
    saturation_jitter: Union[float, Tuple[float, float]] = 0.8
    batch_size: int = 16
    num_workers: int = 0

    cqc_data_module: ColonoscopySiteQualityDataModule = ColonoscopySiteQualityDataModule(
        image_index_dir,
        image_label,
        sample_weight,
        resize_shape,
        center_crop_shape,
        brightness_jitter,
        contrast_jitter,
        saturation_jitter,
        batch_size,
        num_workers,
        True
    )
    cqc_data_module.setup('fit')

    # 统计表
    # Dict[
    #   标签,
    #   Dict{
    #       sample_count: 采样标签计数,
    #       item_count: 不重复的项目计数,
    #       content: Dict[
    #           原始标签,
    #           Dict{
    #               sample_count: 采样原始标签计数,
    #               item_count: 不重复的项目计数,
    #               content: Dict[图像文件名, 计数]
    #           }
    #       ]
    #   }
    # ]
    item_counter: Dict[str, Dict[str, Optional[int, Dict[str, Dict[str, Optional[int, Dict[str, int]]]]]]] = {}

    train_dataloader = cqc_data_module.train_dataloader()

    from tqdm import tqdm
    epochs: int = 21
    samples: int = epochs * cqc_data_module.size('train')
    with tqdm(total=samples) as pbar:
        pbar.set_description('Processing')
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(train_dataloader):
                # 依次为
                # label: 包装后的3类标签{outside|nonsense|fine}
                # origin_label: 原始标签{UIHIMG-ileocecal|Nerthus-0|...}
                # basename: 图像文件名
                label_code, label, origin_label, basename = batch
                # print(f'\tBatch {batch_idx}:' + str({'label': label, 'basename': basename}))
                for lc, lb, olb, bn in zip(label_code, label, origin_label, basename):
                    if item_counter.get(lb) is None:
                        item_counter[lb] = {
                            'sample_count': 1, 'item_count': 0,
                            'code': lc.tolist(),
                            'content': {olb: {'sample_count': 1, 'item_count': 0, 'content': {bn: 1}}}
                        }
                    elif item_counter[lb]['content'].get(olb) is None:
                        item_counter[lb]['sample_count'] += 1
                        item_counter[lb]['content'][olb] = {'sample_count': 1, 'item_count': 0, 'content': {bn: 1}}
                    elif item_counter[lb]['content'][olb]['content'].get(bn) is None:
                        item_counter[lb]['sample_count'] += 1
                        item_counter[lb]['content'][olb]['sample_count'] += 1
                        item_counter[lb]['content'][olb]['content'][bn] = 1
                    else:
                        item_counter[lb]['sample_count'] += 1
                        item_counter[lb]['content'][olb]['sample_count'] += 1
                        item_counter[lb]['content'][olb]['content'][bn] += 1
                pbar.update(len(label))

    # 计算不重复项数
    for lb in item_counter:
        for olb in item_counter[lb]['content']:
            itc = len(item_counter[lb]['content'][olb]['content'])
            item_counter[lb]['content'][olb]['item_count'] = itc
            item_counter[lb]['item_count'] += itc

    # 获取类别编码的方法调用
    print('[One-Hot Label Code for Reference]')
    print('[label to code]', cqc_data_module.get_label_code_dict('train'))
    print('[code to label]', cqc_data_module.get_code_label_dict('train'))

    import json
    os.makedirs('test_output', exist_ok=True)
    with open('test_output/IleocecalDetect_count_log.json', 'w') as count_file:
        json.dump(item_counter, count_file, indent=2)

    import matplotlib.pyplot as plt
    plt.rc('font', family='SimHei')  # 设置字体为黑体
    plt.rc('axes', unicode_minus=False)  # 解决坐标轴负号显示问题
    parameters = {'axes.labelsize': 30,
                  'axes.titlesize': 30,
                  'figure.titlesize': 30,
                  'xtick.labelsize': 20,
                  'ytick.labelsize': 20,
                  }
    plt.rcParams.update(parameters)
    colors = ['#00429d'] * 2 + ['#c76a9f'] * 7 + ['#d0e848'] * 2

    # 采样总数
    x_label: List[str] = []
    y_count: List[int] = []
    for lb, ct in item_counter.items():
        x_label.append(lb)
        y_count.append(item_counter[lb]['sample_count'])
        for olb, ct2 in item_counter[lb]['content'].items():
            x_label.append(olb)
            y_count.append(item_counter[lb]['content'][olb]['sample_count'])
    plt.figure(figsize=(25, 10))
    plt.title("[训练集IleocecalDetect]采样总数")
    plt.xlabel("采样计数")
    plt.ylabel("数据子集")
    bar_graph = plt.barh(x_label, y_count, height=0.5, color=colors)
    for rect in bar_graph:
        w = rect.get_width()
        plt.text(w + 1, rect.get_y() + rect.get_height() / 2, '%d' % int(w), ha='left', va='center', fontsize=20)
    plt.show()

    # 覆盖总数
    x_label: List[str] = []
    y_count: List[int] = []
    for lb, ct in item_counter.items():
        x_label.append(lb)
        y_count.append(item_counter[lb]['item_count'])
        for olb, ct2 in item_counter[lb]['content'].items():
            x_label.append(olb)
            y_count.append(item_counter[lb]['content'][olb]['item_count'])
    plt.figure(figsize=(25, 10))
    plt.title("[训练集IleocecalDetect]覆盖总数")
    plt.xlabel("覆盖计数")
    plt.ylabel("数据子集")
    bar_graph = plt.barh(x_label, y_count, height=0.5, color=colors)
    for rect in bar_graph:
        w = rect.get_width()
        plt.text(w + 1, rect.get_y() + rect.get_height() / 2, '%d' % int(w), ha='left', va='center', fontsize=20)
    plt.show()

    # 采样频数分布直方图
    for lb, ct in item_counter.items():
        x_label.append(lb)
        y_count.append(item_counter[lb]['item_count'])
        for olb, ct2 in item_counter[lb]['content'].items():
            x_label.append(olb)
            all_samples = list(item_counter[lb]['content'][olb]['content'].values())
            plt.figure(figsize=(15, 10))
            plt.title(f"[训练集IleocecalDetect][{olb}]采样频数分布直方图")
            plt.xlabel("采样计数区间")
            plt.ylabel("频数")
            bins_split = max(1, int(max(all_samples) - min(all_samples))) * 2
            nums, bins, patches = plt.hist(all_samples, bins=bins_split, edgecolor='k')
            plt.xticks(bins, bins)
            for num, beg, end in zip(nums, bins[:-1], bins[1:]):
                plt.text((beg + end) / 2, num, '%d' % num, ha='center', va='bottom', fontsize=20)
            plt.show()


def TestColonoscopyMultiLabelSiteQualityDataModule():
    print('TestColonoscopyMultiLabelSiteQualityDataModule')
    image_index_file: str = '../Datasets/UIHIMGMultiLabel/index/fold0.json'
    sample_weight: Union[None, int, float, Dict[str, Union[int, float]]] = {
        'ileocecal': 4000,
        'nofeature': 4000,
        'outside': 4000,
    }

    resize_shape: Tuple[int, int] = (268, 268)
    center_crop_shape: Tuple[int, int] = (224, 224)
    brightness_jitter: Union[float, Tuple[float, float]] = 0.8
    contrast_jitter: Union[float, Tuple[float, float]] = 0.8
    saturation_jitter: Union[float, Tuple[float, float]] = 0.8
    batch_size: int = 16
    num_workers: int = 0

    cqc_data_module: ColonoscopyMultiLabelDataModule = ColonoscopyMultiLabelDataModule(
        image_index_file,
        sample_weight,
        resize_shape,
        center_crop_shape,
        brightness_jitter,
        contrast_jitter,
        saturation_jitter,
        batch_size,
        num_workers,
        True
    )
    cqc_data_module.setup('fit')

    # 统计表
    # Dict[
    #   标签,
    #   Dict{
    #       sample_count: 采样标签计数,
    #       item_count: 不重复的项目计数,
    #       content: Dict[图像文件名, 计数]
    #   }
    # ]
    item_counter: Dict[str, Dict[str, Optional[int, Dict[str, Dict[str, Optional[int, Dict[str, int]]]]]]] = {}

    train_dataloader = cqc_data_module.train_dataloader()

    from tqdm import tqdm
    epochs: int = 21
    samples: int = epochs * cqc_data_module.size('train')
    with tqdm(total=samples) as pbar:
        pbar.set_description('Processing')
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(train_dataloader):
                # 依次为
                # label_code_ts: 标签编码Tensor
                # label: 标签List[str]
                # subset_key: 3个数据子集{outside|ileocecal|nofeature}
                # basename: 图像文件名
                label_code_ts, label, subset_key, image_path = batch

                for lc, sk, pt, pt in zip(label_code_ts, label, subset_key, image_path):
                    if item_counter.get(pt) is None:
                        item_counter[pt] = {
                            'sample_count': 1,
                            'item_count': 0,
                            'content': {pt: 1}
                        }
                    elif item_counter[pt]['content'].get(pt) is None:
                        item_counter[pt]['sample_count'] += 1
                        item_counter[pt]['content'][pt] = 1
                    else:
                        item_counter[pt]['sample_count'] += 1
                        item_counter[pt]['content'][pt] += 1
                pbar.update(len(label))

    # 计算不重复项数
    for sk in item_counter:
        itc = len(item_counter[sk]['content'])
        item_counter[sk]['item_count'] = itc

    # 获取类别编码的方法调用
    print('[Sequencial Label Code for Reference]')
    print('[label to code]', cqc_data_module.get_label_code_dict('train'))
    print('[code to label]', cqc_data_module.get_code_label_dict('train'))

    import json
    os.makedirs('test_output', exist_ok=True)
    with open('test_output/MultiLabel_count_log.json', 'w') as count_file:
        json.dump(item_counter, count_file, indent=2)

    import matplotlib.pyplot as plt
    plt.rc('font', family='SimHei')  # 设置字体为黑体
    plt.rc('axes', unicode_minus=False)  # 解决坐标轴负号显示问题
    parameters = {'axes.labelsize': 30,
                  'axes.titlesize': 30,
                  'figure.titlesize': 30,
                  'xtick.labelsize': 20,
                  'ytick.labelsize': 20,
                  }
    plt.rcParams.update(parameters)
    colors = ['#00429d'] * 2 + ['#c76a9f'] * 7 + ['#d0e848'] * 2

    # 采样总数
    x_label: List[str] = []
    y_count: List[int] = []
    for sk, ct in item_counter.items():
        x_label.append(sk)
        y_count.append(item_counter[sk]['sample_count'])
    plt.figure(figsize=(25, 10))
    plt.title("[MultiLabel训练集]采样总数")
    plt.xlabel("采样计数")
    plt.ylabel("数据子集")
    bar_graph = plt.barh(x_label, y_count, height=0.5, color=colors)
    for rect in bar_graph:
        w = rect.get_width()
        plt.text(w + 1, rect.get_y() + rect.get_height() / 2, '%d' % int(w), ha='left', va='center', fontsize=20)
    plt.show()

    # 覆盖总数
    x_label: List[str] = []
    y_count: List[int] = []
    for sk, ct in item_counter.items():
        x_label.append(sk)
        y_count.append(item_counter[sk]['item_count'])
    plt.figure(figsize=(25, 10))
    plt.title("[MultiLabel训练集]覆盖总数")
    plt.xlabel("覆盖计数")
    plt.ylabel("数据子集")
    bar_graph = plt.barh(x_label, y_count, height=0.5, color=colors)
    for rect in bar_graph:
        w = rect.get_width()
        plt.text(w + 1, rect.get_y() + rect.get_height() / 2, '%d' % int(w), ha='left', va='center', fontsize=20)
    plt.show()

    # 采样频数分布直方图
    for sk, ct in item_counter.items():
        x_label.append(sk)
        y_count.append(item_counter[sk]['item_count'])
        all_samples = list(item_counter[sk]['content'].values())
        plt.figure(figsize=(15, 10))
        plt.title(f"[MultiLabel训练集][{sk}]采样频数分布直方图")
        plt.xlabel("采样计数区间")
        plt.ylabel("频数")
        bins_split = max(1, int(max(all_samples) - min(all_samples))) * 2
        nums, bins, patches = plt.hist(all_samples, bins=bins_split, edgecolor='k')
        plt.xticks(bins, bins)
        for num, beg, end in zip(nums, bins[:-1], bins[1:]):
            plt.text((beg + end) / 2, num, '%d' % num, ha='center', va='bottom', fontsize=20)
        plt.show()


if __name__ == '__main__':
    TestColonoscopySiteQualityDataModule_SiteQuality()
    TestColonoscopySiteQualityDataModule_IleocecalDetect()
    # TestColonoscopyMultiLabelSiteQualityDataModule()
