from typing import Dict, List
import os
import os.path as osp
import json
import argparse

"""
输入格式
4个predict.txt path 4个参数对应四个模型的输出文件
name 顺序在4个文件中一致
{
    name: num
}

# 输出格式
{
    name: [0, 0, 0, 0, 0, 0, 0]
}
"""


# 将四个模型的输出文件合并为一份Json导出，此函数返回值是该Json对应的字典
def merge_4sep_model_result_file(
        outside_result_path: str,
        nonsense_result_path: str,
        ileocecal_result_path: str,
        cleansing_result_path: str,
        output_path: str
) -> Dict[str, List[int]]:
    """
    返回 Dict
    # {
        img_path: [outside, nonsense, ileocecal, bbps0, bbps1, bbps2, bbps3]
    # }
    """

    with open(outside_result_path) as fp:
        outside_result: Dict = json.load(fp)
        print(outside_result)
    with open(nonsense_result_path) as fp:
        nonsense_result: Dict = json.load(fp)
        print(nonsense_result)
    with open(ileocecal_result_path) as fp:
        ileocecal_result: Dict = json.load(fp)
        print(ileocecal_result)
    with open(cleansing_result_path) as fp:
        cleansing_result: Dict = json.load(fp)
        print(cleansing_result)

    merged_json: Dict[str, List[int]] = {}
    # 约定四个模型的输出文件含有相同的键
    # [outside, nonsense, ileocecal, bbps0, bbps1, bbps2, bbps3]

    for img_id, outside_label in outside_result.items():
        nonsense_label: int = nonsense_result[img_id] if outside_label == 0 else 0
        ileocecal_label: int = ileocecal_result[img_id] if outside_label == 0 and nonsense_label == 0 else 0
        seq_label: List[int] = [outside_label, nonsense_label, ileocecal_label]
        if outside_label == 0 and nonsense_label == 0:
            bbps: int = cleansing_result[img_id]
            seq_label += [0] * bbps + [1] + [0] * (3 - bbps)
        else:
            seq_label += [0] * 4
        merged_json[img_id] = seq_label

    out_abs_path: str = osp.abspath(output_path)
    out_dir: str = osp.dirname(out_abs_path)
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, 'w') as fp:
        json.dump(merged_json, fp, indent=2)
    print(f'Done. Result json dump to {out_abs_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-orp', '--outside_result_path', required=True, help='体内外识别模型结果文件（Json）')
    parser.add_argument('-nrp', '--nonsense_result_path', required=True, help='坏帧识别模型结果文件（Json）')
    parser.add_argument('-irp', '--ileocecal_result_path', required=True, help='回盲部识别模型结果文件（Json）')
    parser.add_argument('-crp', '--cleansing_result_path', required=True, help='清洁度识别模型结果文件（Json）')
    parser.add_argument('-o', '--output_path', required=True, help='合并输出文件（Json）')
    args = parser.parse_args()
    print(args)
    # python Merge4SepModelResultFile.py -orp outside_result.json -nrp nonsense_result.json -irp ileocecal_result.json -crp cleansing_result.json -o merged_result.json

    merge_4sep_model_result_file(
        args.outside_result_path,
        args.nonsense_result_path,
        args.ileocecal_result_path,
        args.cleansing_result_path,
        args.output_path
    )
