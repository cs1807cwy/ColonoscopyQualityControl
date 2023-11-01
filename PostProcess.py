import json
import torch


def merge_outside_outliers(idx_101, idx_010):
    idx_ab = []
    idx_ab.extend(idx_101)
    idx_ab.extend(idx_010)
    idx_ab.sort(key=lambda x: x[0])
    idx_merged = [idx_ab.pop(0)]
    for it in idx_ab:
        if it[0] == idx_merged[-1][1] + 1:
            idx_merged[-1][1] = it[1]
        else:
            idx_merged.append(it)
    return idx_merged


def fix_outliers_idx(pred_origin, outlier_list, val_func):
    pred_fix = pred_origin.clone().detach()
    for item in outlier_list:
        a, b = item
        for now_i in range(a, b + 1):
            pred_fix[now_i] = torch.tensor(val_func(pred_origin, item, now_i))
    return pred_fix


def val_reverse_outside(pred, idx_item, now_i):
    a, b = idx_item
    if pred[a][0].item() == 0:
        if pred[now_i][0].item() == 1 or pred[now_i][1].item() == 1:
            print(f'Fix {idx_item}: {now_i} not change')
            return pred[now_i].clone().detach()
        else:
            print(f'Fix {idx_item}: {now_i}=[1, 0, 0, 0, 0, 0, 0]')
            return [1, 0, 0, 0, 0, 0, 0]
    elif pred[a][0].item() == 1:
        if pred[now_i][0].item() == 0:
            print(f'Fix {idx_item}: {now_i} not change')
            return pred[now_i].clone().detach()
        else:
            print(f'Fix {idx_item}: {now_i}=[0, 1, 0, 0, 0, 0, 0]')
            return [0, 1, 0, 0, 0, 0, 0]
    else:
        raise ValueError


def val_reverse_nonsense_101(pred, idx_item, now_i):
    if pred[now_i][1].item() == 0:
        print(f'Fix {idx_item}: {now_i}=[0, 1, 0, 0, 0, 0, 0]')
        return [0, 1, 0, 0, 0, 0, 0]
    elif pred[now_i][1].item() == 1:
        print(f'Fix {idx_item}: {now_i} not change')
        return pred[now_i].clone().detach()
    else:
        raise ValueError


def post_process(pred_to_fix, outlier_json_path, do_fix_outside, do_fix_nonsense):
    outliers = json.load(open(outlier_json_path))

    outside_outliers_101 = outliers['0']['0']
    outside_outliers_010 = outliers['0']['1']
    nonsense_outliers_101 = outliers['1']['0']
    nonsense_outliers_010 = outliers['1']['1']

    outside_outliers = merge_outside_outliers(outside_outliers_101, outside_outliers_010)
    nonsense_outliers = nonsense_outliers_101

    pred_fix = pred_to_fix.clone().detach()

    if do_fix_outside: pred_fix = fix_outliers_idx(pred_fix, outside_outliers, val_reverse_outside)
    if do_fix_nonsense: pred_fix = fix_outliers_idx(pred_fix, nonsense_outliers, val_reverse_nonsense_101)

    return pred_fix.clone().detach()
