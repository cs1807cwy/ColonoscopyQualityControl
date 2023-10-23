import torch
import numpy as np
import os
import os.path as osp

if __name__ == '__main__':
    # a: 0 = 1
    # print(a)
    #
    # a = torch.zeros(1)
    # b = torch.ones(1)
    # print(a != b)
    # if a != b:
    #     print('ne')
    #
    # print(torch.compile())
    # path = '../aaa/bbb/ccc.txt'
    # jpath = osp.join('/mnt/data4/cwy/Test', path)
    # print(jpath)
    # print(osp.abspath(jpath))
    # print(osp.isabs(jpath))
    # print(osp.isabs('data/lll.txt'))
    # os.makedirs(osp.dirname(jpath), exist_ok=True)
    # with open(jpath, 'w') as fp:
    #     fp.write('Hello!')

    # a = torch.from_numpy(np.array([True, False, True, False]))
    # print('# a:')
    # print(a)
    #
    # b = torch.rand(4, 4)
    # print('# b:')
    # print(b)
    #
    # print('# ~a.unsqueeze(1):')
    # print(~a.unsqueeze(1))
    #
    # print('# torch.max(b, dim=-1)[0].unsqueeze(1):')
    # print(torch.max(b, dim=-1)[0].unsqueeze(1))
    #
    # print('# torch.ge(b, torch.max(b, dim=-1)[0].unsqueeze(1)):')
    # print(torch.ge(b, torch.max(b, dim=-1)[0].unsqueeze(1)))
    #
    # out = ~a.unsqueeze(1) & torch.ge(b, torch.max(b, dim=-1)[0].unsqueeze(1))
    # print('# out:')
    # print(out)
    #
    # cat = torch.cat([a.unsqueeze(1), out], dim=-1)
    # print('# cat:')
    # print(cat)

    a = torch.tensor([1])
    a = a.squeeze(0)
    print(a.numpy())