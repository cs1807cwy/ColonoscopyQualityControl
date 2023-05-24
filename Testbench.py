import torch
import numpy as np

if __name__ == '__main__':
    a = torch.from_numpy(np.array([True, False, True, False]))
    print('# a:')
    print(a)

    b = torch.rand(4, 4)
    print('# b:')
    print(b)

    print('# ~a.unsqueeze(1):')
    print(~a.unsqueeze(1))

    print('# torch.max(b, dim=-1)[0].unsqueeze(1):')
    print(torch.max(b, dim=-1)[0].unsqueeze(1))

    print('# torch.ge(b, torch.max(b, dim=-1)[0].unsqueeze(1)):')
    print(torch.ge(b, torch.max(b, dim=-1)[0].unsqueeze(1)))

    out = ~a.unsqueeze(1) & torch.ge(b, torch.max(b, dim=-1)[0].unsqueeze(1))
    print('# out:')
    print(out)

    cat = torch.cat([a.unsqueeze(1), out], dim=-1)
    print('# cat:')
    print(cat)