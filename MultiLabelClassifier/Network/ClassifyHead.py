import torch
import torch.nn as nn


class ClassSpecificResidualAttention(nn.Module):  # one basic block
    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            temperature: int,
            attention_lambda: float
    ):
        super(ClassSpecificResidualAttention, self).__init__()
        self.temperature = temperature  # temperature
        self.attention_lambda = attention_lambda  # Lambda
        self.head = nn.Conv2d(input_dim, num_classes, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x (B d H W)
        # normalize classifier
        # score (B C HxW)
        score = self.head(x) / torch.norm(self.head.weight, dim=1, keepdim=True).transpose(0, 1)
        score = score.flatten(2)
        base_logit = torch.mean(score, dim=2)

        if self.temperature == 99:  # max-pooling
            attention_logit = torch.max(score, dim=2)[0]
        else:
            score_soft = self.softmax(score * self.temperature)
            attention_logit = torch.sum(score * score_soft, dim=2)

        return base_logit + self.attention_lambda * attention_logit


class ClassSpecificMultiHeadAttention(nn.Module):  # multi-head attention
    temperature_settings = {  # softmax temperature settings
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        6: [1, 2, 3, 4, 5, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99]
    }

    def __init__(
            self,
            num_heads: int,
            attention_lambda: float,
            input_dim: int,
            num_classes: int
    ):
        super(ClassSpecificMultiHeadAttention, self).__init__()
        self.temperature_list = self.temperature_settings[num_heads]
        self.multi_head = nn.ModuleList([
            ClassSpecificResidualAttention(input_dim, num_classes, self.temperature_list[i], attention_lambda)
            for i in range(num_heads)
        ])

    def forward(self, x):
        logit = 0.
        for head in self.multi_head:
            logit += head(x)
        return logit
