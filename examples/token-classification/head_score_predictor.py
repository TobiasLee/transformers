import torch
import torch.nn as nn


class SparseLoss(nn.Module):
    def __init__(self):
        super(SparseLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, head_score):
        target = torch.zeros_like(head_score)
        return self.loss(head_score, target)


class MLPPredictor(nn.Module):
    def __init__(self, head_num=12, layer_num=12, hidden_size=128):
        super(MLPPredictor, self).__init__()
        self.hidden1 = nn.Linear(head_num * layer_num, hidden_size)
        self.act = nn.ReLU()
        self.hidden2 = nn.Linear(hidden_size, head_num * layer_num)
        self.output = nn.Sigmoid()
        self.layer_num = layer_num
        self.head_num = head_num

    def forward(self, head_importance):
        head_importance = head_importance.reshape(1, -1)  # flattn
        hidden1 = self.hidden1(head_importance)
        head_score = self.output(self.hidden2(self.act(hidden1)))
        head_score = head_score.view(self.layer_num, self.head_num).contiguous()
        return head_score
