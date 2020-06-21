import torch
import torch.nn as nn

class HardConcretePredictor(nn.Module):
    def __init__(self, args, shape=(12, 12), temperature=0.33, stretch_limits=(-0.1, 1.1), eps=1e-6, hard=False):
        super(HardConcretePredictor, self).__init__()
        self.temperature, self.stretch_limits, self.eps = temperature, stretch_limits, eps
        self.log_a = nn.Parameter(torch.ones(shape))
        self.hard = hard
        self.shape = shape
        self.args = args 
        #print(self.log_a)

    def forward(self, head_importance):
        low, high = self.stretch_limits

        if self.training:
            noise = torch.ones(self.shape).uniform_(self.eps, 1 - self.eps).to(self.args.device)
            concrete = torch.sigmoid((torch.log(noise) - torch.log(1 - noise) + self.log_a) / self.temperature)
        else:
            concrete = torch.sigmoid(self.log_a)
        stretched_concrete = concrete * (high - low) + low
        clipped_concrete = torch.clamp(stretched_concrete, min=0, max=1)
        if self.hard:
            hard_concrete = torch.gt(clipped_concrete, 0.5).float().to(self.args.device)
            clipped_concrete = clipped_concrete + (hard_concrete - clipped_concrete).clone().detach()
        return clipped_concrete


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
