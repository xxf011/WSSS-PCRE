import torch.nn as nn
import torch.nn.functional as F
class Wsddn_Layer(nn.Module):
    def __init__(self,embdding_dim, num_classes,mil = True):
        super().__init__()
        self.cls = nn.Linear(embdding_dim, num_classes)
        self.det = nn.Linear(embdding_dim, num_classes)
        nn.init.xavier_uniform_(self.cls.weight)
        nn.init.xavier_uniform_(self.det.weight)
        self.mil = mil
        for l in [self.cls, self.det]:
            nn.init.constant_(l.bias, 0)
        # if no_mil:
    def forward(self,x):
        C = self.cls(x)
        if self.mil:
            D = self.det(x)
            scores = F.softmax(C, dim=-1) * F.softmax(D, dim=-2)
            return scores
        else:
            return C