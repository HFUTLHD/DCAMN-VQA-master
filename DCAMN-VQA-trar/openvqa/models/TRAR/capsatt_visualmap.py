import sys
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class caps_visual(nn.Module):
    """
    Args:
    """
    def __init__(self, num_iterations, num_capsules, dim, out_dim):
        super(caps_visual, self).__init__()
        self.dp = nn.Dropout(0.1)
        self.num_capsules = num_capsules
        self.num_iterations = num_iterations
        self.out_layer = nn.Linear(dim, out_dim)
        self.att_size = dim
        self.W_h = nn.Parameter(torch.Tensor(dim, self.att_size))
        nn.init.xavier_uniform_(self.W_h)
        self.W_f = nn.Parameter(torch.Tensor(dim, self.att_size))
        nn.init.xavier_uniform_(self.W_f)

    def forward(self, query, feat, feat_mask=None):
        """
        routing algorithm.
        Args:

        """
        query = query @ self.W_h
        feat = feat @ self.W_f
        b = Variable(torch.zeros(feat.shape[0], feat.shape[1])).cuda()
        feat_mask = feat_mask.squeeze(1).squeeze(1)  # b,feat_len
        b = b.masked_fill(feat_mask, -1e18)
        for i in range(self.num_iterations):
            c = F.softmax(b, dim=1)
            if i == self.num_iterations-1:
                outputs = (c.unsqueeze(-1) * feat).sum(dim=1)  # b,dim
                query = query + outputs  # b,dim
                c_out = c
            else:
                # generate higl-level capsules and multimodal capsules
                delta_b = (query.unsqueeze(1) * feat).sum(dim=-1)  # b,feat_len
                delta_b = (delta_b - delta_b.mean(dim=1, keepdim=True)) / (delta_b.std(dim=1, keepdim=True) + 1e-9)
                b = b + delta_b
                # outputs = (c.unsqueeze(-1) * feat).sum(dim=1)  # b,dim
                outputs = (c.unsqueeze(-1) * feat).mean(dim=1)  # b,dim
                query = query + outputs  # b,dim

        return self.out_layer(query), c_out
        # query b,dim

