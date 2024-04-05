import math

import torch
from torch import nn
from torch.nn import functional as F


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.ln0 = None
        self.ln1 = None
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if self.ln0 is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if self.ln1 is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformerEncoder(nn.Module):
    def __init__(self, dim_in, dim_out, layers, num_heads=4, use_isab=False, num_inds=32, ln=False, dropout=0.):
        super(SetTransformerEncoder, self).__init__()

        modules = []
        for _ in range(layers):
            modules.append(
                ISAB(dim_in, dim_out, num_heads, num_inds, ln=ln)
                if use_isab else
                SAB(dim_in, dim_out, num_heads, ln=ln),
            )
            if dropout:
                modules.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*modules)

    def forward(self, X):
        encoded = self.layers(X)
        return encoded


class SetTransformerDecoder(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_outputs, layers, num_heads=4, use_isab=False, num_inds=32, ln=False, dropout=0.):
        super(SetTransformerDecoder, self).__init__()

        modules = [
            PMA(dim_in, num_heads, num_outputs, ln=ln),
        ]
        for _ in range(layers):
            modules.append(
                ISAB(dim_in, dim_hidden, num_heads, num_inds, ln=ln)
                if use_isab else
                SAB(dim_in, dim_hidden, num_heads, ln=ln)
            )
            if dropout:
                modules.append(nn.Dropout(dropout))
        modules.append(nn.Linear(dim_hidden, dim_out))
        self.layers = nn.Sequential(*modules)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(dim_out) if ln else None

    def forward(self, X):
        decoded = self.activation(self.layers(X))
        if self.norm:
            decoded = self.norm(decoded)
        return decoded
