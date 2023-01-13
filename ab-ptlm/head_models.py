import torch
import torch.nn as nn
import torch.nn.functional as F
import enum


class HeadType(enum.Enum):
    ATTNMEAN = "ATTNMEAN"
    CONCAT = "CONCAT"
    ATTNPOOL = "ATTNPOOL"



class AttentionWeightedMean(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.compute_attention_weight = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: [B x L x D]
        # attn_weights: [B x L x 1]
        attn_weights = self.compute_attention_weight(x)
        padding_mask = (x == 0).all(-1, keepdims=True)
        attn_weights = attn_weights.masked_fill(padding_mask, -10000)
        attn_weights = attn_weights.softmax(1)
        attn_weights = self.dropout(attn_weights)
        x = (x.transpose(1, 2) @ attn_weights).squeeze(2)
        return x


class AttentionPooling(nn.Module):
    def __init__(
        self, 
        embed_dim: int
    ):
        super().__init__()
        self.project = nn.Linear(embed_dim, 1)

    def forward(self, x):
        softmax = F.softmax
        att_weights = softmax( self.project(x).squeeze(-1) ).unsqueeze(-1)
        output = torch.sum( x * att_weights, dim = 1 )
        return output


class ConcatProject(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        project_dim: int,
        max_length: int,
    ):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.project_dim = project_dim
        self.project = nn.Linear(embed_dim, project_dim)

    def forward(self, x):
        x = self.project(x)
        x = F.pad(x, [0, 0, 0, self.max_length - x.size(1)])
        output = x.view(x.size(0), self.max_length * self.project_dim)
        return output


class TRRosettaHead(nn.Sequential):

    def __init__(self):
        layers = [
            nn.Conv2d(64, 16, 3, stride=2, padding=1),  # 256
            nn.ELU(inplace=True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # 128
            nn.ELU(inplace=True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # 64
            nn.ELU(inplace=True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # 32
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)),  # 16
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # 8
            nn.ELU(inplace=True),
            nn.Flatten(),
        ]
        super().__init__(*layers)


class OutputHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        mlp: bool = False,
    ):
        super().__init__()
        if not mlp:
            self.mlp: nn.Module = nn.Linear(embed_dim, output_dim)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x, return_embedding: bool = False):
        if return_embedding:
            assert len(self.mlp) > 1
            outputs = []
            for layer in self.mlp:
                x = layer(x)
                outputs.append(x)
            prediction, embedding = outputs[-1], outputs[-2]
            return prediction, embedding
        else:
            return self.mlp(x)
