# BERT4Rec_model.py

import torch
import torch.nn as nn
from config import SEQ_LEN

class BERT4Rec(nn.Module):
    def __init__(self, num_items, embed_dim=64, num_layers=2, num_heads=4, dropout=0.2):
        super(BERT4Rec, self).__init__()
        self.item_embedding = nn.Embedding(num_items + 2, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(SEQ_LEN, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_items + 2)

    def forward(self, x, mask):
        positions = torch.arange(0, SEQ_LEN).unsqueeze(0).to(x.device)
        x = self.item_embedding(x) + self.pos_embedding(positions)
        x = self.norm(self.dropout(x))
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, embed_dim]
        return self.output(x)
