import torch
import torch.nn as nn

class BERT4Rec(nn.Module):
    def __init__(self, num_items, max_seq_len=20,
                 embed_dim=64, num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # Ensure batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Two-layer GELU output head
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_items + 1)
        )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        # input_ids: [B, L]
        B, L = input_ids.size()
        device = input_ids.device

        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = self.item_emb(input_ids) + self.pos_emb(pos_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)

        # transformer expects [B, L, D] with batch_first=True
        h = self.transformer(x)
        return h

    def predict_masked(self, hidden_states, mask_positions):
        # hidden_states[mask_positions] -> [N_masked, D]
        h_masked = hidden_states[mask_positions]
        return self.predictor(h_masked)

    def predict_next(self, hidden_states):
        # hidden_states[:, -1, :] -> [B, D]
        h_last = hidden_states[:, -1, :]
        return self.predictor(h_last)