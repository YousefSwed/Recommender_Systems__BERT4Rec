import torch
import torch.nn as nn
from config import SEQ_LEN, EMBED_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT

class BERT4Rec(nn.Module):
    def __init__(self, num_items, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=DROPOUT):
        super(BERT4Rec, self).__init__()
        
        # Item embedding layer
        self.item_embedding = nn.Embedding(num_items + 2, embed_dim, padding_idx=0)
        
        # Positional embedding layer
        self.pos_embedding = nn.Embedding(SEQ_LEN, embed_dim)

        # Transformer encoder layer config
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True  
        )
        
        # Transofmrer encoder with specified number of layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization and dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Output layer to predict item probabilities
        self.output = nn.Linear(embed_dim, num_items + 2)

    def forward(self, x, mask):
        # First create indices for the sequence
        positions = torch.arange(0, SEQ_LEN).unsqueeze(0).to(x.device)
        
        # Add item and positional embeddings
        x = self.item_embedding(x) + self.pos_embedding(positions)
        
        # Apply normalization and dropout
        x = self.norm(self.dropout(x))

        # Pass through the transformer encoder
        x = self.transformer(x, src_key_padding_mask=mask)

        # Return output logits for every item in the sequence
        return self.output(x) 
