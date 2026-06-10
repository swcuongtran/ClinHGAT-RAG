import torch
from torch import nn


class TemporalDiseaseForecaster(nn.Module):
    """Transformer forecaster that attends over previous disease days."""

    def __init__(
        self,
        daily_dim: int,
        static_dim: int,
        hidden: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(daily_dim, hidden)
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, hidden) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.attention_query = nn.Linear(hidden, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, sequences, static_features, masks):
        hidden = self.input_proj(sequences)
        hidden = hidden + self.pos_embedding[:, : hidden.shape[1], :]
        key_padding_mask = masks == 0
        encoded = self.encoder(hidden, src_key_padding_mask=key_padding_mask)

        static_context = self.static_proj(static_features)
        query = self.attention_query(static_context).unsqueeze(-1)
        attn_logits = torch.bmm(encoded, query).squeeze(-1)
        attn_logits = attn_logits.masked_fill(key_padding_mask, -1e9)
        day_attention = torch.softmax(attn_logits, dim=-1)
        temporal_context = torch.bmm(day_attention.unsqueeze(1), encoded).squeeze(1)

        fused = torch.cat([temporal_context, static_context], dim=-1)
        logits = self.classifier(fused)
        forecast_risk = torch.sigmoid(self.forecast_head(fused)).squeeze(-1)
        return {
            "logits": logits,
            "forecast_risk": forecast_risk,
            "day_attention": day_attention,
            "encoded_days": encoded,
        }
