"""Equivariant pocket and ligand encoders with a shared embedding."""

from __future__ import annotations

import torch
from torch import nn

from schema import EMBED_DIM, LIGAND_DIM, N_LAYERS, POCKET_DIM


def scatter_sum(source, index, dim_size):
    output = source.new_zeros(dim_size, source.size(-1))
    output.index_add_(0, index, source)
    return output


class EGNNLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.phi_e = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1, bias=False),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, features, positions, edge_index):
        source, target = edge_index
        difference = positions[source] - positions[target]
        distance = (difference * difference).sum(-1, keepdim=True) / 10.0
        message = self.phi_e(torch.cat([features[source], features[target], distance], dim=-1))
        positions = positions + scatter_sum(difference * self.phi_x(message), source, positions.size(0))
        aggregated = scatter_sum(message, source, features.size(0))
        features = self.norm(features + self.phi_h(torch.cat([features, aggregated], dim=-1)))
        return features, positions


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden, n_layers):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden)
        self.layers = nn.ModuleList(EGNNLayer(hidden) for _ in range(n_layers))
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, features, positions, edge_index, batch):
        features = self.project(features)
        positions = positions - scatter_mean(positions, batch)[batch]
        for layer in self.layers:
            features, positions = layer(features, positions, edge_index)
        pooled = scatter_mean(features, batch)
        return nn.functional.normalize(self.output(pooled), dim=-1)


def scatter_mean(source, batch):
    if source.dim() == 1:
        source = source.unsqueeze(-1)
        summed = scatter_sum(source, batch, int(batch.max()) + 1).squeeze(-1)
        counts = scatter_sum(torch.ones_like(source), batch, int(batch.max()) + 1).squeeze(-1)
        return summed / counts.clamp(min=1)
    dim_size = int(batch.max()) + 1
    summed = scatter_sum(source, batch, dim_size)
    counts = scatter_sum(torch.ones(source.size(0), 1, device=source.device, dtype=source.dtype), batch, dim_size)
    return summed / counts.clamp(min=1)


class DualEncoder(nn.Module):
    def __init__(self, hidden=EMBED_DIM, n_layers=N_LAYERS):
        super().__init__()
        self.pocket = Encoder(POCKET_DIM, hidden, n_layers)
        self.ligand = Encoder(LIGAND_DIM, hidden, n_layers)
        self.logit_scale = nn.Parameter(torch.tensor(2.6593))  # log(1/0.07), CLIP initialization
        pair_dim = hidden * 3
        self.potency = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def encode_pocket(self, features, positions, edge_index, batch):
        return self.pocket(features, positions, edge_index, batch)

    def encode_ligand(self, features, positions, edge_index, batch):
        return self.ligand(features, positions, edge_index, batch)

    def similarity(self, ligand_embedding, pocket_embedding):
        scale = self.logit_scale.exp().clamp(max=100)
        return scale * ligand_embedding @ pocket_embedding.transpose(0, 1)

    def predict_potency(self, ligand_embedding, pocket_embedding):
        paired = torch.cat([
            ligand_embedding,
            pocket_embedding,
            ligand_embedding * pocket_embedding,
        ], dim=-1)
        return self.potency(paired).squeeze(-1)


class LigandOnlyEncoder(nn.Module):
    """Same ligand encoder and score, with one learned vector per panel target."""

    def __init__(self, n_targets, hidden=EMBED_DIM, n_layers=N_LAYERS):
        super().__init__()
        self.ligand = Encoder(LIGAND_DIM, hidden, n_layers)
        self.target = nn.Parameter(torch.randn(n_targets, hidden) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(2.6593))
        pair_dim = hidden * 3
        self.potency = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def encode_ligand(self, features, positions, edge_index, batch):
        return self.ligand(features, positions, edge_index, batch)

    def target_embedding(self):
        return nn.functional.normalize(self.target, dim=-1)

    def similarity(self, ligand_embedding, target_embedding):
        scale = self.logit_scale.exp().clamp(max=100)
        return scale * ligand_embedding @ target_embedding.transpose(0, 1)

    def predict_potency(self, ligand_embedding, target_embedding):
        paired = torch.cat([
            ligand_embedding,
            target_embedding,
            ligand_embedding * target_embedding,
        ], dim=-1)
        return self.potency(paired).squeeze(-1)


class SequenceEncoder(nn.Module):
    """Amino-acid sequence encoder. Coordinates are not used."""

    def __init__(self, hidden):
        super().__init__()
        self.embed = nn.Embedding(22, hidden, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, tokens, mask):
        hidden = self.embed(tokens).transpose(1, 2)
        hidden = self.conv(hidden).transpose(1, 2)
        hidden = hidden * mask.unsqueeze(-1)
        pooled = hidden.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return nn.functional.normalize(self.output(pooled), dim=-1)


class SequenceLigandEncoder(nn.Module):
    """Ligand encoder plus one sequence embedding per panel target."""

    def __init__(self, tokens, target_index, mask, hidden=EMBED_DIM, n_layers=N_LAYERS):
        super().__init__()
        self.ligand = Encoder(LIGAND_DIM, hidden, n_layers)
        self.sequence = SequenceEncoder(hidden)
        self.register_buffer("tokens", tokens)
        self.register_buffer("target_index", target_index)
        self.register_buffer("chain_mask", mask)
        self.n_targets = int(target_index.max()) + 1
        self.logit_scale = nn.Parameter(torch.tensor(2.6593))
        pair_dim = hidden * 3
        self.potency = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def encode_ligand(self, features, positions, edge_index, batch):
        return self.ligand(features, positions, edge_index, batch)

    def target_embedding(self):
        chain = self.sequence(self.tokens, self.chain_mask)
        pooled = scatter_sum(chain, self.target_index, self.n_targets)
        counts = scatter_sum(
            torch.ones(chain.size(0), 1, device=chain.device, dtype=chain.dtype),
            self.target_index,
            self.n_targets,
        )
        return nn.functional.normalize(pooled / counts.clamp(min=1), dim=-1)

    def similarity(self, ligand_embedding, target_embedding):
        scale = self.logit_scale.exp().clamp(max=100)
        return scale * ligand_embedding @ target_embedding.transpose(0, 1)

    def predict_potency(self, ligand_embedding, target_embedding):
        paired = torch.cat([
            ligand_embedding,
            target_embedding,
            ligand_embedding * target_embedding,
        ], dim=-1)
        return self.potency(paired).squeeze(-1)
