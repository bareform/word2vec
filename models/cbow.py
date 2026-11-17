import torch
import torch.nn as nn

class CBOW(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        internal_node_count: int,
        padding_idx: int,
    ) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.output_embed = nn.Embedding(internal_node_count, embedding_dim)
        self.output_bias = nn.Embedding(internal_node_count, 1)

    def forward(self, inputs: torch.Tensor, path_nodes: torch.Tensor) -> torch.Tensor:
        mask = inputs.ne(self.padding_idx).unsqueeze(-1)
        embedded = self.embed(inputs)
        hidden = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        output = self.output_embed(path_nodes)
        bias = self.output_bias(path_nodes).squeeze(-1)
        return (output * hidden.unsqueeze(1)).sum(dim=-1) + bias
