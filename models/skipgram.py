import torch
import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        internal_node_count: int,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.output_embed = nn.Embedding(internal_node_count, embedding_dim)
        self.output_bias = nn.Embedding(internal_node_count, 1)

    def forward(self, inputs: torch.Tensor, path_nodes: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(inputs)
        output = self.output_embed(path_nodes)
        bias = self.output_bias(path_nodes).squeeze(-1)
        return (output * hidden.unsqueeze(1)).sum(dim=-1) + bias
