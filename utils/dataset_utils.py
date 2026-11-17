from collections import Counter

import torch
import torch.utils.data as data
from datasets import load_dataset

class TextDataset(data.Dataset):
    def __init__(
        self,
        mode: str,
        dataset_id: tuple[str, str],
        window_size: int,
        min_freq: int = 1,
        split: str = "train",
        word2idx: dict[str, int] | None = None,
    ) -> None:
        self.mode = mode
        self.window_size = window_size
        self.min_freq = min_freq

        dataset = load_dataset(*dataset_id)
        texts = [line for line in dataset[split]["text"] if line.strip()]

        self.tokenizer = lambda x: x.lower().split()
        tokens = [self.tokenizer(sentence) for sentence in texts]

        self.vocab = Counter(token for sentence in tokens for token in sentence)
        if word2idx is None:
            words = [
                word for word, count in self.vocab.items()
                if count >= min_freq
            ]
            self.word2idx = {
                word: idx for idx, word in enumerate(words)
            }
        else:
            self.word2idx = word2idx.copy()
        self.idx2word = {
            idx: word
                for word, idx in self.word2idx.items()
        }
        self.padding_idx = len(self.word2idx)
        self.samples = []
        for sentence in tokens:
            for idx, word in enumerate(sentence):
                if word not in self.word2idx:
                    continue
                context = [
                    sentence[i]
                    for i in range(
                        max(0, idx - window_size),
                        min(len(sentence), idx + window_size + 1),
                    )
                    if i != idx and sentence[i] in self.word2idx
                ]
                if self.mode == "cbow" and context:
                    self.samples.append((context, word))
                elif self.mode == "skip-gram":
                    self.samples.extend((word, context_word) for context_word in context)

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx) + 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "cbow":
            context, target = self.samples[idx]
            context_indices = [self.word2idx[w] for w in context]
            max_len = 2 * self.window_size
            context_indices += [self.padding_idx] * (max_len - len(context_indices))
            return (
                torch.tensor(context_indices, dtype=torch.long),
                torch.tensor(self.word2idx[target], dtype=torch.long),
            )
        elif self.mode == "skip-gram":
            target, context = self.samples[idx]
            return (
                torch.tensor(
                    self.word2idx[target],
                    dtype=torch.long,
                ),
                torch.tensor(
                    self.word2idx[context],
                    dtype=torch.long,
                ),
            )
