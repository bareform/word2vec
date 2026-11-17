from models import CBOW, SkipGram

from collections import OrderedDict
import os

import torch
import torch.utils.data as data
from tqdm import tqdm

import torchutils
from utils.dataset_utils import TextDataset
from utils.training_utils import HuffmanTree

def get_argparser():
    parser = torchutils.ArgumentParser("Simple training loop.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wikitext2"],
        help="Dataset to train on. Must be on of: `wikitext2`.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cbow", "skip-gram"],
        help="Model to train. Must be one of: `cbow`, `skip-gram`.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training (default: 256)"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Number of words on each side of the target to use as context (default: 5).",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=5,
        help="Minimum token frequency to include in the vocabulary (default: 5).",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=100,
        help="Embedding dimension (default: 100).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads to use in data loading (default: 4)"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=True,
        help="Whether to pin memory for data loading (default: True)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
        help="Number of epochs to train for (default: 30)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.003,
        help="Learning rate for (default: 0.003).",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1 (default: 0.9).",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 (default: 0.999).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(".", "wikitext2"),
        help="Directory to save checkpoints (default: ./checkpoints/wikitext2).",
    )
    parser.add_argument(
        "--save_checkpoint_interval",
        type=int,
        default=2,
        help="Interval (in epochs) at which to save model checkpoints (default: 2).",
    )
    parser.add_argument(
        "--compute_val_loss_interval",
        type=int,
        default=1,
        help="Interval (in epochs) at which to compute validation loss (default: 1).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    return parser

def main():
    args = get_argparser().parse_args()
    pad_length = len(str(args.num_epochs))

    torchutils.set_seed(args.random_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.dataset == "wikitext2":
        dataset_id = ("Salesforce/wikitext", "wikitext-2-raw-v1")
        train_dataset = TextDataset(
            mode=args.mode,
            dataset_id=dataset_id,
            window_size=args.window_size,
            min_freq=args.min_freq,
        )
        validation_dataset = TextDataset(
            mode=args.mode,
            dataset_id=dataset_id,
            window_size=args.window_size,
            min_freq=args.min_freq,
            split="validation",
            word2idx=train_dataset.word2idx,
        )
    frequencies = {
        word: train_dataset.vocab[word]
        for word in train_dataset.word2idx
    }
    huffman_tree = HuffmanTree(frequencies)
    dataloader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    validation_dataloader = data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model_name = "CBOW" if args.mode == "cbow" else "Skip-Gram"
    print(f"Training a {model_name} model on {args.dataset}")
    if args.mode == "cbow":
        model = CBOW(
            vocab_size=train_dataset.vocab_size,
            embedding_dim=args.embedding_dim,
            internal_node_count=huffman_tree.node_count,
            padding_idx=train_dataset.padding_idx,
        )
    elif args.mode == "skip-gram":
        model = SkipGram(
            vocab_size=train_dataset.vocab_size,
            embedding_dim=args.embedding_dim,
            internal_node_count=huffman_tree.node_count,
        )

    max_path_length = max(len(path) for path in huffman_tree.paths.values())
    path_node_ids = torch.zeros(
        (train_dataset.vocab_size, max_path_length),
        dtype=torch.long,
    )
    path_bits = torch.zeros(
        (train_dataset.vocab_size, max_path_length),
        dtype=torch.float32,
    )
    path_mask = torch.zeros(
        (train_dataset.vocab_size, max_path_length),
        dtype=torch.bool,
    )
    for word, word_idx in train_dataset.word2idx.items():
        path = huffman_tree.paths[word]
        if path:
            path_node_ids[word_idx, :len(path)] = torch.tensor(
                [node_index for node_index, _ in path],
                dtype=torch.long,
            )
            path_bits[word_idx, :len(path)] = torch.tensor(
                [branch for _, branch in path],
                dtype=torch.float32,
            )
            path_mask[word_idx, :len(path)] = True

    model = model.to(device)
    path_node_ids = path_node_ids.to(device)
    path_bits = path_bits.to(device)
    path_mask = path_mask.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    average_validation_loss = None

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        training_examples = 0

        with tqdm(dataloader, desc="Training", unit="batch") as pbar:
            for inputs, targets in pbar:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                batch_path_nodes = path_node_ids[targets]
                batch_path_bits = path_bits[targets]
                batch_path_mask = path_mask[targets]

                optimizer.zero_grad()
                logits = model(inputs, batch_path_nodes)
                path_losses = criterion(logits, batch_path_bits)
                loss = (path_losses * batch_path_mask).sum(dim=1).mean()
                loss.backward()
                optimizer.step()

                batch_size = targets.size(0)
                running_loss += loss.item() * batch_size
                training_examples += batch_size
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                })

        average_loss = running_loss / training_examples

        average_validation_loss = None
        if (epoch + 1) % args.compute_val_loss_interval == 0:
            model.eval()
            validation_loss = 0.0
            validation_examples = 0
            with tqdm(validation_dataloader, desc="Validation", unit="batch") as pbar:
                with torch.inference_mode():
                    for inputs, targets in pbar:
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        batch_path_nodes = path_node_ids[targets]
                        batch_path_bits = path_bits[targets]
                        batch_path_mask = path_mask[targets]

                        logits = model(inputs, batch_path_nodes)
                        path_losses = criterion(logits, batch_path_bits)
                        loss = (path_losses * batch_path_mask).sum(dim=1).mean()

                        batch_size = targets.size(0)
                        validation_loss += loss.item() * batch_size
                        validation_examples += batch_size
                        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            average_validation_loss = validation_loss / validation_examples

        print(f"Epoch: {epoch + 1}/{args.num_epochs}")
        print(f"Train Loss: {average_loss:.5f}")
        if average_validation_loss is not None:
            print(f"Validation Loss: {average_validation_loss:.5f}")

        if (epoch + 1) % args.save_checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f"{args.dataset}-{args.mode}_checkpoint_{epoch + 1:0{pad_length}d}.pth",
            )
            torch.save(
                {
                    "dataset": args.dataset,
                    "mode": args.mode,
                    "embedding_dim": args.embedding_dim,
                    "window_size": args.window_size,
                    "min_freq": args.min_freq,
                    "word2vec": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "word2idx": train_dataset.word2idx,
                    "huffman_codes": huffman_tree.codes,
                },
                checkpoint_path,
            )

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"{args.dataset}-{args.mode}_{args.num_epochs}.pth",
    )
    torch.save(
        {
            "dataset": args.dataset,
            "word2vec": model.state_dict(),
            "mode": args.mode,
            "vocab_size": train_dataset.vocab_size,
            "embedding_dim": args.embedding_dim,
            "internal_node_count": huffman_tree.node_count,
            "padding_idx": train_dataset.padding_idx,
            "window_size": args.window_size,
            "word2idx": train_dataset.word2idx,
            "idx2word": train_dataset.idx2word,
            "huffman_codes": huffman_tree.codes,
            "huffman_paths": huffman_tree.paths,
        },
        checkpoint_path,
    )

if __name__ == "__main__":
    main()
