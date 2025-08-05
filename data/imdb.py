# IMDB dataset loader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import torch

def yield_tokens(data_iter, tokenizer):
    for label, line in data_iter:
        yield tokenizer(line)

def get_imdb_dataset(batch_size=64, max_length=512):
    tokenizer = get_tokenizer("basic_english")

    # Build vocab using training data
    train_iter = IMDB(split='train')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    def encode(text):
        tokens = tokenizer(text)
        token_ids = vocab(tokens)[:max_length]
        if len(token_ids) < max_length:
            token_ids += [vocab["<pad>"]] * (max_length - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.int64)

    label_map = {'neg': 0, 'pos': 1}

    def collate_batch(batch):
        texts, labels = zip(*batch)
        tokenized = [encode(text) for text in texts]
        labels = torch.tensor([label_map[label] for label in labels], dtype=torch.int64)
        return torch.stack(tokenized), labels

    # Reload the iterators after vocab build
    train_iter, test_iter = IMDB(split='train'), IMDB(split='test')
    train_loader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(list(test_iter), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, test_loader, vocab

if __name__ == '__main__':
    train_loader, test_loader, vocab = get_imdb_dataset()

    for inputs, labels in train_loader:
        print("Batch input shape:", inputs.shape)  # [batch_size, max_length]
        print("Batch label shape:", labels.shape)  # [batch_size]
        break
