from torch.utils.data import Dataset
from tqdm import tqdm
import torch


class LMDataset(Dataset):
    def __init__(self, split, data, tokenizer):
        assert split in {"train", "test"}
        self.split = split
        self.data = data
        self.tokenizer = tokenizer
        self.block_size = max([len(self.tokenizer.encode(inp)) for inp in self.data])
        self.process()

    def __len__(self):
        return len(self.data)

    def process(self):
        new_data = []
        for inp in tqdm(self.data):
            x = self.tokenizer.encode(inp)
            # Don't remove the last token - we want to keep [END] for training
            y = x[1:]  # Target is shifted by one position
            x = x[:-1]  # Input is everything except the last token
            x = x + ([-1] * (self.block_size - len(x)))
            y = y + ([-1] * (self.block_size - len(y)))
            new_data.append((x, y))
        self.data = new_data

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)


# Incredibly Simplified Tokenizer so that you can manually hack it!
# Feel free to add special tokens or modify as you wish.
# For real world tokenizer usage, see https://huggingface.co/docs/tokenizers/
class Tokenizer:
    def __init__(self):
        self.DELIM = "|[DELIM]|"
        self.special_tokens = ["[SEP]", "[END]"]
        self.special_tokens = [
            self.stringify(list(bytes(tok, "utf-8"))) for tok in self.special_tokens
        ]
        self.vocab_size = 256 + len(self.special_tokens)

    def stringify(self, b_enc):
        s_enc = [str(b) for b in b_enc]
        return self.DELIM.join(s_enc)

    def get_vocab_size(self):
        return self.vocab_size

    def encode(self, inp):
        s_enc = self.stringify(list(bytes(inp, "utf-8")))
        for i, tok in enumerate(self.special_tokens):
            s_enc = s_enc.replace(tok, str(255 + i + 1))
        return [int(s) for s in s_enc.split(self.DELIM)]

    def decode(self, inp):
        s_enc = self.stringify(inp)
        for i, tok in enumerate(self.special_tokens):
            s_enc = s_enc.replace(str(255 + i + 1), tok)
        return bytes([int(c) for c in s_enc.split(self.DELIM)])
