import pickle
import torch
import pytest
from torch.utils.data import Dataset
from src.transformer import Decoder, generate, TransformerConfig
from src.trainer import Trainer, TrainerConfig


class SortDataset(Dataset):
    """
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self, split, length=6, num_digits=3):
        assert split in {"train", "test"}
        self.split = split
        self.length = length
        self.num_digits = num_digits

    def __len__(self):
        return 10000

    def get_vocab_size(self):
        return self.num_digits

    def get_block_size(self):
        return 20

    def __getitem__(self, idx):
        # use rejection sampling to generate an input example from the desired split
        while True:
            # generate some random integers
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            # half of the time let's try to boost the number of examples that
            # have a large number of repeats, as this is what the model seems to struggle
            # with later in training, and they are kind of rare
            if torch.rand(1).item() < 0.5:
                if inp.unique().nelement() > self.length // 2:
                    # too many unique digits, re-sample
                    continue
            # figure out if this generated example is train or test based on its hash
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = (
                "test" if h % 4 == 0 else "train"
            )  # designate 25% of examples as test
            if inp_split == self.split:
                break  # ok

        # solve the task: i.e. sort
        sol = torch.sort(inp)[0]

        # concatenate the problem specification and the solution
        cat = torch.cat((inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        return x, y


@pytest.fixture
def sort_datasets():
    """Create train and test sort datasets."""
    train_dataset = SortDataset("train")
    test_dataset = SortDataset("test")
    return train_dataset, test_dataset


@pytest.fixture
def decoder_config(sort_datasets):
    """Create configuration for decoder model."""
    train_dataset, _ = sort_datasets
    return TransformerConfig(
        vocab_size=train_dataset.get_vocab_size(),
        block_size=train_dataset.get_block_size(),
        n_layer=3,
        n_embd=48,
        n_head=3,
    )


@pytest.fixture
def trainer_config():
    """Create training configuration."""
    return TrainerConfig(
        learning_rate=5e-4,  # the model we're using is so small that we can go a bit faster
        max_iters=500,
        batch_size=64,  # using default from dataclass
        num_workers=0,
        device="cpu",
    )


def test_decoder(deterministic_setup, sort_datasets, decoder_config, trainer_config):
    """Test decoder model training and generation."""
    train_dataset, test_dataset = sort_datasets

    # Test dataset functionality
    x, y = train_dataset[0]
    print(x.shape, y.shape)
    assert x.shape[0] == 11  # input length
    assert y.shape[0] == 11  # output length

    # Create and train model
    model = Decoder(decoder_config)
    trainer = Trainer(trainer_config, model, train_dataset)

    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
            )

    trainer.run()

    # Test generation
    model.eval()
    input_seq = torch.tensor([[2, 1, 1, 0, 1, 2]])
    expected_output = torch.tensor([[0, 1, 1, 1, 2, 2]])
    generated = generate(model, input_seq, max_new_tokens=6)

    assert torch.allclose(generated[:, -6:], expected_output)
