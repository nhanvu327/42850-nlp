import os
import lightning as L
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk


class CNNDailyMailDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        data_dir="data/cnn_dailymail",
        max_token_length: int = 1024,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.data_dir = data_dir

    def prepare_data(self):
        full_dataset = load_dataset("cnn_dailymail", "3.0.0")

        train_ds_path = os.path.join(self.data_dir, "train_tokenized")
        validation_ds_path = os.path.join(self.data_dir, "validation_tokenized")
        test_ds_path = os.path.join(self.data_dir, "test_tokenized")

        if not os.path.exists(train_ds_path):
            train_tokenized_dataset = full_dataset["train"].map(
                self._tokenize,
                batched=True,
                remove_columns=["article", "highlights", "id"],
            )
            train_tokenized_dataset.save_to_disk(train_ds_path)

        if not os.path.exists(validation_ds_path):
            validation_tokenized_dataset = full_dataset["validation"].map(
                self._tokenize,
                batched=True,
                remove_columns=["article", "highlights", "id"],
            )
            validation_tokenized_dataset.save_to_disk(validation_ds_path)

        if not os.path.exists(test_ds_path):
            test_tokenized_dataset = full_dataset["test"].map(
                self._tokenize,
                batched=True,
                remove_columns=["article", "highlights", "id"],
            )
            test_tokenized_dataset.save_to_disk(test_ds_path)

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.train_dataset = load_from_disk(
                os.path.join(self.data_dir, "train_tokenized")
            ).with_format("torch")
            self.val_dataset = load_from_disk(
                os.path.join(self.data_dir, "validation_tokenized")
            ).with_format("torch")

        if stage == "test":
            self.test_dataset = load_from_disk(
                os.path.join(self.data_dir, "test_tokenized")
            ).with_format("torch")

    def _tokenize(self, batch):
        encoding = self.tokenizer(
            batch["article"],
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )
        targets = self.tokenizer(
            batch["highlights"],
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
        )
        encoding["labels"] = targets["input_ids"]
        return encoding

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
