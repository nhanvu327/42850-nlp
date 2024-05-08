from typing import Optional, Tuple, Union
import lightning as L

import torch
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import Seq2SeqLMOutput
from peft import LoraConfig, TaskType, get_peft_model
from torchmetrics.text import ROUGEScore, BLEUScore


class BartModule(L.LightningModule):
    def __init__(self):
        super().__init__()

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-large-xsum",
        )

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            lora_dropout=0.1,
        )

        self.model = get_peft_model(base_model, peft_config)
        self.model.print_trainable_parameters()

        # Evaluation metrics
        self.rouge = ROUGEScore()
        self.b1 = BLEUScore(n_gram=1)
        self.b2 = BLEUScore(n_gram=2)
        self.b3 = BLEUScore(n_gram=3)
        self.b4 = BLEUScore(n_gram=4)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def _call_and_compute_metrics(self, batch, split):
        input_ids, attention_mask, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        outputs = self(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        metrics = {
            f"{split}_loss": outputs["loss"],
        }

        return outputs, metrics

    def training_step(self, batch, batch_idx):
        # input_ids, attention_mask, labels = self._make_tensor(batch)
        input_ids, attention_mask, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        outputs = self(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        self.log("train_loss", outputs["loss"], on_step=False, on_epoch=True)
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        _, metrics = self._call_and_compute_metrics(batch, "val")
        self.log_dict(metrics, on_step=False, on_epoch=True)

        return metrics

    def test_step(self, batch, batch_idx):
        _, metrics = self._call_and_compute_metrics(batch, "test")
        self.log_dict(metrics, on_step=False, on_epoch=True)

        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-5,
        )

        return optimizer
