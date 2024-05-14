from typing import Optional, Tuple, Union
import lightning as L

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
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

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")

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
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(
            outputs["logits"].argmax(dim=-1), skip_special_tokens=True
        )
        metrics = {
            f"{split}_loss": outputs["loss"],
        }

        self.rouge.update(decoded_preds, decoded_labels)
        self.b1.update(decoded_preds, decoded_labels)
        self.b2.update(decoded_preds, decoded_labels)
        self.b3.update(decoded_preds, decoded_labels)
        self.b4.update(decoded_preds, decoded_labels)

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
        self.log("train_loss", outputs["loss"], on_step=False, on_epoch=True, sync_dist=True)
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        _, metrics = self._call_and_compute_metrics(batch, "val")
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

        return metrics

    def on_validation_epoch_end(self):
        rougue = self.rouge.compute()
        log_dict = {f"val_{k}": v.to("cuda") for k, v in rougue.items()}
        log_dict["val_bleu1"] = self.b1.compute().to("cuda")
        log_dict["val_bleu2"] = self.b2.compute().to("cuda")
        log_dict["val_bleu3"] = self.b3.compute().to("cuda")
        log_dict["val_bleu4"] = self.b4.compute().to("cuda")
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.rouge.reset()
        self.b1.reset()
        self.b2.reset()
        self.b3.reset()
        self.b4.reset()

    def test_step(self, batch, batch_idx):
        _, metrics = self._call_and_compute_metrics(batch, "test")
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

        return metrics

    def on_test_epoch_end(self):
        rougue = self.rouge.compute()
        log_dict = {f"test_{k}": v.to("cuda") for k, v in rougue.items()}
        log_dict["test_bleu1"] = self.b1.compute().to("cuda")
        log_dict["test_bleu2"] = self.b2.compute().to("cuda")
        log_dict["test_bleu3"] = self.b3.compute().to("cuda")
        log_dict["test_bleu4"] = self.b4.compute().to("cuda")
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )
        self.rouge.reset()
        self.b1.reset()
        self.b2.reset()
        self.b3.reset()
        self.b4.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-5,
        )

        return optimizer
