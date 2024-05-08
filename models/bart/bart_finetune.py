import lightning as L
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from datamodule.cnn_dailymail import CNNDailyMailDataModule
from models.bart.bart_module import BartModule
from transformers import AutoTokenizer

if __name__ == "__main__":

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/bart",
        filename="bart-cnndailymail-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    trainer = L.Trainer(
        callbacks=[
            # ModelCheckpoint(
            #     monitor="val_rouge1",
            #     dirpath="checkpoints/bart",
            #     filename="bart-cnndailymail-{epoch:02d}-{val_rouge1:.4f}",
            #     save_top_k=3,
            #     mode="max",
            #     save_last=False,
            # ),
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                mode="min",
            ),
            checkpoint_callback,
        ],
        # fast_dev_run=True,
        max_epochs=50,
    )

    dm = CNNDailyMailDataModule(
        tokenizer=AutoTokenizer.from_pretrained("facebook/bart-large-xsum"),
        batch_size=8,
        num_workers=0,
    )
    model = BartModule()

    trainer.fit(model=model, datamodule=dm)
    best_model_path = checkpoint_callback.best_model_path

    trainer.test(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm, ckpt_path=best_model_path)
