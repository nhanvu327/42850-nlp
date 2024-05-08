import optuna
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model

# Load the T5 tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-large")

# Tokenize the text and create input and target sequences
train_data = pd.read_csv("../../datasets/train_1k.csv")
val_data = pd.read_csv("../../datasets/validation_1k.csv")

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)


def preprocess_function(examples):
    inputs = examples["article"]
    targets = examples["highlight"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

def objective(trial):
    # Define the hyperparameter search space
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)
    lora_r = trial.suggest_int("lora_r", 4, 64)
    lora_alpha = trial.suggest_int("lora_alpha", 8, 128)
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.5)

    # Load the T5 model
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=lora_dropout,
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"output_{trial.number}",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
    )

    # Create the Trainer and fine-tune the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    # Evaluate the model and return the validation loss
    eval_results = trainer.evaluate()
    return eval_results["eval_loss"]

    # Use Rouge scores if necessary


# Create an Optuna study
study = optuna.create_study(direction="minimize")

# Optimize the hyperparameters
study.optimize(objective, n_trials=20)

# Print the best hyperparameters and best validation loss
print("Best hyperparameters:", study.best_params)
print("Best validation loss:", study.best_value)
