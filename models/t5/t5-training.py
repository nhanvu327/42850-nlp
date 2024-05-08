from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, TaskType
import torch

# Load and preprocess your dataset
# Assuming your dataset is loaded into 'train_data' and 'val_data' variables

# Load the T5 tokenizer and model
model_name = "t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

train_data = pd.read_csv("../../datasets/train_1k.csv")
val_data = pd.read_csv("../../datasets/validation_1k.csv")

# Tokenize the text and create input and target sequences
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

def preprocess_function(examples):
    inputs = examples["article"]
    targets = examples["highlights"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
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

# Save the fine-tuned model
model.save_pretrained("nlp_uts_fine_tuned_t5_large_v2")