import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# Suppress tensorflow warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
import tensorflow as tf

max_length = 128
raw_datasets = load_dataset("kde4", lang1="en", lang2="sv")

split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
split_datasets["validation"] = split_datasets.pop("test")

# Data example
print("Data example \n",split_datasets["train"][1]["translation"])


# Load model and tokenizer
model_checkpoint = "Helsinki-NLP/opus-mt-en-sv"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Preprocess helper function
def preprocess_function(examples, max_length=128):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["sv"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
    return model_inputs

# Preprocess the dataset
tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)

print(tokenized_datasets)

# Collate function
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Improved BLEU metric
metric = evaluate.load("sacrebleu")


# Metric function
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}



args = Seq2SeqTrainingArguments(
    f"marian-finetuned-kde4-en-to-sv",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
)



# Trainer function
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print(trainer.evaluate(max_length=max_length)) # Before finetuning
trainer.train()
print(trainer.evaluate(max_length=max_length)) # After finetuning