from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

print(raw_datasets["train"][0],"\n")
print(raw_datasets["train"][3:5])


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])


inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(inputs)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print("len of samples",[len(x) for x in samples["input_ids"]])

batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})

print(tokenized_datasets["train"][0],"\n")