from transformers import BertConfig, BertModel
from transformers import BertTokenizer

# Building the config
config = BertConfig()
checkpoint = "bert-base-uncased"

# Building the model from the config
model = BertModel(config) # random weights
pretrained_model = BertModel.from_pretrained(checkpoint) # pretrained weights


tokenizer = BertTokenizer.from_pretrained(checkpoint)

example_sentence = "Using a Transformer network is simple"


tokens = tokenizer.tokenize(example_sentence)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

print(tokenizer(example_sentence))
print(tokenizer.decode([101,102]))