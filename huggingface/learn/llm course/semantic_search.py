import torch
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModel

issues_dataset = load_dataset("lewtun/github-issues", split="train")

# Filter out issues that are pull requests and without comments
issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)


# Remove non-relevant columns
columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)


issues_dataset.set_format("pandas")
df = issues_dataset[:]
comments_df = df.explode("comments", ignore_index=True)


# Cast back to Dataset
comments_dataset = Dataset.from_pandas(comments_df)


# Add a new column with the length of each comment
comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)

# Filter out short comments
comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)


# Concatenate the title, body, and comments into a single text field
def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }
comments_dataset = comments_dataset.map(concatenate_text)


# Retrieve embedding model
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
device = torch.device("cuda")
model.to(device)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


# Performs model inference on a list of texts and returns their embeddings
def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


# Embedd entire dataset
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)


# Create FAISS index for the dataset
embeddings_dataset.add_faiss_index(column="embeddings")


# Perform a query search
question = "How many parameters does GPT2 have?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()


# Perform nearest neighbor search
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)


# Convert the results to a DataFrame for better readability
samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)


# Print top results
for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()