import html
from pprint import pprint
from datasets import load_dataset

# # Local dataset
# data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
# squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
# print(squad_it_dataset)

# # Remote dataset
# url = "https://github.com/crux82/squad-it/raw/master/" # URL to hosted dataset
# data_files = {
#     "train": url + "SQuAD_it-train.json.gz",
#     "test": url + "SQuAD_it-test.json.gz",
# }
# squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
# print(squad_it_dataset)

# Load TSV data 
data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")


# Print some samples
drug_dataset = drug_dataset["train"].shuffle(seed=42).select(range(1000))

# Rename column
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)

# Remove rows with None in condition
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

def lowercase_condition(example):
    return {"condition": example["condition"].lower()}


# Lowercase the condition names
drug_dataset = drug_dataset.map(lowercase_condition)


def compute_review_length(example):
    return {"review_length": len(example["review"].split())}

# Create a new column with the review length
drug_dataset = drug_dataset.map(compute_review_length)

pprint(drug_dataset[0])

# Filter out reviews with less than 30 words
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset.num_rows)

# Remove parse html characters
drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})


# Same as above but using batching that speeds up execution
new_drug_dataset = drug_dataset.map(lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True)


# Split the dataset into train, validation, and test sets
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
drug_dataset_clean["test"] = drug_dataset["test"]
pprint(drug_dataset_clean)


# Save the dataset to disk
drug_dataset_clean.save_to_disk("drug-reviews")