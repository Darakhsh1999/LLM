import os.path as osp
import torch
import json
import tiktoken
import pandas as pd
from torch.utils.data import Dataset
from pprint import pprint




class MovieData(Dataset):


    def __init__(self, data_dir, pad_token_id=50256):
        
        # Tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # Movies
        self.movies = pd.read_csv(osp.join(data_dir,"movies_overview.csv")) 
        self.movies["target"] = self.movies["genre_ids"].apply(self.string_to_list)
        self.movies.dropna(inplace=True)
        self.find_max_sequence()

        # Genres
        self.genres = pd.read_csv(osp.join(data_dir,"movies_genres.csv")) 
        self.create_genres()

        self.n_classes = len(self.genre_mapping)

        self.encoded_text = [self.tokenizer.encode(text) for text in self.movies["overview"]]
        self.encoded_text = [(text + [pad_token_id] * (self.max_sequence_length - len(text))) for text in self.encoded_text]


    def find_max_sequence(self):
        max_len = 0
        for row_idx, row in self.movies.iterrows():
            max_len = max(max_len, len(self.tokenizer.encode(row["overview"])))
        self.max_sequence_length = max_len
    
    def string_to_list(self, _string):
        try:
            return json.loads(_string)[0]
        except:
            return pd.NaT

    def create_genres(self):
        """ Create genre dictionary """
        mapping = {}
        for row_idx, row in self.genres.iterrows():
            mapping[row["id"]] = row["name"]
        self.genre_mapping = mapping


    def __getitem__(self, index):
        x = torch.tensor(self.encoded_text[index], dtype=torch.long)
        y = torch.tensor(self.movies.iloc[index]["target"], dtype=torch.long)
        return (x,y)

    def __len__(self):
        return len(self.movies)


if __name__ == "__main__":
    
    data_path = osp.join(".")
    data = MovieData(data_dir=data_path)
    print(len(data))
