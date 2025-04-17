import os.path as osp
import torch
import json
import tiktoken
import pandas as pd
from torch.utils.data import Dataset




class MovieData(Dataset):


    def __init__(self, data_dir, pad_token_id=50256, verbose=False):
        
        # Tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.categories = ["Drama", "Comedy", "Action", "Horror"]
        self.categories_idx = [18,35,28,27]

        # Movies
        self.movies = pd.read_csv(osp.join(data_dir,"movies_overview.csv")) 
        print(len(self.movies))
        self.movies["target"] = self.movies["genre_ids"].apply(self.string_to_list)
        self.movies.dropna(inplace=True)
        self.movies = self.movies[self.movies["target"].isin(self.categories_idx)]

        # Genres
        self.genres = pd.read_csv(osp.join(data_dir,"movies_genres.csv")) 
        self.create_genres()

        self.n_classes = len(self.genre_mapping)

        self.find_max_sequence()
        self.encoded_text = [self.tokenizer.encode(text) for text in self.movies["overview"]]
        self.encoded_text = [(text + [pad_token_id] * (self.max_sequence_length - len(text))) for text in self.encoded_text]

        if verbose:
            print(f"Max sequence len: {self.max_sequence_length}, n_samples: {len(self.movies)}, n_classes {len(self.genre_mapping)}")
            print(self.genre_mapping)
            print(self.genre_idx_mapping)


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
        index_mapping = {}
        idx = 0
        for row_idx, row in self.genres.iterrows():
            if row["id"] in self.categories_idx:
                mapping[row["id"]] = row["name"]
                index_mapping[row["id"]] = idx
                idx += 1
        self.genre_mapping = mapping
        self.genre_idx_mapping = index_mapping


    def __getitem__(self, index):
        x = torch.tensor(self.encoded_text[index], dtype=torch.long)
        y = self.movies.iloc[index]["target"]
        y = torch.tensor(self.genre_idx_mapping[y], dtype=torch.long)
        return (x,y)

    def __len__(self):
        return len(self.movies)


if __name__ == "__main__":
    
    data_path = osp.join(".")
    data = MovieData(data_dir=data_path, verbose=True)
    # print(data[0])
    # print(data.max_sequence_length, len(data))
