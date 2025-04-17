import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import sys
import urllib.request
sys.path.append(os.path.join(".."))

import torch
import torch.nn as nn
from ch4.gpt import GPT
import json
import numpy as np
from tqdm import tqdm

import tensorflow as tf


def download_and_load_gpt2(models_dir):
    
    model_size = "124M"

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    def _attempt_download(download_url):
        with urllib.request.urlopen(download_url) as response:
            # Get the total file size from headers, defaulting to 0 if not present
            file_size = int(response.headers.get("Content-Length", 0))

            # Check if file exists and has the same size
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return True  # Indicate success without re-downloading

            block_size = 1024  # 1 Kilobyte

            # Initialize the progress bar with total file size
            progress_bar_description = os.path.basename(download_url)
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                with open(destination, "wb") as file:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            return True

    try:
        if _attempt_download(url):
            return
    except (urllib.error.HTTPError, urllib.error.URLError):
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except urllib.error.HTTPError:
                pass

        # If we reach here, both attempts have failed
        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.position_embedding.weight = assign(gpt.position_embedding.weight, params["wpe"])
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformer_blocks[b].attention.W_q.weight = assign(
            gpt.transformer_blocks[b].attention.W_q.weight, q_w.T)
        gpt.transformer_blocks[b].attention.W_k.weight = assign(
            gpt.transformer_blocks[b].attention.W_k.weight, k_w.T)
        gpt.transformer_blocks[b].attention.W_v.weight = assign(
            gpt.transformer_blocks[b].attention.W_v.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.transformer_blocks[b].attention.W_q.bias = assign(
            gpt.transformer_blocks[b].attention.W_q.bias, q_b)
        gpt.transformer_blocks[b].attention.W_k.bias = assign(
            gpt.transformer_blocks[b].attention.W_k.bias, k_b)
        gpt.transformer_blocks[b].attention.W_v.bias = assign(
            gpt.transformer_blocks[b].attention.W_v.bias, v_b)

        gpt.transformer_blocks[b].attention.head_output_projection.weight = assign(
            gpt.transformer_blocks[b].attention.head_output_projection.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].attention.head_output_projection.bias = assign(
            gpt.transformer_blocks[b].attention.head_output_projection.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.transformer_blocks[b].mlp.layers[0].weight = assign(
            gpt.transformer_blocks[b].mlp.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformer_blocks[b].mlp.layers[0].bias = assign(
            gpt.transformer_blocks[b].mlp.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformer_blocks[b].mlp.layers[2].weight = assign(
            gpt.transformer_blocks[b].mlp.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].mlp.layers[2].bias = assign(
            gpt.transformer_blocks[b].mlp.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.transformer_blocks[b].ln1.weight = assign(
            gpt.transformer_blocks[b].ln1.weight,
            params["blocks"][b]["ln_1"]["g"])
        gpt.transformer_blocks[b].ln1.bias = assign(
            gpt.transformer_blocks[b].ln1.bias,
            params["blocks"][b]["ln_1"]["b"])
        gpt.transformer_blocks[b].ln2.weight = assign(
            gpt.transformer_blocks[b].ln2.weight,
            params["blocks"][b]["ln_2"]["g"])
        gpt.transformer_blocks[b].ln2.bias = assign(
            gpt.transformer_blocks[b].ln2.bias,
            params["blocks"][b]["ln_2"]["b"])

    gpt.output_head_ln.weight = assign(gpt.output_head_ln.weight, params["g"])
    gpt.output_head_ln.bias = assign(gpt.output_head_ln.bias, params["b"])
    gpt.output_head.weight = assign(gpt.output_head.weight, params["wte"])


def load_pretrained_gpt2_model(gpt_config):
    """ Returns OpenAI's pretrained GP2 weights """

    model = GPT(gpt_config)
    _, params = download_and_load_gpt2(models_dir="gpt2")
    load_weights_into_gpt(model, params)
    model.eval()

    return model


if __name__ == "__main__":

    # GPT config for GPT2-small
    gpt_config = {
        "vocab_size": 50257, # Vocabulary size
        "context_length": 1024, # Context length
        "d_e": 768, # Embedding dimension
        "n_heads": 12, # Number of attention heads
        "n_layers": 12, # Number of transformer block layers
        "p_dropout": 0.1, # Dropout rate
        "qkv_bias": True, # Query/Key/Value bias in linear layers
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(gpt_config)

    # Download GPT2 OpenAI weights
    settings, params = download_and_load_gpt2(models_dir="gpt2")

    # Load them into our architecture
    load_weights_into_gpt(model, params)


    model.to(device)
    model.eval()
    model_output = model.generate("A big advantage of having a car is", max_new_tokens=20, decode=True, device=device)
    print(model_output)
