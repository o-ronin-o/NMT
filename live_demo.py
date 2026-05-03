
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
from tqdm import tqdm
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import numpy as np

CHECKPOINT_DIR = "checkpoints"


# -------------------------------
# Paths & Setup
# -------------------------------
MAX_LEN = 64 
DATASET_PATH = "parallel_en_fr_corpus"
EN_TOKENIZER_PATH = "tokenizer_en"
FR_TOKENIZER_PATH = "tokenizer_fr"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
ATTENTION_VIZ_DIR = "attention_viz"
os.makedirs(ATTENTION_VIZ_DIR, exist_ok=True)


# -------------------------------
# Load Datasets (directly from .arrow)
# -------------------------------
print("Loading datasets...")
train_dataset = Dataset.from_file(os.path.join(DATASET_PATH, "train", "dataset.arrow"))
val_dataset   = Dataset.from_file(os.path.join(DATASET_PATH, "validation", "dataset.arrow"))
test_dataset  = Dataset.from_file(os.path.join(DATASET_PATH, "test", "dataset.arrow"))
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
print("Train columns:", train_dataset.column_names)

# -------------------------------
# Load Tokenizers
# -------------------------------
en_tokenizer = PreTrainedTokenizerFast.from_pretrained(EN_TOKENIZER_PATH)
fr_tokenizer = PreTrainedTokenizerFast.from_pretrained(FR_TOKENIZER_PATH)

# Separate source and target pad tokens (critical)
src_pad_id = en_tokenizer.pad_token_id
tgt_pad_id = fr_tokenizer.pad_token_id

bos_token_id = fr_tokenizer.bos_token_id
eos_token_id = fr_tokenizer.eos_token_id

src_vocab_size = len(en_tokenizer)
tgt_vocab_size = len(fr_tokenizer)
print(f"Source vocab size: {src_vocab_size}")
print(f"Target vocab size: {tgt_vocab_size}")


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss']

from model import build_transformer   # Your model code (must store attention scores)

d_model = 32; d_ff = 128; h = 4; N = 3; dropout = 0.1
model = build_transformer(src_vocab_size, tgt_vocab_size,
                          MAX_LEN, MAX_LEN, d_model, N, h, dropout, d_ff)
model.to(device)



load_checkpoint(os.path.join(CHECKPOINT_DIR, "best_checkpoint.pt"), model)

def beam_search(model, src_tokens, src_mask, beam_width=4, max_len=64):
    model.eval()
    encoder_output = model.encode(src_tokens, src_mask)

    beams = [(0.0, [bos_token_id])]
    finished = []

    for _ in range(max_len):
        new_beams = []
        for score, seq in beams:
            if seq[-1] == eos_token_id:
                finished.append((score, seq))
                continue

            tgt_tensor = torch.tensor([seq], device=device)
            # Build target mask (padding + causal) using tgt_pad_id
            tgt_mask = (tgt_tensor != tgt_pad_id).unsqueeze(1).unsqueeze(2).bool()
            causal_mask = torch.tril(torch.ones((len(seq), len(seq)), device=device)).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            tgt_mask = tgt_mask & causal_mask

            decoder_output = model.decode(tgt_tensor, encoder_output, tgt_mask, src_mask)
            logits = model.project(decoder_output)               # (1, seq_len, vocab)
            next_logits = logits[0, -1, :]                       # last time step
            log_probs = torch.log_softmax(next_logits, dim=-1)

            top_log_probs, top_indices = torch.topk(log_probs, beam_width)
            for k in range(beam_width):
                new_score = score + top_log_probs[k].item()
                new_seq = seq + [top_indices[k].item()]
                new_beams.append((new_score, new_seq))

        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]
        if all(seq[-1] == eos_token_id for _, seq in beams):
            break

    for score, seq in beams:
        if seq[-1] != eos_token_id:
            finished.append((score, seq))

    best_seq = max(finished, key=lambda x: x[0])[1]
    if best_seq[0] == bos_token_id:
        best_seq = best_seq[1:]
    if best_seq[-1] == eos_token_id:
        best_seq = best_seq[:-1]
    return best_seq

def translate_sentence(model, sentence, en_tokenizer, fr_tokenizer, beam_width=4):
    tokens = en_tokenizer(sentence, add_special_tokens=True, return_tensors='pt')
    src_ids = tokens['input_ids'].to(device)
    src_mask = (src_ids != src_pad_id).unsqueeze(1).unsqueeze(2).bool()
    pred_ids = beam_search(model, src_ids, src_mask, beam_width)
    return fr_tokenizer.decode(pred_ids, skip_special_tokens=True)
def live_translate(sentence):
    return translate_sentence(model, sentence, en_tokenizer, fr_tokenizer, beam_width=4)



src_test = "i am a cat"
tokens = en_tokenizer(src_test, return_tensors='pt')
src_ids = tokens['input_ids'].to(device)
src_mask = (src_ids != src_pad_id).unsqueeze(1).unsqueeze(2)
enc_out = model.encode(src_ids, src_mask)
# greedy decoding for sanity
tgt_ids = [bos_token_id]
for _ in range(20):
    tgt_tensor = torch.tensor([tgt_ids], device=device)
    # build mask
    tgt_mask = (tgt_tensor != tgt_pad_id).unsqueeze(1).unsqueeze(2)
    causal = torch.tril(torch.ones((len(tgt_ids), len(tgt_ids)), device=device)).bool()
    causal = causal.unsqueeze(0).unsqueeze(0)
    tgt_mask = tgt_mask & causal
    out = model.decode(tgt_tensor, enc_out, tgt_mask, src_mask)
    logits = model.project(out)
    next_token = logits[0, -1].argmax().item()
    if next_token == eos_token_id:
        break
    tgt_ids.append(next_token)
print(f"Greedy translation: {fr_tokenizer.decode(tgt_ids[1:])}")