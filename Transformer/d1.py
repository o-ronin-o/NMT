"""
FULLY FIXED TRAINING SCRIPT
- Correct masking with boolean tensors
- Proper beam search with length constraints
- Fixed type errors
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
from tqdm import tqdm
import os
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from collections import Counter
# ============================================================================
# PATHS & SETUP
# ============================================================================
DATASET_PATH = "parallel_en_fr_corpus"
EN_TOKENIZER_PATH = "tokenizer_en"
FR_TOKENIZER_PATH = "tokenizer_fr"
CHECKPOINT_DIR = "checkpoints_fixed"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# LOAD DATASETS
# ============================================================================
print("\nLoading datasets...")
train_dataset = Dataset.from_file(os.path.join(DATASET_PATH, "train", "dataset.arrow"))
val_dataset   = Dataset.from_file(os.path.join(DATASET_PATH, "validation", "dataset.arrow"))
test_dataset  = Dataset.from_file(os.path.join(DATASET_PATH, "test", "dataset.arrow"))
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# ============================================================================
# LOAD TOKENIZERS
# ============================================================================
print("\nLoading tokenizers...")
en_tokenizer = PreTrainedTokenizerFast.from_pretrained(EN_TOKENIZER_PATH)
fr_tokenizer = PreTrainedTokenizerFast.from_pretrained(FR_TOKENIZER_PATH)

src_pad_id = en_tokenizer.pad_token_id
tgt_pad_id = fr_tokenizer.pad_token_id
bos_token_id = fr_tokenizer.bos_token_id
eos_token_id = fr_tokenizer.eos_token_id

src_vocab_size = len(en_tokenizer)
tgt_vocab_size = len(fr_tokenizer)

print(f"Source vocab: {src_vocab_size}, Target vocab: {tgt_vocab_size}")
print(f"PAD: {tgt_pad_id}, BOS: {bos_token_id}, EOS: {eos_token_id}")

# ============================================================================
# TOKENIZATION
# ============================================================================
MAX_LEN = 32

def tokenize_pair(example):
    src = en_tokenizer(
        example['text_en'],
        add_special_tokens=True,
        max_length=MAX_LEN,
        truncation=True,
        padding=False
    )['input_ids']
    tgt = fr_tokenizer(
        example['text_fr'],
        add_special_tokens=True,
        max_length=MAX_LEN,
        truncation=True,
        padding=False
    )['input_ids']
    return {'src': src, 'tgt': tgt}

print("\nTokenizing...")
train_dataset = train_dataset.map(tokenize_pair, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize_pair, remove_columns=val_dataset.column_names)
test_dataset = test_dataset.map(tokenize_pair, remove_columns=test_dataset.column_names)

train_dataset.set_format('torch', columns=['src', 'tgt'])
val_dataset.set_format('torch', columns=['src', 'tgt'])
test_dataset.set_format('torch', columns=['src', 'tgt'])

# Filter out empty or too-short targets
print("\nFiltering short sequences...")
original_len = len(train_dataset)
train_dataset = train_dataset.filter(lambda x: len(x['tgt']) > 3)
print(f"Removed {original_len - len(train_dataset)} very short sequences")

# ============================================================================
# COLLATE FUNCTION
# ============================================================================
def collate_fn(batch):
    src_seqs = [item['src'] for item in batch]
    tgt_seqs = [item['tgt'] for item in batch]

    src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=src_pad_id)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=tgt_pad_id)

    # Source mask - boolean type
    src_mask = (src_padded != src_pad_id).unsqueeze(1).unsqueeze(2)

    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_mask': src_mask,
    }

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ============================================================================
# BUILD MODEL
# ============================================================================
from model import build_transformer

d_model = 256
d_ff = 1024
h = 8
N = 3
dropout = 0.1

print(f"\nModel: d_model={d_model}, d_ff={d_ff}, heads={h}, layers={N}")

model = build_transformer(
    src_vocab_size, tgt_vocab_size,
    MAX_LEN, MAX_LEN,
    d_model, N, h, dropout, d_ff
)
model.to(device)

# ============================================================================
# TRAINING SETUP
# ============================================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_id,label_smoothing=0.1)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)


# ============================================================================
# GREEDY DECODE (FOR DEBUGGING ONLY)
# ============================================================================
# def greedy_decode(model, src_tokens, src_mask, max_len=32):
#     model.eval()
#     with torch.no_grad():
#         encoder_output = model.encode(src_tokens, src_mask)
        
#         seq = [bos_token_id]
        
#         for _ in range(max_len):
#             tgt_tensor = torch.tensor([seq], device=device)
            
#             tgt_mask = (tgt_tensor != tgt_pad_id).unsqueeze(1).unsqueeze(2)
#             seq_len = len(seq)
#             causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
#             causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
#             tgt_mask = tgt_mask.bool() & causal_mask

#             decoder_output = model.decode(tgt_tensor, encoder_output, tgt_mask, src_mask)
#             logits = model.project(decoder_output)
            
#             # Pick the SINGLE most likely next token
#             next_token = logits[0, -1, :].argmax(dim=-1).item()
            
#             if next_token == eos_token_id:
#                 break
                
#             seq.append(next_token)
            
#     if seq and seq[0] == bos_token_id:
#         seq = seq[1:]
#     return seq

# # ============================================================================
# # DEBUG FUNCTION (UPDATED TO USE GREEDY)
# # ============================================================================
# def debug_prediction(model, epoch):
#     print("\n" + "="*70)
#     print(f"DEBUG (Epoch {epoch}) - USING GREEDY DECODING")
#     print("="*70)
    
#     test_sentences = ["i am a cat", "i am happy", "i am a student"]
    
#     for sent in test_sentences:
#         model.eval()
#         tokens = en_tokenizer(sent, add_special_tokens=True, return_tensors='pt')
#         src_ids = tokens['input_ids'].to(device)
#         src_mask = (src_ids != src_pad_id).unsqueeze(1).unsqueeze(2)
        
#         # USE GREEDY HERE TO SEE RAW MODEL BEHAVIOR
#         pred_ids = greedy_decode(model, src_ids, src_mask, max_len=MAX_LEN)
#         translation = fr_tokenizer.decode(pred_ids, skip_special_tokens=True)
        
#         print(f"'{sent}' → '{translation}'")
#     print("="*70)

# ============================================================================
# BEAM SEARCH (FIXED)
# ============================================================================
def beam_search(model, src_tokens, src_mask, beam_width=4, max_len=32, min_len=5):
    model.eval()
    with torch.no_grad():
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
            
            # Build target mask - convert to boolean
            tgt_mask = (tgt_tensor != tgt_pad_id).unsqueeze(1).unsqueeze(2)
            seq_len = len(seq)
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            # Convert tgt_mask to boolean for & operation
            tgt_mask = tgt_mask.bool() & causal_mask

            with torch.no_grad():
                decoder_output = model.decode(tgt_tensor, encoder_output, tgt_mask, src_mask)
                logits = model.project(decoder_output)
                next_logits = logits[0, -1, :]
                next_logits = next_logits / 1.6

            log_probs = torch.log_softmax(next_logits, dim=-1)

            # Forbid EOS before minimum length
            if len(seq) <= min_len:
                log_probs[eos_token_id] = -float('inf')

            
       
            top_log_probs, top_indices = torch.topk(log_probs, beam_width)

            for k in range(beam_width):
                token_id = top_indices[k].item()
                log_prob = top_log_probs[k].item()
                new_score = score + log_prob
                new_seq = seq + [token_id]
                new_beams.append((new_score, new_seq))

        if not new_beams:
            break

        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]

        if all(seq[-1] == eos_token_id for _, seq in beams):
            break

    for score, seq in beams:
        if seq[-1] != eos_token_id:
            finished.append((score, seq))

    if not finished:
        finished = beams

    best_seq = max(finished, key=lambda x: x[0])[1]
    if best_seq and best_seq[0] == bos_token_id:
        best_seq = best_seq[1:]
    if best_seq and best_seq[-1] == eos_token_id:
        best_seq = best_seq[:-1]
    return best_seq if best_seq else [bos_token_id]

# ============================================================================
# TRANSLATION FUNCTION
# ============================================================================
def translate_sentence(model, sentence, min_len=5):
    model.eval()
    tokens = en_tokenizer(sentence, add_special_tokens=True, return_tensors='pt')
    src_ids = tokens['input_ids'].to(device)
    src_mask = (src_ids != src_pad_id).unsqueeze(1).unsqueeze(2)
    pred_ids = beam_search(model, src_ids, src_mask, min_len=min_len)
    return fr_tokenizer.decode(pred_ids, skip_special_tokens=True)

# ============================================================================
# DEBUG FUNCTION
# ============================================================================
# def debug_prediction(model, epoch):
#     print("\n" + "="*70)
#     print(f"DEBUG (Epoch {epoch}) - USING GREEDY DECODING")
#     print("="*70)
    
#     test_sentences = ["i am a cat", "i am happy", "i am a student"]
    
#     for sent in test_sentences:
#         model.eval()
#         tokens = en_tokenizer(sent, add_special_tokens=True, return_tensors='pt')
#         src_ids = tokens['input_ids'].to(device)
#         src_mask = (src_ids != src_pad_id).unsqueeze(1).unsqueeze(2)
        
#         # USE GREEDY HERE TO SEE RAW MODEL BEHAVIOR
#         pred_ids = greedy_decode(model, src_ids, src_mask, max_len=MAX_LEN)
#         translation = fr_tokenizer.decode(pred_ids, skip_special_tokens=True)
        
#         print(f"'{sent}' → '{translation}'")
#     print("="*70)

def debug_prediction(model, epoch):
    print("\n" + "="*70)
    print(f"DEBUG (Epoch {epoch})")
    print("="*70)
    
    test_sentences = ["i am a cat", "i am happy", "i am a student"]
    
    for sent in test_sentences:
        translation = translate_sentence(model, sent, min_len=3)
        print(f"'{sent}' → '{translation}'")
    print("="*70)

# ============================================================================
# BLEU COMPUTATION
# ============================================================================
def compute_bleu(model, dataloader, max_samples=100):
    model.eval()
    references = []
    hypotheses = []

    for i, batch in enumerate(tqdm(dataloader, desc="BLEU")):
        if i >= max_samples:
            break
            
        src = batch['src'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt = batch['tgt']

        for j in range(src.size(0)):
            src_tokens = src[j:j+1]
            src_mask_j = src_mask[j:j+1]
            
            pred_ids = beam_search(model, src_tokens, src_mask_j, min_len=3)
            
            ref_ids = tgt[j].tolist()
            if ref_ids and ref_ids[0] == bos_token_id:
                ref_ids = ref_ids[1:]
            if ref_ids and ref_ids[-1] == eos_token_id:
                ref_ids = ref_ids[:-1]
            
            if len(pred_ids) > 0 and len(ref_ids) > 0:
                references.append([ref_ids])
                hypotheses.append(pred_ids)

    if not hypotheses:
        return 0.0
        
    smooth = SmoothingFunction().method1
    return corpus_bleu(references, hypotheses, smoothing_function=smooth)

# ============================================================================
# TRAINING LOOP
# ============================================================================
num_epochs = 30
best_val_loss = float('inf')
patience_counter = 0

print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0
    train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
    for batch in train_progress:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_mask = batch['src_mask'].to(device)

        # Shift target for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        noise_mask = torch.rand_like(tgt_input.float()) < 0.10
        tgt_input = torch.where(noise_mask, torch.tensor(tgt_pad_id, device=device), tgt_input)
        # Build target mask - convert to boolean
        tgt_len = tgt_input.size(1)
        tgt_pad_mask = (tgt_input != tgt_pad_id).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        # Convert tgt_pad_mask to boolean for & operation
        tgt_mask = tgt_pad_mask.bool() & causal_mask

        # Forward pass
        encoder_output = model.encode(src, src_mask)
        decoder_output = model.decode(tgt_input, encoder_output, tgt_mask, src_mask)
        logits = model.project(decoder_output)

        # Loss
        loss = criterion(logits.contiguous().view(-1, tgt_vocab_size), 
                        tgt_output.contiguous().view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_train_loss += loss.item()
        train_progress.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Validation'):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_mask = batch['src_mask'].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_len = tgt_input.size(1)
            tgt_pad_mask = (tgt_input != tgt_pad_id).unsqueeze(1).unsqueeze(2)
            causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            tgt_mask = tgt_pad_mask.bool() & causal_mask

            encoder_output = model.encode(src, src_mask)
            decoder_output = model.decode(tgt_input, encoder_output, tgt_mask, src_mask)
            logits = model.project(decoder_output)

            loss = criterion(logits.contiguous().view(-1, tgt_vocab_size),
                           tgt_output.contiguous().view(-1))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    scheduler.step(avg_val_loss)

    print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Debug every 5 epochs
    if (epoch + 1) % 3 == 0:
        debug_prediction(model, epoch + 1)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
        print("  -> New best model saved!")
    else:
        patience_counter += 1
        if patience_counter >= 5:
            print(f"Early stopping after {epoch+1} epochs")
            break

# ============================================================================
# FINAL EVALUATION
# ============================================================================
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

# Load best model
model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"), map_location=device))
model.eval()

# Test loss
total_test_loss = 0
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test Loss"):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_mask = batch['src_mask'].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        tgt_len = tgt_input.size(1)
        tgt_pad_mask = (tgt_input != tgt_pad_id).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_pad_mask.bool() & causal_mask

        encoder_output = model.encode(src, src_mask)
        decoder_output = model.decode(tgt_input, encoder_output, tgt_mask, src_mask)
        logits = model.project(decoder_output)

        loss = criterion(logits.contiguous().view(-1, tgt_vocab_size),
                        tgt_output.contiguous().view(-1))
        total_test_loss += loss.item()

print(f"\nTest Loss: {total_test_loss / len(test_loader):.4f}")

# BLEU score
print("\nComputing BLEU-4 score...")
bleu = compute_bleu(model, test_loader, max_samples=200)
print(f"BLEU-4 Score: {bleu:.4f}")

# Final translations
print("\n" + "="*70)
print("SAMPLE TRANSLATIONS")
print("="*70)
test_examples = [
    "i am a cat",
    "i am happy",
    "i am a student",
    "he is my friend",
    "she is beautiful"
]

for sentence in test_examples:
    translation = translate_sentence(model, sentence, min_len=3)
    print(f"EN: {sentence}")
    print(f"FR: {translation}\n")

print("="*70)
print("TRAINING COMPLETE")
print("="*70)