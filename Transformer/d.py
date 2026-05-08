from tokenizers import Tokenizer, models, trainers, processors, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
import os

def create_fixed_tokenizer(texts, vocab_size=8000):
    """
    Create a BPE tokenizer from a list of texts.
    
    Args:
        texts: List of strings to train on
        vocab_size: Size of the vocabulary (default 8000)
    
    Returns:
        PreTrainedTokenizerFast ready to use
    """
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    # CRITICAL: Define DISTINCT special tokens with specific IDs
    special_tokens = [
        "<pad>",  # ID will be 0
        "<s>",    # BOS, ID will be 1
        "</s>",   # EOS, ID will be 2
        "<unk>",  # ID will be 3
    ]
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2,  # Ignore rare tokens
    )
    
    # Train on the provided texts
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    # Add BOS/EOS during tokenization (post-processing)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )
    
    # Wrap for HuggingFace compatibility
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        padding_side="right",  # Important for transformer
    )

# -------------------------------
# Load your existing dataset
# -------------------------------
DATASET_PATH = "parallel_en_fr_corpus"

# Load your existing splits
train_dataset = Dataset.from_file(os.path.join(DATASET_PATH, "train", "dataset.arrow"))
val_dataset = Dataset.from_file(os.path.join(DATASET_PATH, "validation", "dataset.arrow"))
test_dataset = Dataset.from_file(os.path.join(DATASET_PATH, "test", "dataset.arrow"))

# ✅ FIX: Convert to lists properly
english_texts = list(train_dataset['text_en']) + list(val_dataset['text_en']) + list(test_dataset['text_en'])
french_texts = list(train_dataset['text_fr']) + list(val_dataset['text_fr']) + list(test_dataset['text_fr'])

print(f"Training English tokenizer on {len(english_texts)} sentences")
print(f"Training French tokenizer on {len(french_texts)} sentences")

# Optional: Show a few examples
print("\nSample English texts:")
for i in range(min(3, len(english_texts))):
    print(f"  {english_texts[i]}")

print("\nSample French texts:")
for i in range(min(3, len(french_texts))):
    print(f"  {french_texts[i]}")

# -------------------------------
# Create and save tokenizers
# -------------------------------
print("\nCreating English tokenizer...")
en_tokenizer = create_fixed_tokenizer(english_texts, vocab_size=8000)

print("Creating French tokenizer...")
fr_tokenizer = create_fixed_tokenizer(french_texts, vocab_size=8000)

# Save to new directories
EN_TOKENIZER_PATH = "tokenizer_en_v2"
FR_TOKENIZER_PATH = "tokenizer_fr_v2"

en_tokenizer.save_pretrained(EN_TOKENIZER_PATH)
fr_tokenizer.save_pretrained(FR_TOKENIZER_PATH)

print(f"\n✅ English tokenizer saved to {EN_TOKENIZER_PATH}")
print(f"✅ French tokenizer saved to {FR_TOKENIZER_PATH}")

# -------------------------------
# Verify the tokenizers
# -------------------------------
print("\n=== Tokenizer Verification ===")
print(f"English vocab size: {len(en_tokenizer)}")
print(f"French vocab size: {len(fr_tokenizer)}")
print(f"English pad_token_id: {en_tokenizer.pad_token_id}")
print(f"French pad_token_id: {fr_tokenizer.pad_token_id}")
print(f"English bos_token_id: {en_tokenizer.bos_token_id}")
print(f"French bos_token_id: {fr_tokenizer.bos_token_id}")
print(f"English eos_token_id: {en_tokenizer.eos_token_id}")
print(f"French eos_token_id: {fr_tokenizer.eos_token_id}")

# Check that pad and eos are different
if fr_tokenizer.pad_token_id == fr_tokenizer.eos_token_id:
    print("❌ CRITICAL: PAD and EOS tokens are the SAME!")
else:
    print("✅ PAD and EOS tokens are different (good!)")

# Test tokenization
test_sentence = "i am a cat"
test_tokens = en_tokenizer(test_sentence, add_special_tokens=True)
print(f"\nTest English: '{test_sentence}'")
print(f"Token IDs: {test_tokens['input_ids']}")
print(f"Decoded: {en_tokenizer.decode(test_tokens['input_ids'])}")

test_french = "je suis un chat"
test_tokens_fr = fr_tokenizer(test_french, add_special_tokens=True)
print(f"\nTest French: '{test_french}'")
print(f"Token IDs: {test_tokens_fr['input_ids']}")
print(f"Decoded: {fr_tokenizer.decode(test_tokens_fr['input_ids'])}")

# Test BPE subword splitting
rare_word = "untrustworthy"
tokens = en_tokenizer(rare_word, add_special_tokens=False)
print(f"\nBPE test: '{rare_word}' → {en_tokenizer.tokenize(rare_word)}")
print(f"  Token IDs: {tokens['input_ids']}")

print("\n✓ Tokenizers are ready to use!")
print(f"  Use them in your training script with:")
print(f"  en_tokenizer = PreTrainedTokenizerFast.from_pretrained('{EN_TOKENIZER_PATH}')")
print(f"  fr_tokenizer = PreTrainedTokenizerFast.from_pretrained('{FR_TOKENIZER_PATH}')")