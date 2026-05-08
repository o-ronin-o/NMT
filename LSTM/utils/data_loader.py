import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast
from typing import Tuple, List, Dict
from datasets import load_from_disk

class FrEnNMTDataset(Dataset):
    """
    Custom Dataset for French to English Neural Machine Translation.
    Reads from a Hugging Face dataset split.
    """
    def __init__(self, hf_dataset_split, fr_tokenizer: PreTrainedTokenizerFast, 
                en_tokenizer: PreTrainedTokenizerFast, max_len: int = 32):
        
        self.dataset = hf_dataset_split
        self.fr_tokenizer = fr_tokenizer
        self.en_tokenizer = en_tokenizer
        self.max_len = max_len

        # Peek at the first item to determine how the columns are named
        sample = self.dataset[0]
        if 'translation' in sample:
            self.fr_key, self.en_key = 'fr', 'en'
            self.is_nested = True
        else:
            keys = list(sample.keys())
            self.fr_key = 'fr' if 'fr' in keys else keys[0]
            self.en_key = 'en' if 'en' in keys else keys[1]
            self.is_nested = False

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        row = self.dataset[idx]
        
        # 1. STRICT EXTRACTION: Force French text and English text
        if self.is_nested:
            fr_text = row['translation']['fr']
            en_text = row['translation']['en']
        else:
            # Fallback if the dataset keys are named differently
            keys = list(row.keys())
            en_key = 'en' if 'en' in keys else keys[0]
            fr_key = 'fr' if 'fr' in keys else keys[1]
            fr_text = row[fr_key]
            en_text = row[en_key]

        # 2. SOURCE = FRENCH
        src_ids = self.fr_tokenizer.encode(
            fr_text,
            add_special_tokens=True, 
            truncation=True,
            max_length=self.max_len,
        )

        # 3. TARGET = ENGLISH (Inject <sos> and <eos>)
        target_text_with_specials = f"{self.en_tokenizer.bos_token} {en_text} {self.en_tokenizer.eos_token}"
        tgt_ids = self.en_tokenizer.encode(
            target_text_with_specials,
            add_special_tokens=False, 
            truncation=True,
            max_length=self.max_len + 2, 
        )

        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids
        }
    
class PadCollate:
    def __init__(self, src_pad_idx: int, tgt_pad_idx: int):
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        src_ids_list = [torch.tensor(item['src_ids']) for item in batch]
        tgt_ids_list = [torch.tensor(item['tgt_ids']) for item in batch]

        padded_src = pad_sequence(src_ids_list, batch_first=True, padding_value=self.src_pad_idx)
        padded_tgt = pad_sequence(tgt_ids_list, batch_first=True, padding_value=self.tgt_pad_idx)

        return padded_src, padded_tgt


def get_dataloaders(corpus_dir: str, tokenizer_fr_path: str, tokenizer_en_path: str, 
                    batch_size: int = 32, max_len: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader, PreTrainedTokenizerFast, PreTrainedTokenizerFast]:
    
    # 1. Initialize Tokenizers
    fr_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_fr_path)
    en_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_en_path)

    # 2. Add NMT Special Tokens
    special_tokens_dict = {
        'bos_token': '<sos>',
        'eos_token': '<eos>',
        'pad_token': '<pad>',
        'unk_token': '<unk>'
    }
    fr_tokenizer.add_special_tokens(special_tokens_dict)
    en_tokenizer.add_special_tokens(special_tokens_dict)

    # 3. Define Collator
    pad_collate = PadCollate(src_pad_idx=fr_tokenizer.pad_token_id, tgt_pad_idx=en_tokenizer.pad_token_id)

    # 4. Load the Hugging Face Dataset from disk
    hf_dataset = load_from_disk(corpus_dir)
    datasets = {}
    
    for split in ['train', 'validation', 'test']:
        datasets[split] = FrEnNMTDataset(hf_dataset[split], fr_tokenizer, en_tokenizer, max_len=max_len)

    # 5. Define DataLoaders
    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(datasets['validation'], batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    return train_loader, val_loader, test_loader, fr_tokenizer, en_tokenizer