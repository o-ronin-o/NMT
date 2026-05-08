import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

def beam_search_decode(model, src_tensor, src_mask, bos_token_id, eos_token_id, max_len, beam_width, device):
    """Executes Beam Search decoding for a single source sequence."""
    model.eval()
    
    with torch.no_grad():
        encoder_outputs, hidden_state = model.encoder(src_tensor)
        
        # Initialize beam: (cumulative_log_prob, sequence_of_token_ids, hidden_state)
        beam = [(0.0, [bos_token_id], hidden_state)]
        
        for step in range(max_len):
            new_beam = []
            
            for score, seq, h_state in beam:
                if seq[-1] == eos_token_id:
                    new_beam.append((score, seq, h_state))
                    continue
                    
                trg_tensor = torch.tensor([seq[-1]]).unsqueeze(1).to(device)
                logits, next_h_state, _ = model.decoder(trg_tensor, h_state, encoder_outputs, src_mask)
                log_probs = F.log_softmax(logits.squeeze(1), dim=1)
                
                top_log_probs, top_indices = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    next_token = top_indices[0][i].item()
                    next_log_prob = top_log_probs[0][i].item()
                    
                    new_score = score + next_log_prob
                    new_seq = seq + [next_token]
                    new_beam.append((new_score, new_seq, next_h_state))
            
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:beam_width]
            
            if all(candidate[1][-1] == eos_token_id for candidate in beam):
                break
                
    best_seq = beam[0][1]
    return best_seq[1:]


def evaluate_bleu(model, test_loader, fr_tok, en_tok, device, beam_width=3, max_len=32):
    """Evaluates the model on a test set using Beam Search and calculates Corpus BLEU."""
    model.eval()
    
    sources = []    
    references = []
    hypotheses = []
    
    print(f"Starting BLEU Evaluation (Beam Width: {beam_width})...")
    
    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(test_loader, desc="Translating Test Set"):
            
            for i in range(src_batch.shape[0]):
                # 1. Prepare Source
                src_ids = src_batch[i].tolist()
                if fr_tok.pad_token_id in src_ids:
                    src_ids = src_ids[:src_ids.index(fr_tok.pad_token_id)]
                
                source_text = fr_tok.decode(src_ids, skip_special_tokens=True)
                sources.append(source_text)
                

                src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
                src_mask = model.create_src_mask(src_tensor)
                
                # 2. Prepare Ground Truth Reference (Expected Output)
                tgt_ids = tgt_batch[i].tolist()
                if en_tok.pad_token_id in tgt_ids:
                    tgt_ids = tgt_ids[:tgt_ids.index(en_tok.pad_token_id)]
                
                clean_tgt_ids = [idx for idx in tgt_ids if idx not in [en_tok.bos_token_id, en_tok.eos_token_id]]
                reference_text = en_tok.decode(clean_tgt_ids, skip_special_tokens=True)
                reference_text = reference_text.replace(" ", "").replace("▁", " ").replace("_", " ").strip()
                
                # 3. Generate Hypothesis via Beam Search (Model Output)
                pred_ids = beam_search_decode(
                    model=model,
                    src_tensor=src_tensor,
                    src_mask=src_mask,
                    bos_token_id=en_tok.bos_token_id,
                    eos_token_id=en_tok.eos_token_id,
                    max_len=max_len,
                    beam_width=beam_width,
                    device=device
                )
                
                hypothesis_text = en_tok.decode(pred_ids, skip_special_tokens=True)
                hypothesis_text = hypothesis_text.replace(" ", "").replace("▁", " ").replace("_", " ").strip()
                
                references.append([reference_text.split()])
                hypotheses.append(hypothesis_text.split())

    bleu_score = corpus_bleu(references, hypotheses) * 100
    
    return bleu_score, sources, references, hypotheses