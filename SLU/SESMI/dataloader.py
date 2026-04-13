import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
from processSpan import get_entities 

class JSONLSample:
    def __init__(self, file_path, max_length):
        self.file_path = file_path
        self.max_length = max_length

    def read_jsonl(self):
        samples = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            file_path = self.file_path.lower()
            for line in f:
                line = line.strip()
                if not line: continue
                data = json.loads(line)
                
                utterance = data["utterance"]
                if isinstance(utterance, str):
                    utterance = utterance.split()
                
                slots = data["slots"]
                intents = data["intent"]
                explanation = data.get("explanation", "")
                # explanation = ""

                if len(utterance) > self.max_length - 2:
                    utterance = utterance[:self.max_length - 2]
                    slots = slots[:self.max_length - 2]

                samples.append({
                    "utterance": utterance,
                    "raw_slots": slots,
                    "intent_label": intents,
                    "explanation": explanation
                })
        return samples

class MyDataSet(Dataset):
    def __init__(self, args, file_path, intent_label_set, slot_label_set, tokenizer):
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length 
        
        self.intent_label_id = {w: i for i, w in enumerate(intent_label_set)}
        
        self.slot_label_id = {"O": 0} 
        current_idx = 1
        for label in slot_label_set:
            if label not in ["O", "PAD"]:
                self.slot_label_id[label] = current_idx
                current_idx += 1
        self.num_slot_labels = len(self.slot_label_id)

    def process_one_sample(self, sample):
        sentence = sample["utterance"]     
        explanation = sample["explanation"] 
        raw_slots = sample["raw_slots"]     
        intent_label = sample["intent_label"]
        tokens = []
        words_lengths = [] 

        tokens.append(self.tokenizer.cls_token)
        words_lengths.append(1) 

        for word in sentence:
            sub_tokens = self.tokenizer.tokenize(word)
            if not sub_tokens: sub_tokens = [self.tokenizer.unk_token]
            
            if len(tokens) + len(sub_tokens) >= self.max_seq_length - 1:
                break
            
            tokens.extend(sub_tokens)
            words_lengths.append(len(sub_tokens))

        tokens.append(self.tokenizer.sep_token)
        words_lengths.append(1)
        
        word_attention_mask = [1] * len(tokens) 
        
        # Explanation
        if explanation:
            exp_tokens = self.tokenizer.tokenize(explanation)
            remaining = self.max_seq_length - len(tokens) - 1
            if remaining > 0:
                exp_tokens = exp_tokens[:remaining]
                tokens.extend(exp_tokens)
                # word_attention_mask.extend([0] * len(exp_tokens))
                tokens.append(self.tokenizer.sep_token)
                # word_attention_mask.append(0)
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        valid_word_count = len(words_lengths) 
        current_slots = ["O"] + raw_slots[:valid_word_count] + ["O"]
        
        start_ids = [0] * len(current_slots)
        end_ids = [0] * len(current_slots)
        
        spans = get_entities(current_slots)
        
        for type_, start, end in spans:
            if type_ in self.slot_label_id:
                label_idx = self.slot_label_id[type_]
                if start < len(start_ids) and end < len(end_ids):
                    start_ids[start] = label_idx
                    end_ids[end] = label_idx

        intent_tags = [0] * len(self.intent_label_id)  
        for intent in intent_label:
            if intent in self.intent_label_id:
                intent_tags[self.intent_label_id[intent]] = 1
            elif 'default_intent' in self.intent_label_id:
                 intent_tags[self.intent_label_id['default_intent']] = 1

        count_labels = len(intent_label) - 1  

        return (
            torch.tensor(input_ids, dtype=torch.long), 
            torch.tensor(attention_mask, dtype=torch.long), 
            torch.tensor(words_lengths, dtype=torch.long), 
            torch.tensor(word_attention_mask, dtype=torch.long),
            torch.tensor(intent_tags, dtype=torch.float),
            torch.tensor(start_ids, dtype=torch.long),
            torch.tensor(end_ids, dtype=torch.long),
            torch.tensor(count_labels, dtype=torch.long),
            sentence 
        )

    def __getitem__(self, idx):
        return self.processed_data[idx]
        
    def __len__(self):
        return len(self.processed_data)


def collate_fn(batch, pad_id, max_seq_len):
    input_ids, attention_mask, words_lengths, word_attention_mask, intent_tags, start_ids, end_ids, count_labels, sentences = zip(*batch)

    def pad_tensor(tensor, max_len=None, pad_value=0):
        length = tensor.size(0)
        if max_len is None:
            return tensor
        if length < max_len:
            return F.pad(tensor, (0, max_len - length), value=pad_value)
        else:
            return tensor[:max_len]
    
    input_ids = torch.stack([pad_tensor(t, max_len=max_seq_len, pad_value=pad_id) for t in input_ids])
    attention_mask = torch.stack([pad_tensor(t, max_len=max_seq_len) for t in attention_mask])
    word_attention_mask = torch.stack([pad_tensor(t, max_len=max_seq_len) for t in word_attention_mask])
    
    max_word_len = max(w.size(0) for w in words_lengths)
    
    words_lengths_padded = []
    for w in words_lengths:
        if w.size(0) < max_word_len:
            w = F.pad(w, (0, max_word_len - w.size(0)), value=0)
        words_lengths_padded.append(w)
    words_lengths = torch.stack(words_lengths_padded)
    
    start_ids = torch.stack([pad_tensor(t, max_len=max_word_len, pad_value=0) for t in start_ids])
    end_ids = torch.stack([pad_tensor(t, max_len=max_word_len, pad_value=0) for t in end_ids])
    
    intent_tags = torch.stack(intent_tags)
    count_labels = torch.stack(count_labels)

    return (
        input_ids, attention_mask, words_lengths, word_attention_mask,
        intent_tags, start_ids, end_ids, count_labels, sentences
    )
