import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
from tqdm import tqdm  # 引入 tqdm 显示预处理进度
from processSpan import get_entities  # 确保 processSpan.py 在同一目录下

class JSONLSample:
    """读取 JSONL 格式数据"""
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

                # 1. Word 级别的截断 (保留 CLS/SEP 空间)
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
        
        # 1. 设置标签映射
        self.intent_label_id = {w: i for i, w in enumerate(intent_label_set)}
        
        self.slot_label_id = {"O": 0} 
        current_idx = 1
        for label in slot_label_set:
            if label not in ["O", "PAD"]:
                self.slot_label_id[label] = current_idx
                current_idx += 1
        self.num_slot_labels = len(self.slot_label_id)

        # 2. 读取原始数据
        print(f"📖 正在读取数据文件: {file_path}")
        raw_samples = JSONLSample(file_path, args.max_seq_length).read_jsonl()

        # 3. 🚀 核心修改：在初始化时预处理所有数据到内存中
        # 这样训练时 __getitem__ 就不需要再做分词了
        self.processed_data = []
        print("⚙️ 正在进行预处理 (分词与对齐)...")
        for sample in tqdm(raw_samples, desc="Processing"):
            processed_item = self.process_one_sample(sample)
            self.processed_data.append(processed_item)
        print(f"✅ 预处理完成，共 {len(self.processed_data)} 条数据。")

    def process_one_sample(self, sample):
        """
        这里包含了原本 __getitem__ 中的所有繁重逻辑
        """
        sentence = sample["utterance"]      # List[str] words
        explanation = sample["explanation"] 
        raw_slots = sample["raw_slots"]     # List[str] word-level labels
        intent_label = sample["intent_label"]

        # ===================== 1. Input IDs (Token Level) =====================
        tokens = []
        words_lengths = [] # 记录每个 Word 对应的 Token 数

        # [CLS]
        tokens.append(self.tokenizer.cls_token)
        words_lengths.append(1) 

        for word in sentence:
            sub_tokens = self.tokenizer.tokenize(word)
            if not sub_tokens: sub_tokens = [self.tokenizer.unk_token]
            
            # 截断保护 (Token 级)
            if len(tokens) + len(sub_tokens) >= self.max_seq_length - 1:
                break
            
            tokens.extend(sub_tokens)
            words_lengths.append(len(sub_tokens))

        # [SEP] (Utterance End)
        tokens.append(self.tokenizer.sep_token)
        words_lengths.append(1)
        
        word_attention_mask = [1] * len(tokens) # BERT 可见的 Token 长度
        
        # Explanation
        if explanation:
            exp_tokens = self.tokenizer.tokenize(explanation)
            remaining = self.max_seq_length - len(tokens) - 1
            if remaining > 0:
                exp_tokens = exp_tokens[:remaining]
                tokens.extend(exp_tokens)
                # word_attention_mask.extend([0] * len(exp_tokens)) # Explanation 部分 mask 设为 0
                tokens.append(self.tokenizer.sep_token)
                # word_attention_mask.append(0)
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        # ===================== 2. Start/End Labels (Word Level) =====================
        # 构造 Word 级别的标签序列
        valid_word_count = len(words_lengths) - 2 # 减去 CLS 和 SEP
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

        # ===================== 3. Return Tensors =====================
        intent_tags = [0] * len(self.intent_label_id)  
        for intent in intent_label:
            if intent in self.intent_label_id:
                intent_tags[self.intent_label_id[intent]] = 1
            elif 'default_intent' in self.intent_label_id:
                 intent_tags[self.intent_label_id['default_intent']] = 1

        count_labels = len(intent_label) - 1  

        # 返回 Tensor，存在内存里
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
        # 🚀 极速取数：直接从内存列表拿，没有任何计算
        return self.processed_data[idx]
        
    def __len__(self):
        return len(self.processed_data)


def collate_fn(batch, pad_id, max_seq_len):
    # 解包已经变成 Tensor 的数据
    input_ids, attention_mask, words_lengths, word_attention_mask, intent_tags, start_ids, end_ids, count_labels, sentences = zip(*batch)

    def pad_tensor(tensor, max_len=None, pad_value=0):
        length = tensor.size(0)
        if max_len is None:
            return tensor
        if length < max_len:
            return F.pad(tensor, (0, max_len - length), value=pad_value)
        else:
            return tensor[:max_len]
    
    # Token 级别 Padding (定长)
    input_ids = torch.stack([pad_tensor(t, max_len=max_seq_len, pad_value=pad_id) for t in input_ids])
    attention_mask = torch.stack([pad_tensor(t, max_len=max_seq_len) for t in attention_mask])
    word_attention_mask = torch.stack([pad_tensor(t, max_len=max_seq_len) for t in word_attention_mask])
    
    # Word 级别 Padding (动态长度)
    max_word_len = max(w.size(0) for w in words_lengths)
    
    # words_lengths 列表处理
    words_lengths_padded = []
    for w in words_lengths:
        if w.size(0) < max_word_len:
            w = F.pad(w, (0, max_word_len - w.size(0)), value=0)
        words_lengths_padded.append(w)
    words_lengths = torch.stack(words_lengths_padded)
    
    # Start/End IDs Padding
    start_ids = torch.stack([pad_tensor(t, max_len=max_word_len, pad_value=0) for t in start_ids])
    end_ids = torch.stack([pad_tensor(t, max_len=max_word_len, pad_value=0) for t in end_ids])
    
    intent_tags = torch.stack(intent_tags)
    count_labels = torch.stack(count_labels)

    return (
        input_ids, attention_mask, words_lengths, word_attention_mask,
        intent_tags, start_ids, end_ids, count_labels, sentences
    )