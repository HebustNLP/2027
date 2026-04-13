import torch
from torch import nn
from transformers import AutoModel


class WordRep(nn.Module):
    """
    Word Representation Module
    - cls_output: 全局语义（utterance + explanation）
    - context_embedding: 仅 utterance 部分（token-level，对齐 input_ids）
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(args.model_name_or_path)
        self.dropout = nn.Dropout(args.dropout_rate)

    def forward(self, input_ids, attention_mask, word_attention_mask, words_lengths):
        """
        Args:
            input_ids: [B, L]
            attention_mask: [B, L] → 1 for [CLS], utt, [SEP], exp, [SEP], 0 for PAD
            word_attention_mask: [B, L] → 1 for [CLS], utt, [SEP], 0 for exp & PAD
        Returns:
            cls_output: [B, H] → 全局
            context_embedding: [B, L, H] → 仅 utterance 部分非零
        """
        # 1. BERT 编码（看到 explanation）
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )  
       
        # 2. CLS: 全局语义
        sequence_output = self.dropout(outputs.last_hidden_state)  # [B, L, H]
        cls_output = sequence_output[:, 0, :]  # [B, H]

        # 3. context: 仅 utterance 部分
        batch_size, seq_len, hidden_size = sequence_output.size()
        max_word_len = words_lengths.size(1)

        context_embedding = torch.zeros(
            batch_size,
            max_word_len,
            hidden_size,
            device=sequence_output.device
        )

        # 4. Aggregate subwords -> words
        for b in range(batch_size):
            token_ptr = 1  # skip [CLS]

            for w in range(max_word_len):
                sub_len = words_lengths[b, w].item()

                if sub_len == 0:
                    break

                # collect subword embeddings
                sub_emb = sequence_output[
                    b,
                    token_ptr: token_ptr + sub_len,
                    :
                ]  # [sub_len, H]

                # mean pooling (word representation)
                context_embedding[b, w] = sub_emb.mean(dim=0)

                token_ptr += sub_len

        return cls_output, context_embedding

