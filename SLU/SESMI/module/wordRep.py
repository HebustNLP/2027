import torch
from torch import nn
from transformers import AutoModel


class WordRep(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(args.model_name_or_path)
        self.dropout = nn.Dropout(args.dropout_rate)

    def forward(self, input_ids, attention_mask, word_attention_mask, words_lengths):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )  
       
        sequence_output = self.dropout(outputs.last_hidden_state)  # [B, L, H]
        cls_output = sequence_output[:, 0, :]  # [B, H]

        batch_size, seq_len, hidden_size = sequence_output.size()
        max_word_len = words_lengths.size(1)

        context_embedding = torch.zeros(
            batch_size,
            max_word_len,
            hidden_size,
            device=sequence_output.device
        )

        for b in range(batch_size):
            token_ptr = 1  # skip [CLS]

            for w in range(max_word_len):
                sub_len = words_lengths[b, w].item()

                if sub_len == 0:
                    break

                sub_emb = sequence_output[
                    b,
                    token_ptr: token_ptr + sub_len,
                    :
                ] 

                context_embedding[b, w] = sub_emb.mean(dim=0)

                token_ptr += sub_len

        return cls_output, context_embedding

