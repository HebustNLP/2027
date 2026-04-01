import csv
import json
import torch
from models.transformers import BertTokenizer

class CNerTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=False, **kwargs):
        super().__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case)
        self.vocab_file = str(vocab_file)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        _tokens = []
        for c in text:
            if self.do_lower_case:
                c = c.lower()
            if c in self.vocab:
                _tokens.append(c)
            else:
                _tokens.append('[UNK]')
        return _tokens

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_text(self,input_file):
        lines = []
        with open(input_file,'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words":words,"labels":labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words":words,"labels":labels})
        return lines

    @classmethod
    def _read_json(self,input_file):
        lines = []
        with open(input_file,'r') as f:
            for line in f:
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label',None)
                words = list(text)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for key,value in label_entities.items():
                        for sub_name,sub_index in value.items():
                            for start_index,end_index in sub_index:
                                assert  ''.join(words[start_index:end_index+1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-'+key
                                else:
                                    labels[start_index] = 'B-'+key
                                    labels[start_index+1:end_index+1] = ['I-'+key]*(len(sub_name)-1)
                lines.append({"words": words, "labels": labels, "explanation": line.get("explanation", "")})
        return lines

def get_entity_bios(seq,id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entity_bio(seq,id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entities(seq,id2label,markup='bios'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio','bios']
    if markup =='bio':
        return get_entity_bio(seq,id2label)
    else:
        return get_entity_bios(seq,id2label)

def bert_extract_item(start_logits, end_logits):
    S = []
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                S.append((s_l, i, i + j))
                break
    return S


# import torch
# import torch.nn.functional as F

# def bert_extract_item(
#     start_logits,
#     end_logits,
#     topk_start=5,
#     topk_end=5,
#     max_span_len=30,
#     min_score=None,      # 例如 -8.0（log 概率空间），不设就不过滤
#     margin_over_O=0.0,   # 要求 label 的 logP 比 O 至少大多少，防止瞎报
#     allow_overlap=False
# ):
#     """
#     start_logits/end_logits: [B, L, C]，evaluate 里通常 B=1
#     return: list[(label_id, start_idx, end_idx)]，idx 是去掉 CLS/SEP 后的 0-based
#     """
#     # [L-2, C]
#     s = start_logits[0, 1:-1]
#     e = end_logits[0, 1:-1]
#     L, C = s.shape

#     s_lp = F.log_softmax(s, dim=-1)  # logP
#     e_lp = F.log_softmax(e, dim=-1)

#     candidates = []
#     O = 0

#     for lab in range(1, C):
#         # topK starts by this label
#         s_scores = s_lp[:, lab]
#         s_topv, s_topi = torch.topk(s_scores, k=min(topk_start, L))

#         for sv, si in zip(s_topv.tolist(), s_topi.tolist()):
#             # margin filter vs O
#             if sv - s_lp[si, O].item() < margin_over_O:
#                 continue

#             # end search window
#             j0 = si
#             j1 = min(L - 1, si + max_span_len - 1)
#             e_scores = e_lp[j0:j1 + 1, lab]
#             e_topv, e_topi = torch.topk(e_scores, k=min(topk_end, e_scores.numel()))

#             for ev, ej_rel in zip(e_topv.tolist(), e_topi.tolist()):
#                 ej = j0 + ej_rel
#                 if ev - e_lp[ej, O].item() < margin_over_O:
#                     continue
#                 score = sv + ev  # joint score in log space
#                 if (min_score is None) or (score >= min_score):
#                     candidates.append((score, lab, si, ej))

#     # sort by score desc
#     candidates.sort(key=lambda x: x[0], reverse=True)

#     # greedy select (optionally non-overlap)
#     selected = []
#     occupied = [False] * L

#     for score, lab, si, ej in candidates:
#         if not allow_overlap:
#             if any(occupied[k] for k in range(si, ej + 1)):
#                 continue
#         selected.append((lab, si, ej))
#         if not allow_overlap:
#             for k in range(si, ej + 1):
#                 occupied[k] = True

#     return selected


