import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class BaseJointRE(nn.Module):
    """
    Base class for preprocessing and dataloader
    """

    def __init__(self, classes_to_id, rel_to_id, max_width):
        super().__init__()

        # tags: BIO
        # classes: label name [eg. PERS, ORG, ...]
        self.classes_to_id = classes_to_id
        self.id_to_classes = {k: v for v, k in self.classes_to_id.items()}
        self.rel_to_id = rel_to_id
        self.id_to_rel = {k: v for v, k in self.rel_to_id.items()}
        self.max_width = max_width

    def preprocess(self, tokens, rel_seq=None):
        N = len(tokens)

        spans_idx = []
        for i in range(N):
            spans_idx.extend([(i, i + j) for j in range(self.max_width)])

        # 0 for null labels
        span_label = torch.LongTensor([0 for _ in spans_idx])
        spans_idx = torch.LongTensor(spans_idx)
        # mask for valid spans
        valid_span_mask = spans_idx[:, 1] > N - 1
        # mask invalid positions
        span_label = span_label.masked_fill(valid_span_mask, -1)

        return {
            'tokens': tokens,
            'span_idx': spans_idx,
            'span_label': span_label,
            'seq_length': N,
            'graph': rel_seq,
        }

    def collate_fn(self, batch_list):
        batch = [self.preprocess(tokens, seq)
                 for (tokens, ner, rel, seq) in batch_list]

        span_idx = pad_sequence(
            [b['span_idx'] for b in batch], batch_first=True, padding_value=0)

        span_label = pad_sequence(
            [el['span_label'] for el in batch], batch_first=True, padding_value=-1)

        span_mask = span_label != -1

        graph = [el['graph'] for el in batch]

        return {
            'seq_length': torch.LongTensor([el['seq_length'] for el in batch]),
            'span_idx': span_idx,
            'tokens': [el['tokens'] for el in batch],
            'span_mask': span_mask,
            'graph': graph,
        }

    def create_dataloader(self, data, **kwargs):
        return DataLoader(data, collate_fn=self.collate_fn, **kwargs)
