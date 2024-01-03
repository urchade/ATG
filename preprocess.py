from random import shuffle

import torch
from torch.utils.data import Dataset


class GraphIEData(Dataset):
    def __init__(self, train_data, type="train", max_num_samples=5, mode="noshuffle"):

        self.type = type
        self.train_data = [i.values() for i in train_data]
        self.max_num_samples = max_num_samples
        self.mode = mode

    def __len__(self):
        return len(self.train_data)

    def create_seq(self, idx):
        tokens, ner, rel, _ = self.train_data[idx]
        # template: (ner1, ner2, ner3, ner4, [sep], ner1, ner2, rel12, ner3, ner4, rel34
        dict_rel = {}
        for i, j, k in rel:
            if k in ["CONJUNCTION", "COMPARE"]:
                if torch.rand(1).item() < 0.5:
                    dict_rel[(i, j)] = k
                else:
                    dict_rel[(j, i)] = k
            else:
                dict_rel[(i, j)] = k

        seq = []
        for n in ner:
            seq.append(n)

        seq.append("stop_entity")
        for i, n1 in enumerate(ner):
            for j, n2 in enumerate(ner):
                if i == j:
                    continue
                if (i, j) in dict_rel:
                    seq.append(n1)
                    seq.append(n2)
                    seq.append(dict_rel[(i, j)])

        return tokens, None, None, seq

    def sample_combine(self, current_tokens, current_seq, sampled_idx):

        sampled_tokens, _, _, sampled_seq = self.create_seq(sampled_idx)

        # combine tokens
        new_tokens = current_tokens + sampled_tokens
        N = len(current_tokens)

        # shift sampled by N if element is tuple and keep origina
        sampled_seq = [(i[0] + N, i[1] + N, i[2]) if isinstance(i, tuple) else i for i in sampled_seq]

        # combine seq (make sure to add the offset and separate "stop_entity")
        # before "stop_entity"
        combined_before = current_seq[:current_seq.index("stop_entity")] + sampled_seq[
                                                                           :sampled_seq.index("stop_entity")]
        # after "stop_entity"
        combined_after = current_seq[current_seq.index("stop_entity") + 1:] + sampled_seq[
                                                                              sampled_seq.index("stop_entity") + 1:]

        # new seq
        new_seq = combined_before + ["stop_entity"] + combined_after

        return new_tokens, None, None, new_seq

    def add_samples(self, idx, max_num_samples):

        # randomly generate number of samples with torch

        if torch.rand(1).item() < 1. / max_num_samples:
            num_samples = 1
        else:
            num_samples = torch.randint(1, max_num_samples + 1, (1,)).item()

        tokens, _, _, seq = self.create_seq(idx)

        if num_samples == 1:
            return tokens, None, None, seq

        # generate unique random samples with torch
        sampled_idx = torch.randint(0, len(self.train_data), (num_samples,)).unique().tolist()

        for i in sampled_idx:
            tokens, _, _, seq = self.sample_combine(tokens, seq, i)
            if len(tokens) > 285:
                break
        return tokens, None, None, seq

    def shuffle_entities_relations(self, seq):
        # before "stop_entity"
        before = seq[:seq.index("stop_entity")]
        # after "stop_entity"
        after = seq[seq.index("stop_entity") + 1:]

        # shuffle before (entities)
        shuffle(before)

        # shuffle after (organize by triples)
        after_triples = []
        for i in range(0, len(after), 3):
            # prevent out of range
            if i + 2 >= len(after):
                break
            after_triples.append([after[i], after[i + 1], after[i + 2]])

        shuffle(after_triples)

        after = [i for j in after_triples for i in j]

        # new seq
        new_seq = before + ["stop_entity"] + after

        return new_seq

    def add_shuffle(self, seq):
        # before "stop_entity"
        before = seq[:seq.index("stop_entity")]
        # after "stop_entity"
        after = seq[seq.index("stop_entity") + 1:]

        # shuffle before (entities)
        # shuffle(before)

        # create a copy of before and shuffle it
        before_copy = before.copy()
        shuffle(before_copy)

        # shuffle after (organize by triples)
        after_triples = []
        for i in range(0, len(after), 3):
            # prevent out of range
            if i + 2 >= len(after):
                break
            after_triples.append([after[i], after[i + 1], after[i + 2]])

        shuffle(after_triples)

        after_triples = [i for j in after_triples for i in j]

        if torch.rand(1).item() < 0.5:
            before = before + before_copy

        if torch.rand(1).item() < 0.5:
            after = after + after_triples

        # new seq
        new_seq = before + ["stop_entity"] + after

        return new_seq

    def __getitem__(self, idx):
        if self.type == "train":
            output = self.add_samples(idx, self.max_num_samples)
        else:
            output = self.create_seq(idx)

        tokens, _, _, seq = output

        if self.mode == "shuffle":
            seq = self.shuffle_entities_relations(seq)

        return tokens, None, None, seq
