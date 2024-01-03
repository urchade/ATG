import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


def create_position_code_sep(seq_graph, sep_token="stop_entity"):
    # seq_graph: list of tokens
    # return: list of codes
    # codes: 0: entity, 1: head entity, 2: tail entity, 3: relation
    generating_entities = True
    relation_step = 0
    codes = []
    for node in seq_graph:
        if generating_entities:
            codes.append(4)
        else:
            if relation_step % 3 == 0:
                codes.append(1)
            elif relation_step % 3 == 1:
                codes.append(2)
            else:
                codes.append(3)
            relation_step += 1
        if node == sep_token:
            generating_entities = False
    # add start code
    codes = [0] + codes
    return codes


class PosEmbCodeSep(nn.Module):
    def __init__(self, emb_dim, max_len=1000, add_absolute_emb=True):
        super(PosEmbCodeSep, self).__init__()
        self.emb_dim = emb_dim
        self.add_absolute_emb = add_absolute_emb
        self.structural_embedding = nn.Embedding(5, emb_dim, padding_idx=0)

        if self.add_absolute_emb:
            self.absolute_embedding = nn.Parameter(torch.zeros(max_len, emb_dim))
            torch.nn.init.trunc_normal_(self.absolute_embedding, std=0.02)

        torch.nn.init.trunc_normal_(self.structural_embedding.weight, std=0.02)

    def forward(self, x, seq_graph):
        # x: (batch_size, seq_len, emb_dim)
        # seq_graph: (batch_size,)
        batch_size, seq_len, emb_dim = x.size()
        assert emb_dim == self.emb_dim

        # pos_codes: (batch_size, seq_len)
        pos_codes = self.get_codes(seq_graph).to(x.device)

        # code_emb: (batch_size, seq_len, emb_dim)
        code_emb = self.structural_embedding(pos_codes)

        # out: (batch_size, seq_len, emb_dim)
        x = x + code_emb[:, :seq_len, :]
        if self.add_absolute_emb:
            x = x + self.absolute_embedding[:seq_len, :]

        return x

    def get_codes(self, graph):
        pos_codes = []
        for i in range(len(graph)):
            codes = create_position_code_sep(graph[i])
            pos_codes.append(torch.LongTensor(codes))
        pos_codes = pad_sequence(
            pos_codes,
            batch_first=True,
            padding_value=0)
        return pos_codes

