import torch
import torch.nn as nn

from layers.structure import PosEmbCodeSep


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class TransDec(nn.Module):
    def __init__(self, d_model, num_heads, num_layers=4, cross_attn=False, dropout=0.1, add_pos_emb=True):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cross_attn = cross_attn
        self.dropout = dropout
        self.add_pos_emb = add_pos_emb

        if self.add_pos_emb:
            self.pos_emb = PosEmbCodeSep(d_model, add_absolute_emb=False)

        if cross_attn is False:
            encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model * 4,
                                                        dropout=dropout, batch_first=True)
            self.trans_layer = nn.TransformerEncoder(encoder_layers, num_layers)
        else:
            encoder_layers = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward=d_model * 4,
                                                        dropout=dropout, batch_first=True)
            self.trans_layer = nn.TransformerDecoder(encoder_layers, num_layers)

        self.register_buffer('causal_mask', generate_square_subsequent_mask(1000).to(torch.bool))

    def forward(self, x, seq_graph=None, memory=None, memory_pad_mask=None):
        # x: (batch_size, seq_len + num_q, emb_dim)
        # seq_graph: list of size (batch_size, seq_len)
        # memory: (batch_size, seq_len, emb_dim)
        # memory_pad_mask: (batch_size, seq_len)

        B, L, D = x.size()

        # out: (batch_size, seq_len, emb_dim)
        if self.add_pos_emb and seq_graph is not None:
            # max length in seq_graph
            max_len = max([len(seq) for seq in seq_graph])
            # num_queries
            num_queries = L - max_len - 1
            # add position embedding only to the nodes not the queries
            x[:, num_queries:, :] = self.pos_emb(x[:, num_queries:, :], seq_graph)

        # mask: (seq_len, seq_len)
        causal_mask = self.causal_mask[:x.size(1), :x.size(1)].to(x.device)

        # out: (batch_size, seq_len, emb_dim)
        if self.cross_attn:
            trans_output = self.trans_layer(x, memory=memory, tgt_mask=causal_mask,
                                            memory_key_padding_mask=memory_pad_mask)
        else:
            trans_output = self.trans_layer(x, causal_mask)

        return trans_output
