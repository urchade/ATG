import torch
import torch.nn.functional as F
from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor, \
    BidirectionalEndpointSpanExtractor
from torch import nn


class SpanQuery(nn.Module):

    def __init__(self, hidden_size, max_width, trainable=True):
        super().__init__()

        self.query_seg = nn.Parameter(torch.randn(hidden_size, max_width))

        nn.init.uniform_(self.query_seg, a=-1, b=1)

        if not trainable:
            self.query_seg.requires_grad = False

        self.project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, h, *args):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        span_rep = torch.einsum('bld, ds->blsd', h, self.query_seg)

        return self.project(span_rep)


class SpanMLP(nn.Module):

    def __init__(self, hidden_size, max_width):
        super().__init__()

        self.mlp = nn.Linear(hidden_size, hidden_size * max_width)

    def forward(self, h, *args):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        span_rep = self.mlp(h)

        span_rep = span_rep.view(B, L, -1, D)

        return span_rep.relu()


class SpanCAT(nn.Module):

    def __init__(self, hidden_size, max_width):
        super().__init__()

        self.max_width = max_width

        self.query_seg = nn.Parameter(torch.randn(128, max_width))

        self.project = nn.Sequential(
            nn.Linear(hidden_size + 128, hidden_size),
            nn.ReLU()
        )

    def forward(self, h, *args):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        h = h.view(B, L, 1, D).repeat(1, 1, self.max_width, 1)

        q = self.query_seg.view(1, 1, self.max_width, -1).repeat(B, L, 1, 1)

        span_rep = torch.cat([h, q], dim=-1)

        span_rep = self.project(span_rep)

        return span_rep


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class SpanEndpoints(nn.Module):

    def __init__(self, hidden_size, max_width, width_embedding=128):
        super().__init__()

        self.span_extractor = EndpointSpanExtractor(hidden_size,
                                                    combination='x,y')

        self.downproject = nn.Sequential(
            Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )

    def forward(self, h, span_idx):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        span_rep = self.span_extractor(h, span_idx)

        return self.downproject(span_rep).view(B, L, -1, D)


class SpanAttention(nn.Module):

    def __init__(self, hidden_size, max_width, width_embedding=128):
        super().__init__()

        self.span_extractor = SelfAttentiveSpanExtractor(hidden_size,
                                                         num_width_embeddings=max_width,
                                                         span_width_embedding_dim=width_embedding,
                                                         )
        self.downproject = nn.Sequential(
            nn.Linear(hidden_size + width_embedding, hidden_size),
            nn.ReLU()
        )

    def forward(self, h, span_idx):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        span_rep = self.span_extractor(h, span_idx)

        return self.downproject(span_rep).view(B, L, -1, D)


class Bidir(nn.Module):

    def __init__(self, hidden_size, max_width):
        super().__init__()

        self.span_extractor = BidirectionalEndpointSpanExtractor(hidden_size)

        self.downproject = nn.Sequential(
            nn.Linear(self.span_extractor.get_output_dim(), hidden_size),
            nn.ReLU()
        )

    def forward(self, h, span_idx):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        span_rep = self.span_extractor(h, span_idx)

        return self.downproject(span_rep).view(B, L, -1, D)


class SpanConvBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, span_mode='conv_normal'):
        super().__init__()

        if span_mode == 'conv_conv':
            self.conv = nn.Conv1d(hidden_size, hidden_size,
                                  kernel_size=kernel_size)
        elif span_mode == 'conv_max':
            self.conv = nn.MaxPool1d(kernel_size=kernel_size, stride=1)
        elif span_mode == 'conv_mean' or span_mode == 'conv_sum':
            self.conv = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

        self.span_mode = span_mode

        self.pad = kernel_size - 1

    def forward(self, x):

        x = torch.einsum('bld->bdl', x)

        if self.pad > 0:
            x = F.pad(x, (0, self.pad), "constant", 0)

        x = self.conv(x)

        if self.span_mode == "conv_sum":
            x = x * (self.pad + 1)

        return torch.einsum('bdl->bld', x)


class SpanConv(nn.Module):
    def __init__(self, hidden_size, max_width, span_mode):
        super().__init__()

        kernels = [i + 2 for i in range(max_width - 1)]

        self.convs = nn.ModuleList()

        for kernel in kernels:
            self.convs.append(SpanConvBlock(hidden_size, kernel, span_mode))

        self.project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, x, *args):

        span_reps = [x]

        for conv in self.convs:
            h = conv(x)
            span_reps.append(h)

        span_reps = torch.stack(span_reps, dim=-2)

        return self.project(span_reps)


class SpanEndpointsBlock(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()

        self.kernel_size = kernel_size

    def forward(self, x):
        B, L, D = x.size()

        span_idx = torch.LongTensor(
            [[i, i + self.kernel_size - 1] for i in range(L)]).to(x.device)

        x = F.pad(x, (0, 0, 0, self.kernel_size - 1), "constant", 0)

        # endrep
        start_end_rep = torch.index_select(x, dim=1, index=span_idx.view(-1))

        start_end_rep = start_end_rep.view(B, L, 2, D)

        return start_end_rep


class SpanEndpointsV2(nn.Module):
    def __init__(self, hidden_size, max_width, span_mode='endpoints_mean'):
        super().__init__()

        assert span_mode in ['endpoints_mean',
                             'endpoints_max', 'endpoints_cat']

        self.K = max_width

        kernels = [i + 1 for i in range(max_width)]

        self.convs = nn.ModuleList()

        for kernel in kernels:
            self.convs.append(SpanEndpointsBlock(kernel))

        self.span_mode = span_mode

    def forward(self, x, *args):

        span_reps = []

        for conv in self.convs:
            h = conv(x)

            span_reps.append(h)

        span_reps = torch.stack(span_reps, dim=-3)

        if self.span_mode == 'endpoints_mean':
            span_reps = torch.mean(span_reps, dim=-2)
        elif self.span_mode == 'endpoints_max':
            span_reps = torch.max(span_reps, dim=-2).values
        elif self.span_mode == 'endpoints_cat':
            span_reps = span_reps.view(B, L, self.K, -1)

        return span_reps


class ConvShare(nn.Module):
    def __init__(self, hidden_size, max_width):
        super().__init__()

        self.max_width = max_width

        self.conv_weigth = nn.Parameter(
            torch.randn(hidden_size, hidden_size, max_width))

        nn.init.xavier_normal_(self.conv_weigth)

    def forward(self, x, *args):
        span_reps = []

        x = torch.einsum('bld->bdl', x)

        for i in range(self.max_width):
            pad = i
            x_i = F.pad(x, (0, pad), "constant", 0)
            conv_w = self.conv_weigth[:, :, :i + 1]
            out_i = F.conv1d(x_i, conv_w)
            span_reps.append(out_i.transpose(-1, -2))

        return torch.stack(span_reps, dim=-2)


class ConvShareEndpoints(nn.Module):
    def __init__(self, hidden_size, max_width):
        super().__init__()

        self.max_width = max_width
        self.out_size = hidden_size * 3
        self.span_extractor = EndpointSpanExtractor(hidden_size,
                                                    combination='x,y')
        self.conv_share = ConvShare(hidden_size, max_width)

        self.out_project_end = nn.Linear(hidden_size * 2, hidden_size // 2)

        self.out_project_conv = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, h, span_idx):
        B, L, D = h.size()
        span_rep_end = self.span_extractor(h, span_idx)
        span_rep_end = self.out_project_end(span_rep_end).view(B, L, self.max_width, -1)

        span_rep_conv = self.conv_share(h)
        span_rep_conv = self.out_project_conv(span_rep_conv).view(B, L, self.max_width, -1)

        return torch.cat([span_rep_end, span_rep_conv], dim=-1)


class SpanMarker(nn.Module):

    def __init__(self, hidden_size, max_width, dropout=0.4):
        super().__init__()

        self.max_width = max_width

        self.project_start = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
        )

        self.project_end = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
        )

        self.span_extractor_start = EndpointSpanExtractor(hidden_size,
                                                          combination='x')

        self.span_extractor_end = EndpointSpanExtractor(hidden_size,
                                                        combination='y')

        self.out_project = nn.Linear(hidden_size * 2, hidden_size, bias=True)

    def forward(self, h, span_idx):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        # project start and end
        start_rep = self.project_start(h)
        end_rep = self.project_end(h)

        # extract span
        start_span_rep = self.span_extractor_start(start_rep, span_idx)
        end_span_rep = self.span_extractor_end(end_rep, span_idx)

        # concat start and end
        cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()

        # project
        cat = self.out_project(cat)

        # reshape
        return cat.view(B, L, self.max_width, D)


class SpanMarkConv(nn.Module):
    def __init__(self, hidden_size, max_width, dropout=0.4):
        super().__init__()

        self.max_width = max_width

        self.project = nn.Linear(hidden_size, hidden_size * 2)

        self.span_extractor_start = EndpointSpanExtractor(hidden_size,
                                                          combination='x')

        self.span_extractor_end = EndpointSpanExtractor(hidden_size,
                                                        combination='y')

        self.conv_share = ConvShare(hidden_size, max_width)

        self.out_project = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU()
        )

    def forward(self, h, span_idx):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        # project start and end
        start_rep, end_rep = self.project(h).chunk(2, dim=-1)

        # extract span
        start_span_rep, end_span_rep = self.span_extractor_start(start_rep, span_idx), \
            self.span_extractor_end(end_rep, span_idx)

        conv_span_rep = self.conv_share(h).reshape(B, L * self.max_width, D)  # conv feature

        # concat start and end
        cat = torch.cat([start_span_rep, end_span_rep, conv_span_rep], dim=-1)

        # project
        cat = self.out_project(cat)

        return cat.view(B, L, self.max_width, D)


class SpanRepLayer(nn.Module):
    """
    Various span representation approaches
    """

    def __init__(self, hidden_size, max_width, span_mode, p_drop=0.4):
        super().__init__()

        if span_mode == 'endpoints':
            self.span_rep_layer = SpanEndpoints(hidden_size, max_width)
        elif span_mode == 'attentive':
            self.span_rep_layer = SpanAttention(hidden_size, max_width)
        elif span_mode == 'marker':
            self.span_rep_layer = SpanMarker(hidden_size, max_width, p_drop)
        elif span_mode == 'markconv':
            self.span_rep_layer = SpanMarkConv(hidden_size, max_width)
        elif span_mode == 'birectionnal':
            self.span_rep_layer = Bidir(hidden_size, max_width)
        elif span_mode == 'query':
            self.span_rep_layer = SpanQuery(
                hidden_size, max_width, trainable=True)
        elif span_mode == 'mlp':
            self.span_rep_layer = SpanMLP(hidden_size, max_width)
        elif span_mode == 'cat':
            self.span_rep_layer = SpanCAT(hidden_size, max_width)
        elif span_mode == 'conv_conv':
            self.span_rep_layer = SpanConv(
                hidden_size, max_width, span_mode='conv_conv')
        elif span_mode == 'conv_max':
            self.span_rep_layer = SpanConv(
                hidden_size, max_width, span_mode='conv_max')
        elif span_mode == 'conv_mean':
            self.span_rep_layer = SpanConv(
                hidden_size, max_width, span_mode='conv_mean')
        elif span_mode == 'conv_sum':
            self.span_rep_layer = SpanConv(
                hidden_size, max_width, span_mode='conv_sum')
        elif span_mode == 'conv_share':
            self.span_rep_layer = ConvShare(hidden_size, max_width)
        elif span_mode == 'conv_share_endpoints':
            self.span_rep_layer = ConvShareEndpoints(hidden_size, max_width)
        else:
            self.span_rep_layer = SpanEndpointsV2(
                hidden_size, max_width, span_mode=span_mode)

    def forward(self, x, *args):
        return self.span_rep_layer(x, *args)
