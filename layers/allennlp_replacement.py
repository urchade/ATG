import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LstmSeq2SeqEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x, mask):
        # x: (batch_size, seq_len, input_size)
        # mask: (batch_size, seq_len)
        lengths = mask.sum(dim=1).cpu()
        packed_input = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output


class EndpointSpanExtractor(nn.Module):
    def __init__(self, input_dim, combination="x,y"):
        super().__init__()
        self.input_dim = input_dim
        self.combination = combination

    def get_output_dim(self):
        if self.combination == "x,y":
            return self.input_dim * 2
        return self.input_dim

    def forward(self, sequence_tensor, span_indices):
        batch_size, sequence_length, embedding_dim = sequence_tensor.size()

        original_shape = span_indices.shape
        if len(original_shape) > 3:
            span_indices = span_indices.contiguous().view(batch_size, -1, 2)

        span_indices = span_indices.clamp(0, sequence_length - 1)

        start_indices = span_indices[..., 0]
        end_indices = span_indices[..., 1]

        start_indices_expanded = start_indices.unsqueeze(-1).expand(-1, -1, embedding_dim)
        end_indices_expanded = end_indices.unsqueeze(-1).expand(-1, -1, embedding_dim)

        start_embeddings = torch.gather(sequence_tensor, 1, start_indices_expanded)
        end_embeddings = torch.gather(sequence_tensor, 1, end_indices_expanded)

        if self.combination == "x,y":
            result = torch.cat([start_embeddings, end_embeddings], dim=2)
        elif self.combination == "x":
            result = start_embeddings
        elif self.combination == "y":
            result = end_embeddings
        else:
            raise ValueError(f"Combination not supported: {self.combination}")

        if len(original_shape) > 3:
            result_shape = original_shape[:-1] + (self.get_output_dim(),)
            return result.view(result_shape)

        return result


class SelfAttentiveSpanExtractor(nn.Module):
    def __init__(self, input_dim, num_width_embeddings=None, span_width_embedding_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.span_width_embedding_dim = span_width_embedding_dim or 0
        if num_width_embeddings and span_width_embedding_dim:
            self.width_embedding = nn.Embedding(num_width_embeddings, self.span_width_embedding_dim)
        else:
            self.width_embedding = None
        self.attention_scorer = nn.Linear(input_dim, 1)

    def get_output_dim(self):
        return self.input_dim + self.span_width_embedding_dim

    def forward(self, sequence_tensor, span_indices):
        batch_size, sequence_length, _ = sequence_tensor.size()

        original_shape = span_indices.shape
        if len(original_shape) > 3:
            span_indices = span_indices.contiguous().view(batch_size, -1, 2)

        num_spans = span_indices.size(1)

        span_widths = span_indices[:, :, 1] - span_indices[:, :, 0] + 1

        token_indices = torch.arange(sequence_length, device=sequence_tensor.device).expand(batch_size, num_spans, -1)
        span_mask = (token_indices >= span_indices[:, :, 0].unsqueeze(-1)) & \
                    (token_indices <= span_indices[:, :, 1].unsqueeze(-1))

        attention_scores = self.attention_scorer(sequence_tensor).squeeze(-1).unsqueeze(1)

        attention_scores = attention_scores.expand(-1, num_spans, -1)
        attention_scores = attention_scores.masked_fill(~span_mask, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_embeddings = torch.bmm(attention_weights, sequence_tensor)

        result = attended_embeddings
        if self.width_embedding:
            clamped_widths = span_widths.clamp(0, self.width_embedding.num_embeddings - 1)
            width_embeddings = self.width_embedding(clamped_widths)
            result = torch.cat([attended_embeddings, width_embeddings], dim=-1)

        if len(original_shape) > 3:
            result_shape = original_shape[:-1] + (self.get_output_dim(),)
            return result.view(result_shape)

        return result


class BidirectionalEndpointSpanExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        if input_dim % 2 != 0:
            raise ValueError("input_dim must be even for BidirectionalEndpointSpanExtractor")
        self.input_dim = input_dim
        self.endpoint_extractor = EndpointSpanExtractor(input_dim // 2, combination='x,y')

    def get_output_dim(self):
        return self.input_dim * 2

    def forward(self, sequence_tensor, span_indices):
        forward_sequence, backward_sequence = sequence_tensor.chunk(2, dim=-1)

        forward_endpoints = self.endpoint_extractor(forward_sequence, span_indices)
        backward_endpoints = self.endpoint_extractor(backward_sequence, span_indices)

        forward_start, forward_end = forward_endpoints.chunk(2, dim=-1)
        backward_start, backward_end = backward_endpoints.chunk(2, dim=-1)

        return torch.cat([forward_start, forward_end, backward_start, backward_end], dim=-1)
