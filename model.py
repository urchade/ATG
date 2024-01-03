import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder
from torch.nn.utils.rnn import pad_sequence

from layers.base import BaseJointRE
from layers.span_embedding import SpanRepLayer
from layers.token_embedding import TokenRep
from trans_enc import TransDec


def MLP(units, dropout, activation=nn.ReLU):
    assert len(units) >= 2
    layers = []
    for i in range(len(units) - 2):
        layers.append(nn.Linear(units[i], units[i + 1]))
        layers.append(activation())
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(units[-2], units[-1]))
    return nn.Sequential(*layers)


class IeGenerator(BaseJointRE):
    def __init__(self, classes_to_id, rel_to_id, max_width, num_prompts=5,
                 model_name="/gpfswork/rech/pds/upa43yu/models/scibert_cased",
                 hidden_transformer=512, attention_heads=8, num_transformer_layers=6,
                 span_mode='conv_share_endpoints', use_pos_code=True, p_drop=0.4, cross_attn=True):
        super().__init__(classes_to_id, rel_to_id, max_width)

        self.args_input_dict = {"classes_to_id": classes_to_id,
                                "rel_to_id": rel_to_id,
                                "max_width": max_width,
                                "num_prompts": num_prompts,
                                "model_name": model_name,
                                "hidden_transformer": hidden_transformer,
                                "num_transformer_layers": num_transformer_layers,
                                "attention_heads": attention_heads,
                                "span_mode": span_mode,
                                "use_pos_code": use_pos_code,
                                "p_drop": p_drop,
                                "cross_attn": cross_attn}

        self.cross_attn = cross_attn

        # start/end/relation embeddings combined
        num_embeddings = len(rel_to_id) + 2 + num_prompts

        # Bert token representation layer
        self.token_rep = TokenRep(num_queries=num_embeddings, model_name=model_name, subtoken_pooling="first")

        # BiLSTM on top of BERT
        self.rnn = LstmSeq2SeqEncoder(
            input_size=self.token_rep.hidden_size,
            hidden_size=self.token_rep.hidden_size // 2,
            num_layers=1,
            bidirectional=True,
        )

        # project embeddings
        self.project_queries = MLP([self.token_rep.hidden_size, hidden_transformer * 4, hidden_transformer], p_drop)
        self.project_tokens = MLP([self.token_rep.hidden_size, hidden_transformer * 4, hidden_transformer], p_drop)

        # span representation layer
        self.span_rep = SpanRepLayer(hidden_transformer, max_width, span_mode=span_mode)

        # class-aware span representation
        self.project_span_class = MLP(
            [hidden_transformer, hidden_transformer * 4, hidden_transformer * len(classes_to_id)],
            p_drop
        )

        # emb projection
        self.embed_proj = MLP(
            [hidden_transformer, hidden_transformer * 4, hidden_transformer],
            p_drop
        )

        # Autoregressive Transformer
        self.decoder = TransDec(d_model=hidden_transformer, num_heads=attention_heads,
                                num_layers=num_transformer_layers, cross_attn=cross_attn, dropout=0.1,
                                add_pos_emb=use_pos_code)

        # project memory
        if self.token_rep.hidden_size != hidden_transformer:
            self.project_memory = MLP([self.token_rep.hidden_size, hidden_transformer], p_drop)
        else:
            self.project_memory = nn.Identity()

    def compute_token_embeddings(self, x):
        # compute contextualized embeddings and queries
        out = self.token_rep(x['tokens'], x['seq_length'])

        # out['queries'] of shape (batch, num_queries, hidden_size)
        # out['embeddings'] of shape (batch, seq_len, hidden_size)
        queries, embed = out['queries'], out['embeddings']

        # lstm
        embed = self.rnn(embed, out['mask'])

        # project the queries and embeddings
        embed, queries = self.project_tokens(embed), self.project_queries(queries)

        return queries, embed, out["cache"]

    def get_splits_queries_out_emb(self, x):
        # compute contextualized embeddings and queries
        queries, embed, cache = self.compute_token_embeddings(x)

        # x['span_idx'] of shape (batch, num_spans, 2)
        span_idx = x['span_idx'] * x['span_mask'].unsqueeze(-1)

        # span_rep of shape (batch, num_spans, hidden_size)
        span_rep = self.span_rep(embed, span_idx).relu()

        # span_class of shape (batch, seq_len, num_spans, hidden_size)
        B, L, K, D = span_rep.size()

        # project the span representation to get the class-aware span representation
        span_class = self.project_span_class(span_rep)
        span_class = span_class.view(B, L, K, -1, D)

        # layer norm for span class and queries
        span_class, queries = self.embed_proj(span_class), self.embed_proj(queries)

        # separate the queries for the relations, start and end (2 queries) and the rest
        rel_queries, start_query, end_query, rest_queries = self.separate_queries(queries)

        # split the span_class tensor into a list of tensors, one for each sequence in the batch
        all_splits = []
        for i, l in enumerate(x['seq_length']):
            all_splits.append(span_class[i, :l])

        # compute the output vocabulary for each input
        all_out_emb = []
        for i, el in enumerate(all_splits):
            all_out_emb.append(self.get_output_embedding(el, rel_queries[i], end_query[i]))

        # return dict with all the splits, queries and output embeddings
        return {'splits': all_splits,
                'queries': [rel_queries, start_query, end_query, rest_queries],
                'out_emb': all_out_emb, 'cache': cache}

    def get_transformer_input(self, representation, true_graph, start_rep, rel_rep):
        inputs = [start_rep.squeeze(0)]
        label_revel = []
        L, K, C, D = representation.shape
        max_rev = np.ravel_multi_index((L - 1, K - 1, C - 1), (L, K, C))

        for index, el in enumerate(true_graph):
            if isinstance(el, tuple):
                s, e, l = el
                pos = (s, e - s, self.classes_to_id[l] - 1)
                rev = np.ravel_multi_index(pos, (L, K, C))
                label_revel.append(rev)
                current_rep = representation[s, e - s, self.classes_to_id[l] - 1]
                inputs.append(current_rep)
            else:
                current_rep = rel_rep[self.rel_to_id[el]]
                inputs.append(current_rep)
                label_revel.append(max_rev + 1 + self.rel_to_id[el])

        label_revel += [max_rev + len(self.rel_to_id) + 1]

        return torch.stack(inputs, dim=0), label_revel

    def get_output_embedding(self, all_reps, rel_rep, end_query):
        # vocabulary consists of all the spans, the relations and the end token
        L, K, C, D = all_reps.shape
        flat_all_reps = all_reps.contiguous().view(-1, D)
        out_emb = torch.cat([flat_all_reps, rel_rep, end_query])

        # mask vocab
        mask = self.get_vocab_mask(L, K, C, device=out_emb.device)

        # mask put to 0 the embeddings for the padding and overflows
        out_emb = out_emb * mask.view(-1, 1).float()
        return out_emb

    def get_vocab_mask(self, L, K, C, device="cpu"):
        """
        Create a mask for the vocabulary, to avoid computing the loss for the padding and overflows
        """
        # mask for size lower than L
        mask = (torch.arange(L).unsqueeze(1) + torch.arange(K)) < L

        # repeat for all classes
        mask = mask.unsqueeze(2).repeat(1, 1, C).view(-1)

        # keep all relations and the end token
        extra_mask = len(self.rel_to_id) + 1

        # original mask + extra mask (all set to True)
        mask_end = torch.ones((mask.size(0) + extra_mask)).bool()

        # set mask to token before the relations and the end token
        mask_end[:-extra_mask] = mask

        # add dimension for batch
        return mask_end.unsqueeze(0).to(device)

    def get_all_trans_input_labels(self, splits, start_query, rel_queries, graphs):
        # compute the transformer input for each input, and the label
        all_labels = []
        all_trans_input = []

        for i, (representation, graph) in enumerate(zip(splits, graphs)):
            transformer_input, label = self.get_transformer_input(
                representation, graph, start_query[i], rel_queries[i]
            )
            all_labels.append(torch.LongTensor(label).to(start_query.device))
            all_trans_input.append(transformer_input)

        # pad the transformer input
        all_trans_input = pad_sequence(all_trans_input, batch_first=True)

        return all_labels, all_trans_input

    def forward(self, x):
        # get splits, queries and output embeddings
        # splits corresponds to the span_class tensor split into a list of tensors, one for each sequence in the batch
        ### each split has shape (l, K, C, dim)
        # queries corresponds to the queries for the relations, start and end (2 queries) and the rest
        ### each query has shape (num_queries, dim)
        # out_emb corresponds to the output vocabulary for each input
        ### each tensor has shape (L*K*C + len(rel_to_id) + 1, dim)
        splits, queries, all_out_emb, cache = self.get_splits_queries_out_emb(x).values()
        rel_queries, start_query, end_query, rest_queries = queries

        # all_labels corresponds to the label for each input
        # all_trans_input corresponds to the transformer input for each input of shape (batch_size, num_ent_rel, dim)
        all_labels, all_trans_input = self.get_all_trans_input_labels(
            splits, start_query, rel_queries,
            x['graph'])

        # append the prompts to the transformer input
        if not self.cross_attn:
            all_trans_input = torch.cat([rest_queries, all_trans_input], dim=1)

        memory = self.project_memory(cache["memory"])
        # compute the transformer output
        all_output_transformer = self.decoder(all_trans_input, x["graph"], memory, cache["memory_pad_mask"])

        # remove the prompts from the transformer output
        if not self.cross_attn:
            all_output_transformer = all_output_transformer[:, rest_queries.size(1):]

        # pad all out_emb
        all_out_emb = pad_sequence(all_out_emb, batch_first=True, padding_value=0.)  # B, all_ent, dim

        # out_emb mask (for 0 padding), keepdims for broadcasting
        out_mask = all_out_emb.sum(dim=-1) != 0  # B, all_ent

        # out_mast to shape B, 1, all_ent
        out_mask = out_mask.unsqueeze(1)

        # pad all labels
        all_labels = pad_sequence(all_labels, batch_first=True, padding_value=-1)  # B, num_ent

        # compute the loss2 (no grad for all_out_emb)
        loss = self.compute_loss(all_out_emb, all_output_transformer, all_labels, out_mask)

        return loss

    def compute_loss(self, all_out_emb, all_output_transformer, all_labels, out_mask):
        # compute the pointing scores
        all_scores = torch.einsum('bad,bld->bla', all_out_emb, all_output_transformer)

        # mask the scores
        all_scores = all_scores.masked_fill(~out_mask, -1e9)

        # compute the loss
        loss = F.cross_entropy(all_scores.view(-1, all_scores.size(-1)), all_labels.view(-1),
                               ignore_index=-1,
                               reduction='mean')

        return loss

    def separate_queries(self, queries):
        # separate the queries into len(rel_to_id) + 2 + num_prompts
        rel_queries = queries[:, :len(self.rel_to_id)]  # (B, len(rel_to_id), dim)
        start_query = queries[:, len(self.rel_to_id): len(self.rel_to_id) + 1]  # (B, 1, dim)
        end_query = queries[:, len(self.rel_to_id) + 1: len(self.rel_to_id) + 2]  # (B, 1, dim)
        others = queries[:, len(self.rel_to_id) + 2:]  # (B, num_prompts, dim)
        return rel_queries, start_query, end_query, others

    @torch.no_grad()
    def decode_batch(self, x):
        # Set model to evaluation mode
        self.eval()

        # Get the device of the first model parameter
        device = next(self.parameters()).device

        # Move input tensors to the device
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(device)

        # Compute span embeddings and separate queries
        splits, queries, all_out_emb, cache = self.get_splits_queries_out_emb(x).values()
        rel_queries, start_query, end_query, prompts = queries

        memory = self.project_memory(cache["memory"])

        # Decode for each split
        all_outs = [
            self.decode_one(splits[i], all_out_emb[i], start_query[i], prompts[i], memory[i])
            for i in range(len(splits))
        ]

        return all_outs

    @torch.no_grad()
    def decode_one(self, splits_i, out_emb_i, start_query_i, rest_queries_i, memory_i):

        # Get the device of the first model parameter
        device = next(self.parameters()).device

        # Prepare input
        inp = start_query_i.view(1, 1, -1)

        memory_i = memory_i.unsqueeze(0)

        rest_queries_i = rest_queries_i.unsqueeze(0)

        # Initialize variables
        L, K, C, Dim = splits_i.shape

        # Get the maximum index in the vocabulary (before the relation part)
        max_rev = np.ravel_multi_index((L - 1, K - 1, C - 1), (L, K, C))
        all_x_dec = []  # decoded sequence

        # True if we are generating entities,
        # False if we are generating relations
        generating_entities = True

        # 0 if we are generating the head of a relation,
        # 1 if we are generating the tail of a relation,
        # 2 if we are generating the relation itself
        relation_step = 0

        generated_entity_ids = []  # List of generated entity ids
        previous = -1  # previously generated idx
        index = 0  # generation step
        while True:  # Decode loop
            index += 1
            if index > 64 and relation_step % 3 == 0:
                break

            inp_mod = torch.clone(inp)

            if not self.cross_attn:
                inp_mod = torch.cat([rest_queries_i, inp_mod], dim=1)

            # compute transformer output
            out_tr = self.decoder.forward(inp_mod, [all_x_dec], memory_i)

            # if not relation
            # Compute scores and sample next input
            flat_scores = torch.einsum('ld, d->l', out_emb_i, out_tr[0, -1])

            # create constraint mask
            constraint_score = self.create_mask_constraint(
                L, K, C, generating_entities, relation_step, generated_entity_ids, previous, device)

            # mask the scores
            flat_scores = flat_scores.masked_fill(constraint_score == False, float("-inf"))

            # sample next input
            next_x = flat_scores.argmax().item()

            # Check if end of sequence is reached
            if next_x == out_emb_i.size(0) - 1:
                break

            if next_x <= max_rev:
                # add the span to the output
                i, k, c = np.unravel_index(next_x, (L, K, C))
                entity_type = self.id_to_classes[c + 1]
                all_x_dec.append((i, i + k, entity_type))

                # if generating entities, add the entity id to the list
                if generating_entities:
                    generated_entity_ids.append(next_x)
            else:
                # add the relation to the output
                rel = next_x - max_rev
                relation_type = self.id_to_rel[rel - 1]
                all_x_dec.append(relation_type)

            # update generating_entities and relation_step
            if not generating_entities:
                relation_step += 1

            if all_x_dec[-1] == 'stop_entity':
                generating_entities = False

            current_embed = out_emb_i[next_x, :].view(1, 1, -1)

            # Update the input
            inp = torch.cat([inp, current_embed], dim=1)

            previous = next_x

        return all_x_dec

    def create_mask_constraint(self, L, K, C, generating_entities, relation_step, generated_entity_ids, previous,
                               device="cpu"):
        # Create a 2D mask of size (L, K) where each cell represents whether the sum of its row and column indices is less than L
        mask = (torch.arange(L).unsqueeze(1) + torch.arange(K)) < L
        # Expand the mask to have an additional dimension for the number of channels
        mask = mask.unsqueeze(2).repeat(1, 1, C).view(-1)
        # Calculate the extra mask size
        extra_mask = len(self.rel_to_id) + 1
        # Create a 1D mask of size (mask.size(0) + extra_mask) with all values set to True
        mask_end = torch.ones((mask.size(0) + extra_mask)).bool()
        # Set the values of the first part of the mask to the values of the original mask
        mask_end[:-extra_mask] = mask

        # mask all relations if we are generating entities exept the end_entity token
        if generating_entities:
            mask_end[-extra_mask:-2] = False
            mask_end[-1] = False
            for i in generated_entity_ids:
                mask_end[i] = False
            # also mask all index before the last generated entity
            # if len(generated_entity_ids) > 0:
            #     mask_end[:generated_entity_ids[-1]] = False
        # do not allow end_entity token
        else:
            # if generating relations (which corresponds to generating two entities and one relation type)
            # so it depends on modularity of the generation step (relation_step)
            if relation_step % 3 == 0:
                # allow entities or end of sequence token
                mask_end[-extra_mask:-1] = False
                mask_end[-1] = True
                mask_end[:-extra_mask] = False
                for i in generated_entity_ids:
                    mask_end[i] = True
            elif relation_step % 3 == 1:
                # allow only entities
                mask_end[:] = False
                for i in generated_entity_ids:
                    mask_end[i] = True
                mask_end[previous] = False
            elif relation_step % 3 == 2:
                # allow only relations
                mask_end[:] = False
                mask_end[-extra_mask:-2] = True

            mask_end[-2] = False

            if len(generated_entity_ids) == 1:
                mask_end[:] = False
                mask_end[-1] = True

        return mask_end.view(-1).to(device)


def nucleus_sampling(flat_scores, p=0.9):
    sorted_flat_scores, sorted_indices = torch.sort(flat_scores, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_flat_scores, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    flat_scores.masked_fill_(indices_to_remove, float("-inf"))

    index = torch.multinomial(F.softmax(flat_scores, dim=-1), num_samples=1)

    return index
