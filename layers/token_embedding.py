import torch
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
from torch import nn
from torch.nn.utils.rnn import pad_sequence


#  get_output_dim()

class TokenRep(nn.Module):

    def __init__(self, num_queries=40, model_name="bert-base-cased", fine_tune=True, subtoken_pooling="first"):

        super().__init__()

        self.bert_layer = TransformerWordEmbeddings(model_name, fine_tune=fine_tune, subtoken_pooling=subtoken_pooling)

        self.hidden_size = self.bert_layer.embedding_length

        self.num_queries = num_queries

        # embedding_size
        e_size = self.bert_layer.model.embeddings.word_embeddings.embedding_dim

        if self.num_queries == 0:
            self.query_embedding = None
        else:
            self.query_embedding = nn.Parameter(torch.randn(num_queries, e_size))
            nn.init.uniform_(self.query_embedding, -0.01, 0.01)

    def forward(self, tokens, lengths):

        sentences = [Sentence(i) for i in tokens]

        hidden, queries, memory = self.get_embeddings(sentences, self.query_embedding)

        B = len(lengths)
        max_length = lengths.max()
        mask = (torch.arange(max_length).view(1, -1).repeat(B, 1) < lengths.cpu().unsqueeze(1)).to(
            hidden.device).long()

        return {"embeddings": hidden, "mask": mask, "queries": queries, "cache": memory}

    def compute_token_embeddings(self, sentence_hidden_states, sentences, all_token_subtoken_lengths):

        embedder = self.bert_layer

        all_sentence_embeddings = []

        for sentence_hidden_state, sentence, subtoken_lengths in zip(
                sentence_hidden_states, sentences, all_token_subtoken_lengths
        ):
            subword_start_idx = embedder.begin_offset

            n_layer, _, n_dim = sentence_hidden_state.size()

            sent_emb = torch.zeros(len(sentence), n_layer, n_dim).to(sentence_hidden_state.device)

            for i, (token, n_subtokens) in enumerate(zip(sentence, subtoken_lengths)):
                if n_subtokens == 0:
                    token.set_embedding(embedder.name, torch.zeros(embedder.embedding_length))
                    continue
                subword_end_idx = subword_start_idx + n_subtokens
                assert subword_start_idx < subword_end_idx <= sentence_hidden_state.size()[1]
                current_embeddings = sentence_hidden_state[:, subword_start_idx:subword_end_idx]
                subword_start_idx = subword_end_idx
                if embedder.subtoken_pooling == "first":
                    final_embedding = current_embeddings[:, 0]
                elif embedder.subtoken_pooling == "last":
                    final_embedding = current_embeddings[:, -1]
                elif embedder.subtoken_pooling == "first_last":
                    final_embedding = torch.cat([current_embeddings[:, 0], current_embeddings[:, -1]], dim=1)
                elif embedder.subtoken_pooling == "mean":
                    final_embedding = current_embeddings.mean(dim=1)
                else:
                    raise ValueError(f"Invalid subtoken pooling method: {embedder.subtoken_pooling}")

                sent_emb[i] = final_embedding

            all_sentence_embeddings.append(sent_emb)

        return all_sentence_embeddings

    def get_embeddings(self, sentences, queries=None):

        embedder = self.bert_layer

        tokenized_sentences, all_token_subtoken_lengths, subtoken_lengths = embedder._gather_tokenized_strings(
            sentences)

        # encode inputs
        batch_encoding = embedder.tokenizer(
            tokenized_sentences,
            stride=embedder.stride,
            return_overflowing_tokens=embedder.allow_long_sentences,
            truncation=embedder.truncate,
            padding=True,
            return_tensors="pt",
        )

        input_ids, model_kwargs = embedder._build_transformer_model_inputs(batch_encoding, tokenized_sentences,
                                                                           sentences)

        attention_mask = model_kwargs["attention_mask"]

        gradient_context = torch.enable_grad() if (embedder.fine_tune and embedder.training) else torch.no_grad()

        with gradient_context:

            if queries is not None:
                num_queries, _ = queries.size()

                queries = queries.unsqueeze(0).repeat(input_ids.size(0), 1, 1)

                input_embeddings = embedder.model.get_input_embeddings().forward(input_ids)

                input_embeddings = torch.cat([queries, input_embeddings], dim=1)

                device = next(self.parameters()).device

                attention_mask = torch.cat([torch.ones(input_ids.size(0), num_queries).to(device), attention_mask],
                                           dim=1)

                hidden_states = embedder.model(inputs_embeds=input_embeddings, attention_mask=attention_mask)[-1]

                hidden_states = torch.stack(hidden_states)

                queries, hidden_states = hidden_states[:, :, :num_queries, :], hidden_states[:, :, num_queries:, :]

                queries = queries[-1]

                # revert attention mask
                attention_mask = attention_mask[:, num_queries:]
            else:
                hidden_states = embedder.model(input_ids, attention_mask=attention_mask)[-1]

                # make the tuple a tensor; makes working with it easier.
                hidden_states = torch.stack(hidden_states)

            if embedder.allow_long_sentences:
                sentence_hidden_states = embedder._combine_strided_sentences(
                    hidden_states,
                    sentence_parts_lengths=torch.unique(
                        batch_encoding["overflow_to_sample_mapping"],
                        return_counts=True,
                        sorted=True,
                    )[1].tolist(),
                )
            else:
                sentence_hidden_states = hidden_states.permute((1, 0, 2, 3))

            sentence_hidden_states = [
                sentence_hidden_state[:, : subtoken_length + 1, :]
                for (subtoken_length, sentence_hidden_state) in zip(subtoken_lengths, sentence_hidden_states)
            ]

            all_sentence_embeddings = self.compute_token_embeddings(sentence_hidden_states, sentences,
                                                                    all_token_subtoken_lengths)

        h = pad_sequence(all_sentence_embeddings, batch_first=True, padding_value=0.0)[:, :, -1]

        dict_trans = {"memory": hidden_states[-1], "memory_pad_mask": attention_mask == False}

        return h, queries, dict_trans

#
# # test with main
# if __name__ == '__main__':
#     model = TokenRep(num_queries=3, model_name="bert-base-uncased")
#
#     tokens = ["This is a test", "This is another testhjfkf fhjzfhryb"]
#     lengths = torch.tensor([len(i.split()) for i in tokens])
#
#     tokens = [i.split() for i in tokens]
#
#     out = model(tokens, lengths)
#
#     print(out["embeddings"].shape)
#     print(out["original"][0].shape)
#     print(out["original"][1].shape)
#     print(out["original"][1])
