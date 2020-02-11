from imports import *
from params import *
from model_zoo.layers import *


class QA_Transformer(nn.Module):
    def __init__(self, model, nb_layers=1, pooler_ft=None, use_special_tokens=False):
        super().__init__()
        self.name = model
        ref = None
        self.use_special_tokens = use_special_tokens

        model_class, tokenizer_class, pretrained_weights = TRANSFORMERS[model]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

        self.transformer = model_class.from_pretrained(
            pretrained_weights, output_hidden_states=True
        )
        self.nb_features = self.transformer.pooler.dense.out_features

        with torch.no_grad():
            self.tokenizer.add_tokens(["[q]", "[a]"])

            w = self.transformer.embeddings.word_embeddings.weight
            sep_w = w[102].view(1, -1).detach()
            self.transformer.embeddings.word_embeddings = nn.Embedding.from_pretrained(
                torch.cat([w, sep_w.clone(), sep_w.clone()])
            )

            w = self.transformer.embeddings.token_type_embeddings.weight
            self.transformer.embeddings.token_type_embeddings = nn.Embedding.from_pretrained(
                torch.cat(
                    [
                        w[0].view(1, -1).detach().clone(),
                        w[0].detach().view(1, -1).clone(),
                        w[1].detach().view(1, -1).clone(),
                    ],
                    0,
                )
            )

        if pooler_ft is None or pooler_ft == self.nb_features:
            pooler_ft = self.nb_features
            ref = self.transformer.pooler.dense.weight.detach().clone()

        self.pooler_all = BertMultiPooler(
            nb_layers=nb_layers,
            input_size=self.nb_features,
            nb_ft=pooler_ft,
            weights=ref,
        )

        if self.use_special_tokens:  # Use features corresponding to [q] and [a] tokens
            self.pooler_q = BertMultiPooler(
                nb_layers=nb_layers,
                input_size=self.nb_features,
                nb_ft=pooler_ft,
                weights=ref,
            )

            self.pooler_a = BertMultiPooler(
                nb_layers=nb_layers,
                input_size=self.nb_features,
                nb_ft=pooler_ft,
                weights=ref,
            )

            self.logit = nn.Linear(3 * pooler_ft * nb_layers, len(TARGETS))

        else:
            self.logit = nn.Linear(pooler_ft * nb_layers, len(TARGETS))

    def forward(self, tokens, token_types, positions, q_idx, a_idx):

        _, _, hidden_states = self.transformer(
            tokens, attention_mask=(tokens > 0).long(), token_type_ids=token_types,
        )

        hidden_states = hidden_states[::-1]

        ft = self.pooler_all(hidden_states, 0)

        if self.use_special_tokens:
            ft_q = self.pooler_q(hidden_states, q_idx)
            ft_a = self.pooler_a(hidden_states, a_idx)
            ft = torch.cat([ft, ft_q, ft_a], 1)

        return self.logit(ft)


class QA_TransformerSpecial(nn.Module):
    """ Individual pooler and logits for each targe """

    def __init__(self, model, nb_layers=1, pooler_ft=None):
        super().__init__()
        self.name = model

        model_class, tokenizer_class, pretrained_weights = TRANSFORMERS[model]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

        self.transformer = model_class.from_pretrained(
            pretrained_weights, output_hidden_states=True
        )
        self.nb_features = self.transformer.pooler.dense.out_features

        with torch.no_grad():
            self.tokenizer.add_tokens(["[q]", "[a]"])

            w = self.transformer.embeddings.word_embeddings.weight
            sep_w = w[102].view(1, -1).detach()
            self.transformer.embeddings.word_embeddings = nn.Embedding.from_pretrained(
                torch.cat([w, sep_w.clone(), sep_w.clone()])
            )

            self.tokenizer.add_tokens(SPECIAL_TOKENS)
            cls_w = w[101].view(1, -1).detach()

            w = self.transformer.embeddings.word_embeddings.weight
            self.transformer.embeddings.word_embeddings = nn.Embedding.from_pretrained(
                torch.cat([w] + [cls_w.clone() for _ in range(len(SPECIAL_TOKENS))])
            )

            w = self.transformer.embeddings.token_type_embeddings.weight
            self.transformer.embeddings.token_type_embeddings = nn.Embedding.from_pretrained(
                torch.cat(
                    [
                        w[0].view(1, -1).detach().clone(),
                        w[0].detach().view(1, -1).clone(),
                        w[1].detach().view(1, -1).clone(),
                    ],
                    0,
                )
            )

        if pooler_ft is None:
            pooler_ft = self.nb_features

        self.pooler = nn.ModuleList([])
        self.logit = nn.ModuleList([])

        for i in range(NUM_TARGETS):
            self.pooler.append(
                BertMultiPooler(
                    nb_layers=nb_layers, input_size=self.nb_features, nb_ft=pooler_ft,
                )
            )
            self.logit.append(
                nn.Sequential(nn.Dropout(0.1), nn.Linear(pooler_ft * nb_layers, 1))
            )
    
    def forward(self, tokens, token_types, host, cat):
        _, _, hidden_states = self.transformer(
            tokens, attention_mask=(tokens > 0).long(), token_type_ids=token_types,
        )

        hidden_states = hidden_states[::-1]

        pooled = [self.pooler[i](hidden_states, idx=i) for i in range(NUM_TARGETS)]
        outputs = [self.logit[i](pooled[i]) for i in range(NUM_TARGETS)]

        return torch.cat(outputs, 1)


class QA_Transformer2(nn.Module):
    def __init__(self, model, nb_layers=1, pooler_ft=None, use_special_tokens=False):
        super().__init__()
        self.name = model
        ref = None
        self.use_special_tokens = use_special_tokens

        model_class, tokenizer_class, pretrained_weights = TRANSFORMERS[model]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

        self.transformer = model_class.from_pretrained(
            pretrained_weights, output_hidden_states=True
        )
        self.nb_features = self.transformer.pooler.dense.out_features

        if pooler_ft is None:
            pooler_ft = self.nb_features

        with torch.no_grad():
            self.tokenizer.add_tokens(["[q]", "[a]"])

            w = self.transformer.embeddings.word_embeddings.weight
            sep_w = w[101].view(1, -1).detach()
            self.transformer.embeddings.word_embeddings = nn.Embedding.from_pretrained(
                torch.cat([w, sep_w.clone(), sep_w.clone()])
            )

            w = self.transformer.embeddings.token_type_embeddings.weight
            self.transformer.embeddings.token_type_embeddings = nn.Embedding.from_pretrained(
                torch.cat(
                    [
                        w[0].detach().view(1, -1).clone(),
                        w[0].detach().view(1, -1).clone(),
                        w[1].detach().view(1, -1).clone(),
                    ],
                    0,
                )
            )

        # print(nb_layers, self.nb_features, pooler_ft)
        self.pooler_a = BertMultiPooler(
            nb_layers=nb_layers, input_size=self.nb_features, nb_ft=pooler_ft,
        )

        self.pooler_q = BertMultiPooler(
            nb_layers=nb_layers, input_size=self.nb_features, nb_ft=pooler_ft,
        )

        self.logit = nn.Linear(2 * pooler_ft * nb_layers, len(TARGETS))

    def forward(self, tokens_q, tokens_a, token_types_q, token_types_a):

        _, _, hidden_states_q = self.transformer(
            tokens_q,
            attention_mask=(tokens_q > 0).long(),
            token_type_ids=token_types_q,
        )

        _, _, hidden_states_a = self.transformer(
            tokens_a,
            attention_mask=(tokens_a > 0).long(),
            token_type_ids=token_types_a,
        )

        hidden_states_q = hidden_states_q[::-1]
        hidden_states_a = hidden_states_a[::-1]

        ft_q = self.pooler_q(hidden_states_q, 0)
        ft_a = self.pooler_a(hidden_states_a, 0)

        return self.logit(torch.cat([ft_a, ft_q], -1))


class QA_Transformer2sep(nn.Module):
    def __init__(self, model, nb_layers=1, pooler_ft=None, use_special_tokens=False):
        super().__init__()
        self.name = model
        ref = None
        self.use_special_tokens = use_special_tokens

        model_class, tokenizer_class, pretrained_weights = TRANSFORMERS[model]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

        self.transformer = model_class.from_pretrained(
            pretrained_weights, output_hidden_states=True
        )
        self.nb_features = self.transformer.pooler.dense.out_features

        if pooler_ft is None:
            pooler_ft = self.nb_features

        with torch.no_grad():
            self.tokenizer.add_tokens(["[q]", "[a]"])

            w = self.transformer.embeddings.word_embeddings.weight
            sep_w = w[101].view(1, -1).detach()
            self.transformer.embeddings.word_embeddings = nn.Embedding.from_pretrained(
                torch.cat([w, sep_w.clone(), sep_w.clone()])
            )

            w = self.transformer.embeddings.token_type_embeddings.weight
            self.transformer.embeddings.token_type_embeddings = nn.Embedding.from_pretrained(
                torch.cat(
                    [
                        w[0].detach().view(1, -1).clone(),
                        w[0].detach().view(1, -1).clone(),
                        w[1].detach().view(1, -1).clone(),
                    ],
                    0,
                )
            )

        # print(nb_layers, self.nb_features, pooler_ft)
        self.pooler_a = BertMultiPooler(
            nb_layers=nb_layers, input_size=self.nb_features, nb_ft=pooler_ft,
        )

        self.pooler_q = BertMultiPooler(
            nb_layers=nb_layers, input_size=self.nb_features, nb_ft=pooler_ft,
        )

        self.logit1 = nn.Linear(pooler_ft * nb_layers, 21)
        self.logit2 = nn.Linear(pooler_ft * nb_layers, len(TARGETS) - 21)

    def forward(self, tokens_q, tokens_a, token_types_q, token_types_a):

        _, _, hidden_states_q = self.transformer(
            tokens_q,
            attention_mask=(tokens_q > 0).long(),
            token_type_ids=token_types_q,
        )

        _, _, hidden_states_a = self.transformer(
            tokens_a,
            attention_mask=(tokens_a > 0).long(),
            token_type_ids=token_types_a,
        )

        hidden_states_q = hidden_states_q[::-1]
        hidden_states_a = hidden_states_a[::-1]

        ft_q = self.pooler_q(hidden_states_q, 0)
        ft_a = self.pooler_a(hidden_states_a, 0)

        return torch.cat([self.logit1(ft_q), self.logit2(ft_a)], -1)


class QA_TransformerMix(nn.Module):
    def __init__(self, model, nb_layers=1, pooler_ft=None, use_special_tokens=False):
        super().__init__()
        self.name = model
        self.use_special_tokens = use_special_tokens

        model_class, tokenizer_class, pretrained_weights = TRANSFORMERS[model]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

        self.transformer = model_class.from_pretrained(
            pretrained_weights, output_hidden_states=True
        )
        self.nb_features = self.transformer.pooler.dense.out_features

        if pooler_ft is None:
            pooler_ft = self.nb_features

        with torch.no_grad():
            self.tokenizer.add_tokens(["[q]", "[a]"])

            w = self.transformer.embeddings.word_embeddings.weight
            sep_w = w[101].view(1, -1).detach()
            self.transformer.embeddings.word_embeddings = nn.Embedding.from_pretrained(
                torch.cat([w, sep_w.clone(), sep_w.clone()])
            )

            w = self.transformer.embeddings.token_type_embeddings.weight
            self.transformer.embeddings.token_type_embeddings = nn.Embedding.from_pretrained(
                torch.cat(
                    [
                        w[0].detach().view(1, -1).clone(),
                        w[0].detach().view(1, -1).clone(),
                        w[1].detach().view(1, -1).clone(),
                    ],
                    0,
                )
            )

        self.pooler_a = BertMultiPooler(
            nb_layers=nb_layers, input_size=self.nb_features, nb_ft=pooler_ft,
        )

        self.pooler_q = BertMultiPooler(
            nb_layers=nb_layers, input_size=self.nb_features, nb_ft=pooler_ft,
        )

        # 0 = both, 1 = q, 2 = a   > I have to work on this
        self.mix = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 2]
        assert len(self.mix) == len(TARGETS)

        self.logit = nn.ModuleList([])

        for i in range(len(self.mix)):
            self.logit.append(
                nn.Linear((2 - (self.mix[i] > 0)) * pooler_ft * nb_layers, 1)
            )

    def forward(self, tokens_q, tokens_a, token_types_q, token_types_a):

        _, _, hidden_states_q = self.transformer(
            tokens_q,
            attention_mask=(tokens_q > 0).long(),
            token_type_ids=token_types_q,
        )

        _, _, hidden_states_a = self.transformer(
            tokens_a,
            attention_mask=(tokens_a > 0).long(),
            token_type_ids=token_types_a,
        )

        hidden_states_q = hidden_states_q[::-1]
        hidden_states_a = hidden_states_a[::-1]

        ft_q = self.pooler_q(hidden_states_q, 0)
        ft_a = self.pooler_a(hidden_states_a, 0)

        outs = []

        for i in range(len(self.mix)):
            if self.mix[i] == 0:
                outs.append(self.logit[i](torch.cat([ft_q, ft_a], -1)))
            elif self.mix[i] == 1:
                outs.append(self.logit[i](ft_q))
            else:
                outs.append(self.logit[i](ft_a))

        return torch.cat(outs, -1)


class QA_TransformerDouble(nn.Module):
    def __init__(self, model, nb_layers=1, pooler_ft=None, use_special_tokens=False):
        super().__init__()
        self.name = model
        ref = None
        self.use_special_tokens = use_special_tokens

        model_class, tokenizer_class, pretrained_weights = TRANSFORMERS[model]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

        self.transformer = model_class.from_pretrained(
            pretrained_weights, output_hidden_states=True
        )

        self.transformer2 = model_class.from_pretrained(
            pretrained_weights, output_hidden_states=True
        )

        self.nb_features = self.transformer.pooler.dense.out_features

        if pooler_ft is None:
            pooler_ft = self.nb_features

        self.pooler_a = BertMultiPooler(
            nb_layers=nb_layers, input_size=self.nb_features, nb_ft=pooler_ft,
        )

        self.pooler_q = BertMultiPooler(
            nb_layers=nb_layers, input_size=self.nb_features, nb_ft=pooler_ft,
        )

        # 0 = both, 1 = q, 2 = a
        self.mix = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 2]
        self.mix = [1] * 21 + [2] * 9
        self.mix = [0] * 30 #+ [2] * 9

        assert len(self.mix) == len(TARGETS)

        self.logit = nn.ModuleList([])

        for i in range(len(self.mix)):
            self.logit.append(
                nn.Linear((2 - (self.mix[i] > 0)) * pooler_ft * nb_layers, 1)
            )

    def forward(self, tokens_q, tokens_a, token_types_q, token_types_a):

        _, _, hidden_states_q = self.transformer(
            tokens_q,
            attention_mask=(tokens_q > 0).long(),
            token_type_ids=token_types_q,
        )

        _, _, hidden_states_a = self.transformer2(
            tokens_a,
            attention_mask=(tokens_a > 0).long(),
            token_type_ids=token_types_a,
        )

        hidden_states_q = hidden_states_q[::-1]
        hidden_states_a = hidden_states_a[::-1]

        ft_q = self.pooler_q(hidden_states_q, 0)
        ft_a = self.pooler_a(hidden_states_a, 0)

        outs = []

        for i in range(len(self.mix)):
            if self.mix[i] == 0:
                outs.append(self.logit[i](torch.cat([ft_q, ft_a], -1)))
            elif self.mix[i] == 1:
                outs.append(self.logit[i](ft_q))
            else:
                outs.append(self.logit[i](ft_a))

        return torch.cat(outs, -1)


class QA_TransformerFt(nn.Module):
    def __init__(self, model, nb_layers=1, pooler_ft=None, use_special_tokens=False):
        super().__init__()
        self.name = model
        ref = None

        model_class, tokenizer_class, pretrained_weights = TRANSFORMERS[model]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

        self.transformer = model_class.from_pretrained(
            pretrained_weights, output_hidden_states=True
        )
        self.nb_features = self.transformer.pooler.dense.out_features

        with torch.no_grad():
            self.tokenizer.add_tokens(["[q]", "[a]"])

            w = self.transformer.embeddings.word_embeddings.weight
            sep_w = w[102].view(1, -1).detach()
            self.transformer.embeddings.word_embeddings = nn.Embedding.from_pretrained(
                torch.cat([w, sep_w.clone(), sep_w.clone()])
            )

            w = self.transformer.embeddings.token_type_embeddings.weight
            self.transformer.embeddings.token_type_embeddings = nn.Embedding.from_pretrained(
                torch.cat(
                    [
                        w[0].view(1, -1).detach().clone(),
                        w[0].detach().view(1, -1).clone(),
                        w[1].detach().view(1, -1).clone(),
                    ],
                    0,
                )
            )

        self.pooler_all = BertMultiPooler(
            nb_layers=nb_layers,
            input_size=self.nb_features,
            nb_ft=pooler_ft,
        )

        cat_ft = 64
        host_ft = 64

        self.host_emb = nn.Embedding(64, host_ft)
        self.cat_emb = nn.Embedding(6, cat_ft)

        self.logit = nn.Linear(pooler_ft * nb_layers + host_ft + cat_ft, len(TARGETS))

    def forward(self, tokens, token_types, host, cat):

        _, _, hidden_states = self.transformer(
            tokens, attention_mask=(tokens > 0).long(), token_type_ids=token_types,
        )

        hidden_states = hidden_states[::-1]

        ft = self.pooler_all(hidden_states, 0)

        cat_emb = F.tanh(self.cat_emb(cat))
        host_emb = F.tanh(self.host_emb(host))

        ft = torch.cat((ft, cat_emb, host_emb), 1)

        return self.logit(ft)