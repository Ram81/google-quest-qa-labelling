import torch
import torch.nn as nn
import torch.nn.functional as F


class BertMultiPooler(nn.Module):
    def __init__(
        self, nb_layers=1, input_size=768, nb_ft=768, drop_p=0.1, weights=None
    ):
        super().__init__()

        self.nb_layers = nb_layers
        self.input_size = input_size
        self.poolers = nn.ModuleList([])

        for i in range(nb_layers):
            pooler = nn.Sequential(
                nn.Linear(input_size, nb_ft),
                # nn.Dropout(drop_p),
                nn.Tanh(),
            )

            if weights is not None:
                with torch.no_grad():
                    pooler[0].weight = nn.Parameter(weights.clone())
                    # print('loaded')
            self.poolers.append(pooler)

    def forward(self, hidden_states, idx=0):
        bs = hidden_states[0].size()[0]
        if type(idx) == int:
            idx = torch.tensor([idx] * bs).cuda()

        outputs = []
        idx = idx.view(-1, 1, 1).repeat(1, 1, self.input_size)

        for i, (state) in enumerate(hidden_states[: self.nb_layers]):
            token_tensor = state.gather(1, idx).view(bs, -1)

            pooled = self.poolers[i](token_tensor)
            outputs.append(pooled)

        return torch.cat(outputs, -1)


class GGNN(nn.Module):
    def __init__(self, input_features, A_in, A_out, c=20, time_steps=3):
        super().__init__()

        self.input_features = input_features
        self.time_steps = time_steps

        self.c = c
        self.d = A_in.shape[0]
        self.n_nodes = A_in.shape[1]

        self.A_in = torch.tensor(A_in).float().cuda()
        self.A_out = torch.tensor(A_out).float().cuda()

        self.in_fcs = nn.ModuleList([])
        self.out_fcs = nn.ModuleList([])
        for i in range(self.d):
            self.in_fcs.append(nn.Linear(input_features, input_features))
            self.out_fcs.append(nn.Linear(input_features, input_features))

        self.Wz = nn.Linear(2 * input_features, input_features, bias=False)
        self.Uz = nn.Linear(input_features, input_features, bias=False)
        self.Wr = nn.Linear(2 * input_features, input_features, bias=False)
        self.Ur = nn.Linear(input_features, input_features, bias=False)
        self.W = nn.Linear(2 * input_features, input_features, bias=False)
        self.U = nn.Linear(input_features, input_features, bias=False)

    def forward(self, xv):
        bs, n_nodes, input_fts = xv.size()
        h = xv

        in_states = []
        out_states = []
        for i in range(self.d):
            in_states.append(self.in_fcs[i](h))
            out_states.append(self.out_fcs[i](h))

        in_states = torch.cat(in_states, 1)  # (bs, n_nodes * d,  input_fts)
        out_states = torch.cat(out_states, 1)

        A_in = (
            self.A_in.view(1, -1, self.n_nodes).transpose(1, 2).repeat(bs, 1, 1)
        )  # (bs, n_nodes, n_nodes * d)
        A_out = (
            self.A_out.view(1, -1, self.n_nodes).transpose(1, 2).repeat(bs, 1, 1)
        )  # (bs, n_nodes, n_nodes * d)

        for t in range(self.time_steps):
            a_in = torch.bmm(A_in, in_states)
            a_out = torch.bmm(A_out, out_states)

            a = torch.cat([a_in, a_out], 2)
            z = F.sigmoid(self.Wz(a) + self.Uz(h))
            r = F.sigmoid(self.Wr(a) + self.Ur(h))

            h_tilde = F.tanh(self.W(a) + self.U(r * h))
            h = (1 - z) * h + (z * h_tilde)

        return h
