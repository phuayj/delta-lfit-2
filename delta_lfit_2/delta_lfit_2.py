from delta_lfit_2.logic import length, get_max_rule_len
from delta_lfit_2.set_transformer import (
    SetTransformerEncoder,
    SetTransformerDecoder,
)

import torch
from torch import nn


class InputEmbeddings(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(InputEmbeddings, self).__init__()
        self.activation = nn.ReLU()
        self.ln = nn.LayerNorm(dim_out)
        self.embed = nn.Embedding(dim_in, dim_out)

    def forward(self, X):
        return self.ln(self.activation(self.embed(X)))


class ProgramPredictor(nn.Module):
    def __init__(
        self, dim_in, dim_hidden, layers, num_vars, delays,
        dropout=0., ln=False,
    ):
        super(ProgramPredictor, self).__init__()

        max_rule_len = get_max_rule_len(num_vars, delays)

        modules = [
            nn.Linear(dim_in, dim_hidden),
        ]
        for _ in range(layers):
            modules.append(nn.Linear(dim_hidden, dim_hidden))
            if ln:
                modules.append(nn.LayerNorm(dim_hidden))
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*modules)
        self.rules_pred = nn.ModuleList([
            nn.Linear(dim_hidden, length(0, num_vars*delays) + 1),
        ])
        self.zeros_pad = [
            torch.zeros(1, max_rule_len - length(0, num_vars*delays)),
        ]
        self.register_buffer('zeros_pad_0', self.zeros_pad[0])
        for i in range(1, num_vars*delays + 1):
            l = length(i, num_vars*delays)
            self.rules_pred.append(nn.Linear(
                dim_hidden,
                l - length(i-1, num_vars*delays) + 1,
            ))
            self.zeros_pad.append(torch.zeros(
                1, max_rule_len - l + length(i-1, num_vars*delays),
            ))
            self.register_buffer(f'zeros_pad_{i}', self.zeros_pad[i])

    def forward(self, latent, rule_len, rule_len_is_diff=False):
        output = self.layers(latent)
        if rule_len_is_diff:
            # This is a bit slow, but it's only used during inference
            output_mat = []
            for i, l in enumerate(rule_len):
                out = self.rules_pred[l.item()](output[i])
                out = torch.cat((
                    out[:,:-1], getattr(self, f'zeros_pad_{l.item()}'), out[:,-1:],
                ), dim=1)
                output_mat.append(out)
            output = torch.stack(output_mat)
        else:
            # This is the training path
            out = self.rules_pred[rule_len[0].item()](output)
            zeros_pad = getattr(self, f'zeros_pad_{rule_len[0].item()}')
            zeros_pad = zeros_pad.expand(
                output.shape[0],
                getattr(self, f'zeros_pad_{rule_len[0].item()}').shape[1],
            ).unsqueeze(1)
            output = torch.cat((out[:,:,:-1], zeros_pad, out[:,:,-1:]), dim=2)
        return output


class DeltaLFIT2(nn.Module):
    def __init__(
        self,
        num_vars,
        delays,
        set_transformer_encoder_config,
        set_transformer_decoder_config,
        program_predictor_config,
    ):
        super(DeltaLFIT2, self).__init__()
        self.transition_embedding = InputEmbeddings(
            2 ** (num_vars*delays) * 2,
            set_transformer_encoder_config['dim_in'],
        )
        self.encoder = SetTransformerEncoder(**set_transformer_encoder_config)
        self.decoder = SetTransformerDecoder(**set_transformer_decoder_config)

        program_predictor_config['num_vars'] = num_vars
        program_predictor_config['delays'] = delays
        self.program_predictor = ProgramPredictor(
            **program_predictor_config,
        )

    def forward(self, transitions, rule_len, rule_len_is_diff=False):
        embeddings = self.transition_embedding(transitions)
        latent = self.decoder(self.encoder(embeddings))
        rules_probability = self.program_predictor(
            latent, rule_len, rule_len_is_diff,
        )

        return rules_probability
