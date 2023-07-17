from embeddings import PositionalEncoding
from utils.pad import pad_masking, subsequent_masking
from performer_pytorch.performer_pytorch_opt import SelfAttention

import torch
from torch import nn
import numpy as np
from collections import defaultdict

PAD_TOKEN_ID = 0

def build_model(config):

    encoder = TransformerEncoder(
        d_input = config['d_input'],
        layers_count=config['layers_count'],
        d_model=config['d_model'],
        heads_count=config['heads_count'],
        d_ff=config['d_ff'],
        dropout_prob=config['dropout_prob'])

    model = Transformer(encoder, config['d_model'], config['num_clusters'])

    return model


class Transformer(nn.Module):

    def __init__(self, encoder, d_model, num_clusters):
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.fc = torch.nn.Linear(d_model, num_clusters)
        self.sm = torch.nn.Softmax(dim=1)
        # self.decoder = decoder

    def forward(self, sources):
        # sources : (batch_size, sources_len)
        # inputs : (batch_size, targets_len - 1)
        # batch_size, sources_len = sources.size()
        # batch_size, inputs_len = inputs.size()

        # sources_mask = pad_masking(sources, sources_len)
        # memory_mask = pad_masking(sources, inputs_len)
        # inputs_mask = subsequent_masking(inputs) | pad_masking(inputs, inputs_len)

        emb, rep_loss = self.encoder(sources)  # (batch_size, seq_len, d_model)
        yhat = torch.squeeze(self.fc(emb))
        # outputs, state = self.decoder(inputs, memory, memory_mask, inputs_mask)  # (batch_size, seq_len, d_model)

        return torch.squeeze(emb), yhat, self.sm(yhat), rep_loss


class TransformerEncoder(nn.Module):

    def __init__(self, d_input, layers_count, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.d_input = d_input
        # self.dimension_reduction = Dimension_Reduction(d_input, d_model, dropout_prob)

        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads_count, d_ff, dropout_prob) for _ in range(layers_count)]
        )

    def forward(self, sources, mask=None):
        """

        args:
           sources: embedded_sequence, (batch_size, seq_len, embed_size)
        """
        # sources = self.dimension_reduction(sources)  # dimenstion reduction
        total_rep_loss = []
        for encoder_layer in self.encoder_layers:
            sources, rep_loss = encoder_layer(sources)
            total_rep_loss.append(rep_loss)

        return sources, sum(total_rep_loss)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()

        # self.self_attention_layer = Sublayer(MultiHeadAttention(heads_count, d_model, dropout_prob), d_model)
        # self.self_attention_layer = Sublayer1(SelfAttention(dim = d_model, heads = heads_count, causal = False,).cuda(), d_model)
        self.self_attention_layer = Sublayer1(SelfAttention(dim = d_model, heads = heads_count, causal = False,), d_model)

        self.pointwise_feedforward_layer = Sublayer2(PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, sources, sources_mask=None):
        # x: (batch_size, seq_len, d_model)
        import pdb; pdb.set_trace()
        sources, rep_loss = self.self_attention_layer(sources, sources, sources, sources_mask)
        sources = self.dropout(sources)
        sources = self.pointwise_feedforward_layer(sources)
        return sources, rep_loss

class Sublayer1(nn.Module):
    def __init__(self, sublayer, d_model):
        super(Sublayer1, self).__init__()
        self.sublayer = sublayer
        self.layer_normalization = LayerNormalization(d_model)

    def forward(self, *args):
        x = args[0]
        x_temp , rep_loss = self.sublayer(*args)
        x = x_temp[0] + x
        return self.layer_normalization(x), rep_loss

class Sublayer2(nn.Module):
    def __init__(self, sublayer, d_model):
        super(Sublayer2, self).__init__()
        self.sublayer = sublayer
        self.layer_normalization = LayerNormalization(d_model)

    def forward(self, *args):
        x = args[0]
        x = self.sublayer(*args) + x
        return self.layer_normalization(x)

class LayerNormalization(nn.Module):

    def __init__(self, features_count, epsilon=1e-6):
        super(LayerNormalization, self).__init__()

        self.gain = nn.Parameter(torch.ones(features_count))
        self.bias = nn.Parameter(torch.zeros(features_count))
        self.epsilon = epsilon

    def forward(self, x):

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gain * (x - mean) / (std + self.epsilon) + self.bias


class PointwiseFeedForwardNetwork(nn.Module):

    def __init__(self, d_ff, d_model, dropout_prob):
        super(PointwiseFeedForwardNetwork, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        """

        Args:
             x: (batch_size, seq_len, d_model)
        """
        return self.feed_forward(x)

class Dimension_Reduction(nn.Module):

    def __init__(self, d_input, d_model, dropout_prob):
        super(Dimension_Reduction, self).__init__()

        self.dimension_reduction = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        """

        Args:
             x: (batch_size, seq_len, d_model)
        """
        return self.dimension_reduction(x)
