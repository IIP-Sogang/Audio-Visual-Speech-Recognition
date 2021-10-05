# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import LinearizedConvolution
from examples.audio_visual_speech_recognition.data.data_utils import lengths_to_encoder_padding_mask
from fairseq.modules import TransformerDecoderLayer, TransformerEncoderLayer, VGGBlock, TransformerBiModalDecoderLayer, MultiheadAttention
from torch import einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import pdb
# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# helper classes
class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)

class ConvolutionModule(nn.Module):
    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(self, x):
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)

class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    Args:
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Normalized tensor.
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return (
            super(LayerNorm, self)
            .forward(x.transpose(self.dim, -1))
            .transpose(self.dim, -1)
        )

class EncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv_module,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x_input, mask, cache=None):

        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            if pos_emb is not None:
                return (x, pos_emb), mask
            return x, mask

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward_macaron(x)
            )
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + stoch_layer_coeff * self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
            self.feed_forward(x)
        )
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask

def get_activation(act):
    """Return activation function."""
    # Lazy load to avoid unused import
    from espnet.nets.pytorch_backend.conformer.swish import Swish

    activation_funcs = {
        "hardtanh": torch.nn.Hardtanh,
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "swish": Swish,
    }

    return activation_funcs[act]()

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):

        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)

class Encoder(torch.nn.Module):
    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="vgg2l",
        normalize_before=True,
        concat_after=False,
        positionwise_conv_kernel_size=1,
        macaron_style=False,
        selfattention_layer_type="selfattn",
        activation_type="swish",
        use_cnn_module=False,
        zero_triu=False,
        cnn_module_kernel=31,
        padding_idx=-1,
        stochastic_depth_rate=0.0,
        intermediate_layers=None,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()

        activation = get_activation(activation_type)
        

        if input_layer == "vgg2l":
            self.embed = VGG2L(idim, attention_dim)
            self.conv_subsampling_factor = 4
        
        self.normalize_before = normalize_before

        # self-attention module definition
        if selfattention_layer_type == "selfattn":
            logging.info("encoder self-attention layer type = self-attention")
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )

        elif selfattention_layer_type == "rel_selfattn":
            logging.info("encoder self-attention layer type = relative self-attention")
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
                zero_triu,
            )

        # feed-forward module definition
        
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
                attention_dim,
                linear_units,
                dropout_rate,
                activation,
            )
        

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate * float(1 + lnum) / num_blocks,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

        self.intermediate_layers = intermediate_layers

    def forward(self, xs, masks):

        if isinstance(self.embed, (Conv2dSubsampling, VGG2L)):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        if self.intermediate_layers is None:
            xs, masks = self.encoders(xs, masks)
        else:
            intermediate_outputs = []
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs, masks = encoder_layer(xs, masks)

                if (
                    self.intermediate_layers is not None
                    and layer_idx + 1 in self.intermediate_layers
                ):
                    # intermediate branches also require normalization.
                    encoder_output = xs
                    if isinstance(encoder_output, tuple):
                        encoder_output = encoder_output[0]
                        if self.normalize_before:
                            encoder_output = self.after_norm(encoder_output)
                    intermediate_outputs.append(encoder_output)

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.normalize_before:
            xs = self.after_norm(xs)

        if self.intermediate_layers is not None:
            return xs, masks, intermediate_outputs
        return xs, masks

class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.
    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
    """

    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(self, x):
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)
        
@register_model("avsr_vggconformer_espnet")
class VGGTransformerModel(BaseFairseqModel):
    """
    Transformers with convolutional context for ASR
    https://arxiv.org/abs/1904.11660
    """
    def __init__(self, audio_encoder, video_encoder, decoder):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.decoder = decoder
    
    def forward(self, audio_src_tokens, audio_src_lengths, video_src_tokens, video_src_lengths, prev_output_tokens, **kwargs):
        audio_encoder_out = self.audio_encoder(audio_src_tokens, src_lengths=audio_src_lengths, **kwargs)
        video_encoder_out = self.video_encoder(video_src_tokens, src_lengths=video_src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, audio_encoder_out=audio_encoder_out, video_encoder_out=video_encoder_out, **kwargs)
        return decoder_out
    
    def forward_decoder(self, prev_output_tokens, audio_encoder_outs, video_encoder_outs, incremental_state, **kwargs):
        return self.decoder(prev_output_tokens, audio_encoder_outs, video_encoder_outs, incremental_state, **kwargs)
    
    def extract_features(self, audio_src_tokens, audio_src_lengths, video_src_tokens, video_src_lengths, prev_output_tokens, **kwargs):
        audio_encoder_out = self.audio_encoder(audio_src_tokens, src_lengths=audio_src_lengths, **kwargs)
        video_encoder_out = self.video_encoder(video_src_tokens, src_lengths=video_src_lengths, **kwargs)
        features = self.decoder.extract_features(prev_output_tokens, audio_encoder_out=audio_encoder_out, video_encoder_out=video_encoder_out, **kwargs)
        return features
 
    def output_layer(self, features, **kwargs):
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        return (self.audio_encoder.max_positions(), self.video_encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        return self.decoder.max_positions()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--input-feat-per-channel",
            type=int,
            metavar="N",
            help="encoder input dimension per input channel",
        )
        parser.add_argument(
            "--vggblock-enc-config",
            type=str,
            metavar="EXPR",
            help="""
    an array of tuples each containing the configuration of one vggblock:
    [(out_channels,
      conv_kernel_size,
      pooling_kernel_size,
      num_conv_layers,
      use_layer_norm), ...])
            """,
        )
        parser.add_argument(
            "--conformer-enc-config",
            type=str,
            metavar="EXPR",
            help=""""
    a tuple containing the configuration of the encoder transformer layers
    configurations
            """,
        )
        parser.add_argument(
            "--enc-output-dim",
            type=int,
            metavar="N",
            help="""
    encoder output dimension, can be None. If specified, projecting the
    transformer output to the specified dimension""",
        )
        parser.add_argument(
            "--in-channels",
            type=int,
            metavar="N",
            help="number of encoder input channels",
        )
        parser.add_argument(
            "--tgt-embed-dim",
            type=int,
            metavar="N",
            help="embedding dimension of the decoder target tokens",
        )
        parser.add_argument(
            "--transformer-dec-config",
            type=str,
            metavar="EXPR",
            help="""
    a tuple containing the configuration of the decoder transformer layers
    configurations:
    [(input_dim,
      num_heads,
      ffn_dim,
      normalize_before,
      dropout,
      attention_dropout,
      relu_dropout), ...]
            """,
        )
        parser.add_argument(
            "--conv-dec-config",
            type=str,
            metavar="EXPR",
            help="""
    an array of tuples for the decoder 1-D convolution config
        [(out_channels, conv_kernel_size, use_layer_norm), ...]""",
        )

    @classmethod
    def build_audio_encoder(cls, args, task):
        return VGGConformerEncoder(
            input_feat_per_channel=args.audio_input_feat_per_channel,
            vggblock_config=eval(args.audio_vggblock_enc_config),
            conformer_config=eval(args.conformer_enc_config),
            encoder_output_dim=args.audio_enc_output_dim,
            in_channels=args.in_channels,
        )
    
    @classmethod
    def build_video_encoder(cls, args, task):
        return VGGConformerEncoder(
            input_feat_per_channel=args.video_input_feat_per_channel,
            vggblock_config=eval(args.video_vggblock_enc_config),
            conformer_config=eval(args.conformer_enc_config),
            encoder_output_dim=args.video_enc_output_dim,
            in_channels=args.in_channels,
        )

    @classmethod
    def build_decoder(cls, args, task):
        return TransformerBiModalDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.tgt_embed_dim,
            transformer_config=eval(args.transformer_dec_config),
            conv_config=eval(args.conv_dec_config),
            encoder_output_dim=args.enc_output_dim,
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted
        # (in case there are any new ones)
        base_architecture(args)

        audio_encoder = cls.build_audio_encoder(args, task)
        video_encoder = cls.build_video_encoder(args, task)
        decoder = cls.build_decoder(args, task)
        return cls(audio_encoder, video_encoder, decoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

DEFAULT_ENC_VGGBLOCK_CONFIG = ((32, 3, 2, 2, False),) * 2
DEFAULT_ENC_TRANSFORMER_CONFIG = ((256, 4, 1024, True, 0.2, 0.2, 0.2),) * 2
# 256: embedding dimension
# 4: number of heads
# 1024: FFN
# True: apply layerNorm before (dropout + resiaul) instead of after
# 0.2 (dropout): dropout after MultiheadAttention and second FC
# 0.2 (attention_dropout): dropout in MultiheadAttention
# 0.2 (relu_dropout): dropout after ReLu
DEFAULT_DEC_TRANSFORMER_CONFIG = ((256, 2, 1024, True, 0.2, 0.2, 0.2),) * 2
DEFAULT_DEC_CONV_CONFIG = ((256, 3, True),) * 2


# TODO: repace transformer encoder config from one liner
# to explicit args to get rid of this transformation
def prepare_conformer_encoder_params(
    encoder_dim,
    num_attention_head,
    feed_forward_expansion_factor,
    conv_expansion_factor,
    feed_forward_dropout,
    attention_dropout,
    conv_dropout,
    conv_kernel_size,
    half_step_residual,
):
    # pdb.set_trace()
    args = argparse.Namespace()
    args.encoder_dim = encoder_dim,
    args.num_attention_head = num_attention_head,
    args.feed_forward_expansion_factor = feed_forward_expansion_factor,
    args.conv_expansion_factor = conv_expansion_factor,
    args.feed_forward_dropout = feed_forward_dropout,
    args.attention_dropout = attention_dropout,
    args.conv_dropout = conv_dropout,
    args.conv_kernel_size = conv_kernel_size,
    args.half_step_residual = half_step_residual,
    return args


def prepare_transformer_decoder_params(
    input_dim,
    num_heads,
    ffn_dim,
    normalize_before,
    dropout,
    attention_dropout,
    relu_dropout,
):
    args = argparse.Namespace()
    args.decoder_embed_dim = input_dim
    args.decoder_attention_heads = num_heads
    args.attention_dropout = attention_dropout
    args.dropout = dropout
    args.activation_dropout = relu_dropout
    args.decoder_normalize_before = normalize_before
    args.decoder_ffn_embed_dim = ffn_dim
    return args


class VGGConformerEncoder(FairseqEncoder):
    """VGG + Transformer encoder"""

    def __init__(
        self,
        input_feat_per_channel,
        vggblock_config,
        conformer_config,
        encoder_output_dim=512,
        in_channels=1,
        conformer_context=None,
        conformer_sampling=None,
        modality=None,
    ):
        """constructor for VGGTransformerEncoder

        Args:
            - input_feat_per_channel: feature dim (not including stacked,
              just base feature)
            - in_channel: # input channels (e.g., if stack 8 feature vector
                together, this is 8)
            - vggblock_config: configuration of vggblock, see comments on
                DEFAULT_ENC_VGGBLOCK_CONFIG
            - transformer_config: configuration of transformer layer, see comments
                on DEFAULT_ENC_TRANSFORMER_CONFIG
            - encoder_output_dim: final transformer output embedding dimension
            - transformer_context: (left, right) if set, self-attention will be focused
              on (t-left, t+right)
            - transformer_sampling: an iterable of int, must match with
              len(transformer_config), transformer_sampling[i] indicates sampling
              factor for i-th transformer layer, after multihead att and feedfoward
              part
        """
        super().__init__(None)
        
        self.num_vggblocks = 0
        if vggblock_config is not None:
            if not isinstance(vggblock_config, Iterable):
                raise ValueError("vggblock_config is not iterable")
            self.num_vggblocks = len(vggblock_config)

        self.conv_layers = nn.ModuleList()
        self.in_channels = in_channels
        self.input_dim = input_feat_per_channel

        if vggblock_config is not None:
            for _, config in enumerate(vggblock_config):
                (
                    out_channels,
                    conv_kernel_size,
                    pooling_kernel_size,
                    num_conv_layers,
                    layer_norm,
                ) = config
                self.conv_layers.append(
                    VGGBlock(
                        in_channels,
                        out_channels,
                        conv_kernel_size,
                        pooling_kernel_size,
                        num_conv_layers,
                        input_dim=input_feat_per_channel,
                        layer_norm=layer_norm,
                    )
                )
                in_channels = out_channels
                input_feat_per_channel = self.conv_layers[-1].output_dim

        conformer_input_dim = self.infer_conv_output_dim(
            self.in_channels, self.input_dim
        ) # transformer_input_dim = 2944 = 23 * 128


        # transformer_input_dim is the output dimension of VGG part

        self.validate_transformer_config(conformer_config) ##error 寃異?
        self.conformer_context = self.parse_transformer_context(conformer_context)
        ##none ?꾩슂?녿뒗寃?媛숈쓬

        self.conformer_sampling = self.parse_transformer_sampling(
            conformer_sampling, len(conformer_config)
        )
        #(1,1,1,1,1,1) ?쒗븯?붿???紐⑤Ⅴ寃좎쓬 


        self.conformer_layers = nn.ModuleList()
        # pdb.set_trace()
        if conformer_input_dim != conformer_config[0][0]:
            self.conformer_layers.append(
                Linear(conformer_input_dim, conformer_config[0][0])
            )
        #?닿굔 誘몄낀?? 2944 -> 144濡?議곗졇二쇰뒗 ??븷 
        
        self.conformer_layers.append(
            ConformerBlock(
                prepare_conformer_encoder_params(*conformer_config[0])
            )
        )
        
        for i in range(1, len(conformer_config)):
            if conformer_config[i - 1][0] != conformer_config[i][0]:
                self.conformer_layers.append(
                    Linear(conformer_config[i - 1][0], conformer_config[i][0])
                )
                ##?꾩뿉嫄??덈룎寃??ㅼ젙?섏뼱?덉쓬
            self.conformer_layers.append(
                ConformerBlock(
                    prepare_conformer_encoder_params(*conformer_config[i])
                )
            )

        self.encoder_output_dim = encoder_output_dim

        self.conformer_layers.extend(
            [
                Linear(conformer_config[-1][0], encoder_output_dim),
                LayerNorm(encoder_output_dim),
            ]
        )

    def forward(self, src_tokens, src_lengths, **kwargs):
        """
        src_tokens: padded tensor (B, T, C * feat)
        src_lengths: tensor of original lengths of input utterances (B,)
        """
        
        bsz, max_seq_len, _ = src_tokens.size() #[B 126 90] = [B T F]
        x = src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim) #[B T 1 90]
        x = x.transpose(1, 2).contiguous() #[B 1 T 90] [B 1 126 90]
        # (B, C, T, feat)

        for layer_idx in range(len(self.conv_layers)):
            x = self.conv_layers[layer_idx](x)

        bsz, _, output_seq_len, _ = x.size() #[B 128 32 23] F異뺤씠 1/4, T異뺣룄 1/4 Channel=128
        # pdb.set_trace()
        # (B, C, T, feat) -> (B, T, C, feat) -> (T, B, C, feat) -> (T, B, C * feat)
        x = x.transpose(1, 2).transpose(0, 1)
        x = x.contiguous().view(output_seq_len, bsz, -1)

        subsampling_factor = int(max_seq_len * 1.0 / output_seq_len + 0.5)
        # TODO: shouldn't subsampling_factor determined in advance ?
        input_lengths = (src_lengths.float() / subsampling_factor).ceil().long()
        input_lengths = torch.clamp(input_lengths, min=0, max=x.size(0))
        encoder_padding_mask, _ = lengths_to_encoder_padding_mask(
            input_lengths, batch_first=True
        )
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        ## encoder_padding_mask = None

        attn_mask = self.lengths_to_attn_mask(input_lengths, subsampling_factor)
        ##attn_mask = none

        conformer_layer_idx = 0

        for layer_idx in range(len(self.conformer_layers)):

            if isinstance(self.conformer_layers[layer_idx], ConformerBlock):
                x = self.conformer_layers[layer_idx](
                    #x, encoder_padding_mask, attn_mask
                    x
                )

                if self.conformer_sampling[conformer_layer_idx] != 1:
                    sampling_factor = self.conformer_sampling[conformer_layer_idx]
                    x, encoder_padding_mask, attn_mask = self.slice(
                        x, encoder_padding_mask, attn_mask, sampling_factor
                    )

                conformer_layer_idx += 1

            else:
                x = self.conformer_layers[layer_idx](x)

        # encoder_padding_maks is a (T x B) tensor, its [t, b] elements indicate
        # whether encoder_output[t, b] is valid or not (valid=0, invalid=1)

        return {
            "encoder_out": x,  # (T, B, C)
            "encoder_padding_mask": encoder_padding_mask.t()
            if encoder_padding_mask is not None
            else None,
            # (B, T) --> (T, B)
        }

    def infer_conv_output_dim(self, in_channels, input_dim):
        sample_seq_len = 200
        sample_bsz = 10
        x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
        for i, _ in enumerate(self.conv_layers):
            x = self.conv_layers[i](x)
        x = x.transpose(1, 2)
        mb, seq = x.size()[:2]
        return x.contiguous().view(mb, seq, -1).size(-1)

    def validate_transformer_config(self, transformer_config):
        # pdb.set_trace()
        for config in transformer_config:
            input_dim, num_heads = config[:2]
            if input_dim % num_heads != 0:
                msg = (
                    "ERROR in transformer config {}:".format(config)
                    + "input dimension {} ".format(input_dim)
                    + "not dividable by number of heads".format(num_heads)
                )
                raise ValueError(msg)

    def parse_transformer_context(self, transformer_context):
        """
        transformer_context can be the following:
        -   None; indicates no context is used, i.e.,
            transformer can access full context
        -   a tuple/list of two int; indicates left and right context,
            any number <0 indicates infinite context
                * e.g., (5, 6) indicates that for query at x_t, transformer can
                access [t-5, t+6] (inclusive)
                * e.g., (-1, 6) indicates that for query at x_t, transformer can
                access [0, t+6] (inclusive)
        """
        if transformer_context is None:
            return None

        if not isinstance(transformer_context, Iterable):
            raise ValueError("transformer context must be Iterable if it is not None")

        if len(transformer_context) != 2:
            raise ValueError("transformer context must have length 2")

        left_context = transformer_context[0]
        if left_context < 0:
            left_context = None

        right_context = transformer_context[1]
        if right_context < 0:
            right_context = None

        if left_context is None and right_context is None:
            return None

        return (left_context, right_context)

    def parse_transformer_sampling(self, transformer_sampling, num_layers):
        """
        parsing transformer sampling configuration

        Args:
            - transformer_sampling, accepted input:
                * None, indicating no sampling
                * an Iterable with int (>0) as element
            - num_layers, expected number of transformer layers, must match with
              the length of transformer_sampling if it is not None

        Returns:
            - A tuple with length num_layers
        """
        if transformer_sampling is None:
            return (1,) * num_layers

        if not isinstance(transformer_sampling, Iterable):
            raise ValueError(
                "transformer_sampling must be an iterable if it is not None"
            )

        if len(transformer_sampling) != num_layers:
            raise ValueError(
                "transformer_sampling {} does not match with the number "
                + "of layers {}".format(transformer_sampling, num_layers)
            )

        for layer, value in enumerate(transformer_sampling):
            if not isinstance(value, int):
                raise ValueError("Invalid value in transformer_sampling: ")
            if value < 1:
                raise ValueError(
                    "{} layer's subsampling is {}.".format(layer, value)
                    + " This is not allowed! "
                )
        return transformer_sampling

    def slice(self, embedding, padding_mask, attn_mask, sampling_factor):
        """
        embedding is a (T, B, D) tensor
        padding_mask is a (B, T) tensor or None
        attn_mask is a (T, T) tensor or None
        """
        embedding = embedding[::sampling_factor, :, :]
        if padding_mask is not None:
            padding_mask = padding_mask[:, ::sampling_factor]
        if attn_mask is not None:
            attn_mask = attn_mask[::sampling_factor, ::sampling_factor]

        return embedding, padding_mask, attn_mask

    def lengths_to_attn_mask(self, input_lengths, subsampling_factor=1):
        """
        create attention mask according to sequence lengths and transformer
        context

        Args:
            - input_lengths: (B, )-shape Int/Long tensor; input_lengths[b] is
              the length of b-th sequence
            - subsampling_factor: int
                * Note that the left_context and right_context is specified in
                  the input frame-level while input to transformer may already
                  go through subsampling (e.g., the use of striding in vggblock)
                  we use subsampling_factor to scale the left/right context

        Return:
            - a (T, T) binary tensor or None, where T is max(input_lengths)
                * if self.transformer_context is None, None
                * if left_context is None,
                    * attn_mask[t, t + right_context + 1:] = 1
                    * others = 0
                * if right_context is None,
                    * attn_mask[t, 0:t - left_context] = 1
                    * others = 0
                * elsif
                    * attn_mask[t, t - left_context: t + right_context + 1] = 0
                    * others = 1
        """
        if self.conformer_context is None:
            return None

        maxT = torch.max(input_lengths).item()
        attn_mask = torch.zeros(maxT, maxT)

        left_context = self.conformer_context[0]
        right_context = self.conformer_context[1]
        if left_context is not None:
            left_context = math.ceil(self.conformer_context[0] / subsampling_factor)
        if right_context is not None:
            right_context = math.ceil(self.conformer_context[1] / subsampling_factor)

        for t in range(maxT):
            if left_context is not None:
                st = 0
                en = max(st, t - left_context)
                attn_mask[t, st:en] = 1
            if right_context is not None:
                st = t + right_context + 1
                st = min(st, maxT - 1)
                attn_mask[t, st:] = 1

        return attn_mask.to(input_lengths.device)

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
            1, new_order
        )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(1, new_order)
        return encoder_out


class TransformerBiModalDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerBiModalDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """

    def __init__(
        self,
        dictionary,
        embed_dim=512,
        transformer_config=DEFAULT_ENC_TRANSFORMER_CONFIG,
        conv_config=DEFAULT_DEC_CONV_CONFIG,
        encoder_output_dim=512,
    ):

        super().__init__(dictionary)
        vocab_size = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(vocab_size, embed_dim, self.padding_idx)

        self.conv_layers = nn.ModuleList()
        if conv_config is not None:
            for i in range(len(conv_config)):
                out_channels, kernel_size, layer_norm = conv_config[i]
                if i == 0:
                    conv_layer = LinearizedConv1d(
                        embed_dim, out_channels, kernel_size, padding=kernel_size - 1
                    )
                else:
                    conv_layer = LinearizedConv1d(
                        conv_config[i - 1][0],
                        out_channels,
                        kernel_size,
                        padding=kernel_size - 1,
                    )
                self.conv_layers.append(conv_layer)
                if layer_norm:
                    self.conv_layers.append(nn.LayerNorm(out_channels))
                self.conv_layers.append(nn.ReLU())

        self.layers = nn.ModuleList()
        if conv_config is not None:
            if conv_config[-1][0] != transformer_config[0][0]:
                self.layers.append(Linear(conv_config[-1][0], transformer_config[0][0]))
        self.layers.append(TransformerBiModalDecoderLayer(
            prepare_transformer_decoder_params(*transformer_config[0])
        ))

        for i in range(1, len(transformer_config)):
            if transformer_config[i - 1][0] != transformer_config[i][0]:
                self.layers.append(
                    Linear(transformer_config[i - 1][0], transformer_config[i][0])
                )
            self.layers.append(TransformerBiModalDecoderLayer(
                prepare_transformer_decoder_params(*transformer_config[i])
            ))
        self.fc_out = Linear(transformer_config[-1][0], vocab_size)
    
    def extract_features(self, prev_output_tokens, audio_encoder_out=None, video_encoder_out=None, incremental_state=None):
        raise NotImplementedError

    def forward(self, prev_output_tokens, audio_encoder_out=None, video_encoder_out=None, incremental_state=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            audio_encoder_out or video_encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        target_padding_mask = (
            (prev_output_tokens == self.padding_idx).to(prev_output_tokens.device)
            if incremental_state is None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)

        # B x T x C -> T x B x C
        x = self._transpose_if_training(x, incremental_state)

        for layer in self.conv_layers:
            if isinstance(layer, LinearizedConvolution):
                x = layer(x, incremental_state)
            else:
                x = layer(x)

        # B x T x C -> T x B x C
        x = self._transpose_if_inference(x, incremental_state)

        # decoder layers
        for layer in self.layers:
            if isinstance(layer, TransformerBiModalDecoderLayer):
                x, _ = layer(
                    x,
                    (audio_encoder_out["encoder_out"] if audio_encoder_out is not None else None),
                    (video_encoder_out["encoder_out"] if video_encoder_out is not None else None),
                    (
                        audio_encoder_out["encoder_padding_mask"].t()
                        if audio_encoder_out["encoder_padding_mask"] is not None
                        else None
                    ),
                    (
                        video_encoder_out["encoder_padding_mask"].t()
                        if video_encoder_out["encoder_padding_mask"] is not None
                        else None
                    ),
                    incremental_state,
                    self_attn_mask=(
                        self.buffered_future_mask(x)
                        if incremental_state is None
                        else None
                    ),
                    self_attn_padding_mask=(
                        target_padding_mask if incremental_state is None else None
                    ),
                )
            else:
                x = layer(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        x = self.fc_out(x)

        return x, None

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def _transpose_if_training(self, x, incremental_state):
        if incremental_state is None:
            x = x.transpose(0, 1)
        return x

    def _transpose_if_inference(self, x, incremental_state):
        if incremental_state is not None:
            x = x.transpose(0, 1)
        return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    # nn.init.uniform_(m.weight, -0.1, 0.1)
    # nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    # m.weight.data.uniform_(-0.1, 0.1)
    # if bias:
    #     m.bias.data.uniform_(-0.1, 0.1)
    return m


def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


# seq2seq models
def base_architecture(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 40)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", DEFAULT_ENC_VGGBLOCK_CONFIG
    )
    args.transformer_enc_config = getattr(
        args, "transformer_enc_config", DEFAULT_ENC_TRANSFORMER_CONFIG
    )
    args.enc_output_dim = getattr(args, "enc_output_dim", 512)
    args.in_channels = getattr(args, "in_channels", 1)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 128)
    args.transformer_dec_config = getattr(
        args, "transformer_dec_config", DEFAULT_ENC_TRANSFORMER_CONFIG
    )
    args.conv_dec_config = getattr(args, "conv_dec_config", DEFAULT_DEC_CONV_CONFIG)
    args.transformer_context = getattr(args, "transformer_context", "None")

#
#@register_model_architecture("asr_vggtransformer", "vggtransformer_1")
#def vggtransformer_1(args):
#    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
#    args.vggblock_enc_config = getattr(
#        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
#    )
#    args.transformer_enc_config = getattr(
#        args,
#        "transformer_enc_config",
#        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 14",
#    )
#    args.enc_output_dim = getattr(args, "enc_output_dim", 1024)
#    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 128)
#    args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
#    args.transformer_dec_config = getattr(
#        args,
#        "transformer_dec_config",
#        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 4",
#    )
#
#
#@register_model_architecture("asr_vggtransformer", "vggtransformer_2")
#def vggtransformer_2(args):
#    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
#    args.vggblock_enc_config = getattr(
#        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
#    )
#    args.transformer_enc_config = getattr(
#        args,
#        "transformer_enc_config",
#        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 16",
#    )
#    args.enc_output_dim = getattr(args, "enc_output_dim", 1024)
#    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 512)
#    args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
#    args.transformer_dec_config = getattr(
#        args,
#        "transformer_dec_config",
#        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 6",
#    )
#
#
#@register_model_architecture("asr_vggtransformer", "vggtransformer_base")
#def vggtransformer_base(args):
#    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
#    args.vggblock_enc_config = getattr(
#        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
#    )
#    args.transformer_enc_config = getattr(
#        args, "transformer_enc_config", "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 12"
#    )
#
#    args.enc_output_dim = getattr(args, "enc_output_dim", 512)
#    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 512)
#    args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
#    args.transformer_dec_config = getattr(
#        args, "transformer_dec_config", "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 6"
#    )
#    # Size estimations:
#    # Encoder:
#    #   - vggblock param: 64*1*3*3 + 64*64*3*3 + 128*64*3*3  + 128*128*3 = 258K
#    #   Transformer:
#    #   - input dimension adapter: 2560 x 512 -> 1.31M
#    #   - transformer_layers (x12) --> 37.74M
#    #       * MultiheadAttention: 512*512*3 (in_proj) + 512*512 (out_proj) = 1.048M
#    #       * FFN weight: 512*2048*2 = 2.097M
#    #   - output dimension adapter: 512 x 512 -> 0.26 M
#    # Decoder:
#    #   - LinearizedConv1d: 512 * 256 * 3 + 256 * 256 * 3 * 3
#    #   - transformer_layer: (x6) --> 25.16M
#    #        * MultiheadAttention (self-attention): 512*512*3 + 512*512 = 1.048M
#    #        * MultiheadAttention (encoder-attention): 512*512*3 + 512*512 = 1.048M
#    #        * FFN: 512*2048*2 = 2.097M
#    # Final FC:
#    #   - FC: 512*5000 = 256K (assuming vocab size 5K)
#    # In total:
#    #       ~65 M

@register_model_architecture("avsr_vggconformer_espnet", "BiModalvggconformer_espnet_avsr_base")
def BiModalvggtransformer_avsr_base(args):
    args.audio_input_feat_per_channel = getattr(args, "audio_input_feat_per_channel", 90)
    args.video_input_feat_per_channel = getattr(args, "video_input_feat_per_channel", 512)
    args.audio_vggblock_enc_config = getattr(
        args, "audio_vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )
    args.video_vggblock_enc_config = getattr(
        args, "video_vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )   
#    args.audio_vggblock_enc_config = 'None'
#    args.video_vggblock_enc_config = 'None'
    args.conformer_enc_config = getattr(
        args, "conformer_enc_config", "((256, 8, 8, 2, 0.1, 0.1, 0.1, 31, True),) * 6" #encoder_dim , num_head, feed_expansion
    )
    args.audio_enc_output_dim = getattr(args, "audio_enc_output_dim", 512)
    args.video_enc_output_dim = getattr(args, "video_enc_output_dim", 512)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 512)
    args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
#    args.conv_dec_config = 'None'
    args.transformer_dec_config = getattr(
        args, "transformer_dec_config", "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 6"
    )


 
