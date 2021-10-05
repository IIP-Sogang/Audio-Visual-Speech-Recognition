# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
from collections.abc import Iterable
import pdb
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
from examples.speech_recognition_rnnt.data.data_utils import lengths_to_encoder_padding_mask
from fairseq.modules import TransformerDecoderLayer, TransformerEncoderLayer, VGGBlock, MultiheadAttention
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

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# attention, feedforward, and conv module

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        dropout,
        max_pos_emb,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None, mask = None, context_mask = None):
        
        x = x.permute(1,0,2) #T B C -> B T C
        #x.shape = [9,45,144] [B T F]
        time, device, heads, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x) #context = x

        query, key, value = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        #[B T 144]
        query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = heads), (query, key, value))
        #[B 4 T 36]
        dots = einsum('b h i d, b h j d -> b h i j', query, key) * self.scale
        ##[9 4 T T] [9 4 45 45]


        # shaw's relative positional embedding
        seq = torch.arange(time, device = device)
        #seq = [0,1,2,3,4,..,time-1]

        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        #rearrange(seq,'i->i()').shape=[9,1],rearrange(seq, 'j -> () j').shape = [1,9]

        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        #dist.shape [45 45]
        rel_pos_emb = self.rel_pos_emb(dist).to(query)
        ##self.rel_pos_emb(dist).shape [45 45 36], query.shape[9,4,45,36]
        ##rel_pos_emb.shape = [45,45,36] 


        pos_attn = einsum('b h n d, n r d -> b h n r', query, rel_pos_emb) * self.scale
        ##[9 4 45 45]


        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, value)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.dropout(out)
        out = out.permute(1,0,2)
        return out

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        expansion_factor,
        kernel_size,
        dropout,
        causal=False,
        ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('T B C -> B C T'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('B C T -> T B C'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        
        self.dim = args.encoder_embed_dim
        self.dim_head = args.encoder_embed_dim_head
        self.heads = args.encoder_attention_heads
        self.ff_mult = args.encoder_ff_mult
        self.conv_expansion_factor = args.encoder_conv_expansion_factor
        self.conv_kernel_size = args.encoder_conv_kernel_size
        self.attn_dropout = args.attention_dropout
        self.ff_dropout = args.ff_dropout
        self.conv_dropout = args.conv_dropout
        self.max_pos_emb = args.max_pos_emb

        self.ff1 = FeedForward(dim = self.dim, mult = self.ff_mult, dropout = self.ff_dropout)
        self.attn = Attention(dim = self.dim, dim_head = self.dim_head, heads = self.heads, dropout = self.attn_dropout,max_pos_emb = self.max_pos_emb )
        self.conv = ConformerConvModule(dim = self.dim, causal = False, expansion_factor = self.conv_expansion_factor, kernel_size = self.conv_kernel_size, dropout = self.conv_dropout)
        self.ff2 = FeedForward(dim = self.dim, mult = self.ff_mult, dropout = self.ff_dropout)

        self.attn = PreNorm(self.dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(self.dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(self.dim, self.ff2))

        self.post_norm = nn.LayerNorm(self.dim)

    def forward(self, x, mask = None):
        x = self.ff1(x) + x
        x = self.attn(x, mask = mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x



@register_model("libri_vggconformer_rnnt2")
class VGGConformerModel_audio_only(BaseFairseqModel):
    """
    Transformers with convolutional context for ASR
    https://arxiv.org/abs/1904.11660
    """
    def __init__(self, audio_encoder, video_encoder, decoder,jointnet):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.video_encoder = None
        self.decoder = decoder
        self.jointnet = jointnet


    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        audio_encoder_out = self.audio_encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        video_encoder_out = None
        decoder_out = self.decoder(prev_output_tokens, encoder_out=None,**kwargs)
        joint_out = self.jointnet(audio_encoder_out,decoder_out,**kwargs)
        return joint_out

    def forward_encoder(self, src_tokens, src_lengths, **kwargs):
        audio_encoder_out = self.audio_encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        return audio_encoder_out

    def forward_decoder(self, prev_output_tokens, audio_encoder_outs, video_encoder_outs, incremental_state, **kwargs):
        decoder_out = self.decoder(prev_output_tokens, encoder_out=None,**kwargs)
        return decoder_out

    def forward_jointnet(self, prev_output_tokens, audio_encoder_outs, video_encoder_outs, incremental_state, **kwargs):
        joint_out = self.jointnet(audio_encoder_out,decoder_out,**kwargs)
        return joint_out
    
    def extract_features(self, audio_src_tokens, audio_src_lengths, video_src_tokens, video_src_lengths, prev_output_tokens, **kwargs):
        audio_encoder_out = self.audio_encoder(audio_src_tokens, src_lengths=audio_src_lengths, **kwargs)
        video_encoder_out = None
        features = self.decoder.extract_features(prev_output_tokens, audio_encoder_out=audio_encoder_out, **kwargs)
        return features
 
    def output_layer(self, features, **kwargs):
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        return (self.audio_encoder.max_positions(), 1e6, self.decoder.max_positions())

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
            "--lstm-dec-config",
            type=str,
            metavar="EXPR",
            help="""
    a tuple containing the configuration of the decoder transformer layers
    configurations"""
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
        return None
#        return VGGTransformerEncoder(
#            input_feat_per_channel=args.video_input_feat_per_channel,
#            vggblock_config=eval(args.video_vggblock_enc_config),
#            transformer_config=eval(args.transformer_enc_config),
#            encoder_output_dim=args.video_enc_output_dim,
#            in_channels=args.in_channels,
#        )

    @classmethod
    def build_decoder(cls, args, task):
        return VGGLSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.tgt_embed_dim,
            lstm_config=eval(args.lstm_dec_config),
            encoder_output_dim=args.audio_enc_output_dim,
        )
    
    @classmethod
    def build_jointnet(cls, args, task):
        return Jointnet(
            dictionary=task.target_dictionary,
            embed_dim=args.tgt_embed_dim,
            lstm_config=eval(args.lstm_dec_config),
            encoder_output_dim=args.audio_enc_output_dim,
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
        jointnet = cls.build_jointnet(args, task)
        return cls(audio_encoder, video_encoder, decoder,jointnet)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

DEFAULT_ENC_VGGBLOCK_CONFIG = ((32, 3, 2, 2, False),) * 2
DEFAULT_ENC_TRANSFORMER_CONFIG = ((144, 36, 4, 4, 2, 32, 0.1,0.1,0.1),) * 2
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
    dim,
    dim_head,
    heads,
    ff_mult,
    conv_expansion_factor,
    conv_kernel_size,
    attn_dropout,
    ff_dropout,
    conv_dropout,
    max_pos_emb,
):
    
    args = argparse.Namespace()
    args.encoder_embed_dim = dim
    args.encoder_embed_dim_head = dim_head
    args.encoder_attention_heads = heads
    args.encoder_ff_mult = ff_mult
    args.encoder_conv_expansion_factor = conv_expansion_factor
    args.encoder_conv_kernel_size = conv_kernel_size
    args.attention_dropout = attn_dropout
    args.ff_dropout = ff_dropout
    args.conv_dropout = conv_dropout
    args.max_pos_emb = max_pos_emb
    return args


def prepare_lstm_decoder_params(
    input_size,
    hidden_size,
    drop_out,
):
    
    args = argparse.Namespace()
    args.input_size = input_size
    args.hidden_size = hidden_size
    args.drop_out = drop_out
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
        #pdb.set_trace()
        bsz, max_seq_len, _ = src_tokens.size() #[B 126 90] = [B T F]
        x = src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim) #[B T 1 90]
        x = x.transpose(1, 2).contiguous() #[B 1 T 90] [B 1 126 90]
        # (B, C, T, feat)

        for layer_idx in range(len(self.conv_layers)):
            x = self.conv_layers[layer_idx](x)

        bsz, _, output_seq_len, _ = x.size() #[B 128 32 23] F異뺤씠 1/4, T異뺣룄 1/4 Channel=128

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
        # if not encoder_padding_mask.any():
        #     encoder_padding_mask = None
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

class simpleLSTM(nn.Module):
    def __init__(self,args):
        super().__init__()
        
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.drop_out = args.drop_out

        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size,num_layers=1,batch_first=False, dropout=self.drop_out)

    def forward(self,x):
        y ,_ = self.lstm(x)
        return y

class VGGLSTMDecoder(FairseqIncrementalDecoder):
    
    def __init__(
        self, 
        dictionary,
        embed_dim,
        lstm_config,
        encoder_output_dim,
    ):
        super().__init__(dictionary)
        
        # Our decoder will embed the inputs before feeding them to the LSTM.
        vocab_size = len(dictionary)
        
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(vocab_size,embed_dim,self.padding_idx)

        self.layers = nn.ModuleList()
        self.layers.append(simpleLSTM(
            prepare_lstm_decoder_params(*lstm_config[0])
        ))

        # Define the output projection.



        self.fc_out = Linear(lstm_config[-1][1]+encoder_output_dim, vocab_size)

    # During training Decoders are expected to take the entire target sequence
    # (shifted right by one position) and produce logits over the vocabulary.
    # The *prev_output_tokens* tensor begins with the end-of-sentence symbol,
    # ``dictionary.eos()``, followed by the target sequence.

    
    def extract_features(self, prev_output_tokens, audio_encoder_out=None, video_encoder_out=None, incremental_state=None):
        raise NotImplementedError

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        ##encoder_out.shape [45 9 144] [T B F]
        
        #pdb.set_trace()
        prev_output_tokens
        target_padding_mask = (
            (prev_output_tokens == self.padding_idx).to(prev_output_tokens.device)
            if incremental_state is None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        #[9 39 512]


        # B x T x C -> T x B x C
        x = self._transpose_if_training(x, incremental_state)
        ##[39,9,512]


        # for layer in self.conv_layers:
        #     if isinstance(layer, LinearizedConvolution):
        #         x = layer(x, incremental_state)
        #     else:
        #         x = layer(x)
        
        ###x.shape = [39, 9 ,256]
        # B x T x C -> T x B x C
        x = self._transpose_if_inference(x, incremental_state)

        # decoder layers
        for layer in self.layers:
            
            if isinstance(layer, simpleLSTM):
                #x= layer(x,encoder_out["encoder_out"] if encoder_out is not None else None), 
                x= layer(x)         
            else:
                x = layer(x)

        return x

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




class Jointnet(FairseqIncrementalDecoder):
    
    def __init__(
        self, 
        dictionary,
        embed_dim,
        lstm_config,
        encoder_output_dim,
    ):
        super().__init__(dictionary)
        
        # Our decoder will embed the inputs before feeding them to the LSTM.
        vocab_size = len(dictionary)
        
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(vocab_size,embed_dim,self.padding_idx)
        self.fc_out = Linear(lstm_config[-1][1]+encoder_output_dim, vocab_size)

    # During training Decoders are expected to take the entire target sequence
    # (shifted right by one position) and produce logits over the vocabulary.
    # The *prev_output_tokens* tensor begins with the end-of-sentence symbol,
    # ``dictionary.eos()``, followed by the target sequence.

    
    def extract_features(self, prev_output_tokens, audio_encoder_out=None, video_encoder_out=None, incremental_state=None):
        raise NotImplementedError

    def forward(self, audio_encoder_out,decoder_out, incremental_state=None):
        
        x = decoder_out
        x = x.transpose(0, 1)
        x = x.unsqueeze(1) # B 1 t C
        len_x = x.size(2)

        y = audio_encoder_out["encoder_out"] # T B F
        y = y.permute(1,0,2)
        y = y.unsqueeze(2) # B T 1 C
        len_y = y.size(1)

        x = x.repeat([1,len_y,1,1])
        y = y.repeat([1,1,len_x,1])
        joint = torch.cat((y,x),dim=-1)
        x = self.fc_out(joint)

        return x, audio_encoder_out["encoder_padding_mask"],None

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
    args.conformer_enc_config = getattr(
        args, "conformer_enc_config", DEFAULT_ENC_TRANSFORMER_CONFIG
    )
    args.enc_output_dim = getattr(args, "enc_output_dim", 512)
    args.in_channels = getattr(args, "in_channels", 1)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 512)
    args.lstm_dec_config = getattr(
        args, "lstm_dec_config", DEFAULT_ENC_TRANSFORMER_CONFIG
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

@register_model_architecture("libri_vggconformer_rnnt2", "libri_vggconformer_rnnt2_base")
def BiModalvggconformer_avsr_audio_only_base(args):
    args.audio_input_feat_per_channel = getattr(args, "audio_input_feat_per_channel", 90)
    #args.video_input_feat_per_channel = getattr(args, "video_input_feat_per_channel", 512)
    args.audio_vggblock_enc_config = getattr(
        args, "audio_vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )
    args.video_vggblock_enc_config = getattr(
        args, "video_vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )   
#    args.audio_vggblock_enc_config = 'None'
#    args.video_vggblock_enc_config = 'None'
    args.conformer_enc_config = getattr(
        args, "conformer_enc_config", "((144, 36, 4, 4, 2, 32, 0.1,0.1,0.1,144),) * 16"
    )
    args.audio_enc_output_dim = getattr(args, "audio_enc_output_dim", 144)
    args.video_enc_output_dim = getattr(args, "video_enc_output_dim", 512)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 512)
    #args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
#    args.conv_dec_config = 'None'
    args.lstm_dec_config = getattr(
        args, "lstm_dec_config", "((512,320,0.1),) * 1"
    )
 
