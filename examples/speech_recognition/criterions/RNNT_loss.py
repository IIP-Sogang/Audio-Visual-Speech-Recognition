#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from itertools import groupby

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from examples.speech_recognition.data.data_utils import encoder_padding_mask_to_lengths
from examples.speech_recognition.utils.wer_utils import Code, EditDistance, Token
import sys
sys.path.append('/home/nas/user/jungwook/fairseq_avsr/warp-transducer/')
from warprnnt_pytorch import RNNTLoss
import pdb
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def arr_to_toks(arr):
    toks = []
    for a in arr:
        toks.append(Token(str(a), 0.0, 0.0))
    return toks


def compute_rnnt_uer(logprobs, targets, input_lengths, target_lengths, blank_idx):
    """
    acts: Tensor of [batch x seqLength x (labelLength + 1) x outputDim] containing output from network
     (+1 means first blank label prediction)
    labels: 2 dimensional Tensor containing all the targets of the batch with zero padded
    act_lens: Tensor of size (batch) containing size of each output sequence from the network
    label_lens: Tensor of (batch) containing label length of each example
    """
    batch_errors = 0.0
    batch_total = 0.0
    for b in range(logprobs.shape[0]):
        predicted = logprobs[b][: input_lengths[b]].argmax(1).tolist()
        target = targets[b][: target_lengths[b]].tolist()
        # dedup predictions
        predicted = [p[0] for p in groupby(predicted)]
        # remove blanks
        nonblanks = []
        for p in predicted:
            if p != blank_idx:
                nonblanks.append(p)
        predicted = nonblanks

        # compute the alignment based on EditDistance
        alignment = EditDistance(False).align(
            arr_to_toks(predicted), arr_to_toks(target)
        )

        # compute the number of errors
        # note that alignment.codes can also be used for computing
        # deletion, insersion and substitution error breakdowns in future
        for a in alignment.codes:
            if a != Code.match:
                batch_errors += 1
        batch_total += len(target)

    return batch_errors, batch_total


@register_criterion("rnnt_loss")
class RNNTCriterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.blank_idx = task.target_dictionary.index("<rnnt_blank>")
        self.pad_idx = task.target_dictionary.pad()
        self.task = task

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--use-source-side-sample-size",
            action="store_true",
            default=False,
            help=(
                "when compute average loss, using number of source tokens "
                + "as denominator. "
                + "This argument will be no-op if sentence-avg is used."
            ),
        )

    def forward(self, model, sample, reduce=True, log_probs=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        #pdb.set_trace()
        net_output = model(**sample["net_input"])
        #pdb.set_trace()
        lprobs = model.get_normalized_probs(net_output, log_probs=log_probs)
        if not hasattr(lprobs, "batch_first"):
            logging.warning(
                "ERROR: we need to know whether "
                "batch first for the encoder output; "
                "you need to set batch_first attribute for the return value of "
                "model.get_normalized_probs. Now, we assume this is true, but "
                "in the future, we will raise exception instead. "
            )

        batch_first = getattr(lprobs, "batch_first", True)

        if not batch_first:
            max_seq_len = lprobs.size(0)
            bsz = lprobs.size(1)
        else:
            max_seq_len = lprobs.size(1)
            bsz = lprobs.size(0)
        device = net_output[0].device
        #pdb.set_trace()
        input_lengths = encoder_padding_mask_to_lengths(
            net_output[1], max_seq_len, bsz, device
        )
        target_lengths = sample["target_lengths"]
        targets = sample["target"]
        bsz = targets.size(0)
        targets = targets[targets!=2].reshape(bsz,-1)
        target_lengths = target_lengths-1
        # targets = torch.IntTensor(targets)
        # input_lengths = torch.IntTensor(input_lengths)
        # target_lengths = torch.IntTensor(target_lengths)
        targets = targets.int()
        input_lengths = input_lengths.int()
        target_lengths = target_lengths.int()

        pad_mask = sample["target"] != self.pad_idx
        #targets_flat = targets.masked_select(pad_mask)

        criterion = RNNTLoss(blank=self.blank_idx,reduction='sum')
        loss = criterion(
            lprobs,
            targets,
            input_lengths,
            target_lengths,
        )
        
        #pdb.set_trace()
        #lprobs = lprobs.transpose(0, 1)  # T N D -> N T D
        errors, total = compute_rnnt_uer(
            lprobs, targets, input_lengths, target_lengths, self.blank_idx
        )

        if self.args.sentence_avg:
            sample_size = sample["target"].size(0)
        else:
            if self.args.use_source_side_sample_size:
                sample_size = torch.sum(input_lengths).item()
            else:
                sample_size = sample["ntokens"]

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "errors": errors,
            "total": total,
            "nframes": torch.sum(sample["net_input"]["src_lengths"]).item(),
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        errors = sum(log.get("errors", 0) for log in logging_outputs)
        total = sum(log.get("total", 0) for log in logging_outputs)
        nframes = sum(log.get("nframes", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / sample_size / math.log(2),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "nframes": nframes,
            "sample_size": sample_size,
            "acc": 100.0 - min(errors * 100.0 / total, 100.0),
        }
        if sample_size != ntokens:
            agg_output["nll_loss"] = loss_sum / ntokens / math.log(2)
        return agg_output
