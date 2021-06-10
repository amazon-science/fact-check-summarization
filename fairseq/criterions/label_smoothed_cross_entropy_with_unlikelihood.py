# Copyright (c) Facebook, Inc. and its affiliates.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import torch

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def hellinger_loss(probs, target, ignore_index=None, reduce=True):
    if target.dim() == probs.dim() - 1:
        target = target.unsqueeze(-1)
    select_probs = probs.gather(dim=-1, index=target)
    hellinger_l = (1 - torch.sqrt(select_probs))**2
    # hellinger_l = 1.0 - torch.sqrt(select_probs)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            hellinger_l.masked_fill_(pad_mask, 0.)
    else:
        hellinger_l = hellinger_l.squeeze(-1)
    if reduce:
        hellinger_l = hellinger_l.sum()
    return hellinger_l, hellinger_l


@register_criterion('label_smoothed_cross_entropy_with_unlikelihood')
class LabelSmoothedCrossEntropyCriterionWithUnlikelihood(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, use_hellinger_loss):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.use_hellinger_loss = use_hellinger_loss
        print("Initializing label_smoothed_cross_entropy_with_unlikelihood: use_hellinger_loss={}".format(use_hellinger_loss))

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--use-hellinger-loss', type=int, default=0, metavar='N',
                            help='0(default): do not use hellinger loss for unlikelihood training'
                                 '1: use hellinger loss for negative examples, use the nll loss for positive examples'
                                 '2: use hellinger loss for both negative and positive examples')


        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        the samples consists of likelihood as well as unlikelihood samples

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # print("=====source-target=======")
        # print("net_input {}: {}".format(sample['source-target']['id'], sample['source-target']['net_input']['src_tokens']))
        # print("target: {}".format(sample['source-target']['target']))
        # if sample['source-target'].get('sample_weights', None) is not None:
        #     print("sample weights={}".format(sample['source-target']['sample_weights']))
        #
        # print("=====source-untarget=======")
        # print("net_input {}: {}".format(sample['source-untarget']['id'], sample['source-untarget']['net_input']['src_tokens']))
        # print("target: {}".format(sample['source-untarget']['target']))
        # if sample['source-untarget'].get('sample_weights', None) is not None:
        #     print("sample weights={}".format(sample['source-untarget']['sample_weights']))

        net_output_likelihood = model(**sample['source-target']['net_input'])
        loss_likelihood, nll_loss_likelihood = self.compute_loss(model, net_output_likelihood, sample['source-target'],
                                                                 reduce=reduce,
                                                                 hellinger=self.use_hellinger_loss)
        sample_size_likelihood = sample['source-target']['target'].size(0) if self.sentence_avg else \
            sample['source-target']['ntokens']

        net_output_unlikelihood = model(**sample['source-untarget']['net_input'])
        loss_unlikelihood, nll_loss_unlikelihood = self.compute_loss_unlikelihood(model, net_output_unlikelihood,
                                                                                  sample['source-untarget'],
                                                                                  reduce=reduce,
                                                                                  hellinger=self.use_hellinger_loss)
        sample_size_unlikelihood = sample['source-untarget']['target'].size(0) if self.sentence_avg else \
            sample['source-untarget']['ntokens']

        logging_output = {
            'loss': loss_likelihood.data + loss_unlikelihood.data,
            'nll_loss': nll_loss_likelihood.data + nll_loss_unlikelihood.data,
            'ntokens': sample['source-target']['ntokens'] + sample['source-untarget']['ntokens'],
            'nsentences': sample['source-target']['target'].size(0) + sample['source-untarget']['target'].size(0),
            'sample_size': sample_size_likelihood + sample_size_unlikelihood,
        }
        return loss_likelihood + loss_unlikelihood, sample_size_likelihood + sample_size_unlikelihood, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, hellinger=0):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        if sample.get('sample_weights', None) is not None:
            lprobs = lprobs * sample['sample_weights'][:, None, None]

        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        if hellinger == 2:
            loss, nll_loss = hellinger_loss(lprobs.exp(), target, ignore_index=self.padding_idx, reduce=reduce)
        else:
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
        return loss, nll_loss

    def compute_loss_unlikelihood(self, model, net_output, sample, reduce=True, hellinger=0):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        if sample.get('sample_weights', None) is not None:
            lprobs = lprobs * sample['sample_weights'][:, None, None]
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        if hellinger > 0:
            loss, nll_loss = hellinger_loss(1.0 - lprobs.exp(), target, ignore_index=self.padding_idx, reduce=reduce)
        else:
            one_minus_probs = torch.clamp((1.0 - lprobs.exp()), min=1e-20)
            lprobs = torch.log(one_minus_probs)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
