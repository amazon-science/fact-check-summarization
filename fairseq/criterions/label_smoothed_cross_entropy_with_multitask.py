# Copyright (c) Facebook, Inc. and its affiliates.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
import torch.nn.functional as F
import torch

def safe_divide(numerator, denominator):
    if denominator == 0:
        return denominator + 1.0
    return numerator * 1.0 / denominator


@register_criterion('label_smoothed_cross_entropy_with_multitask')
class LabelSmoothedCrossEntropyCriterionWithMultitask(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, multitask_lambda, classification_head_name, validation_only):
        super().__init__(task, sentence_avg, label_smoothing)
        self.multitask_lambda = multitask_lambda
        self.classification_head_name = classification_head_name
        self.validation_only = validation_only

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--multitask-lambda', default=0.05, type=float, metavar='D',
                            help='weight for the classification loss in multitask training')
        parser.add_argument('--classification-head-name',
                            default='multitask',
                            help='name of the classification head to use')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, 'classification_heads')
            and self.classification_head_name in model.classification_heads
        ), 'model must provide classification head for --criterion=label_smoothed_cross_entropy_with_multitask'
        # print(sample['net_input'].keys())
        net_output = model(classification_head_name=self.classification_head_name, **sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'ntokens_src': sample['ntokens_src'],
        }
        # print("size of loss=".format(loss.shape))
        # print("loss={}".format(loss.data))
        classification_loss = None

        # Compute classification loss only for training set and non dummy batches.
        if 'cls_labels' in sample and sample['cls_labels'] is not None:
            classification_loss, ncorrect, cls_metrics = self.compute_classification_loss(sample, net_output)

        if 'indices_labels' in sample:
            if sample['indices_labels'] is not None:
                classification_loss, ncorrect, cls_metrics = self.compute_classification_loss_indices(sample, net_output)
                logging_output['n_indices'] = sample['indices_labels'].numel()
            else:
                # indices_labels are empty because some sample doesn't have indices_labels, skip but do logging.
                z_e_r_o = (sample['net_input']['src_tokens'] == sample['net_input']['src_tokens']).sum() * 0
                logging_output['classification_loss'] = z_e_r_o * 1.0
                logging_output['n_indices'] = z_e_r_o.numel() * 0
                logging_output['ncorrect'] = z_e_r_o
                cls_metrics = {
                    't0_p0': z_e_r_o.clone(),
                    't0_p1': z_e_r_o.clone(),
                    't1_p0': z_e_r_o.clone(),
                    't1_p1': z_e_r_o.clone(),
                    'precision_0': z_e_r_o * 1.0,
                    'recall_0': z_e_r_o * 1.0,
                    'precision_1': z_e_r_o * 1.0,
                    'recall_1': z_e_r_o * 1.0,
                }
                for k in cls_metrics:
                    logging_output[k] = cls_metrics[k]

        if classification_loss is not None:
            logging_output['classification_loss'] = classification_loss.data
            loss += self.multitask_lambda * classification_loss

            logging_output['ncorrect'] = ncorrect
            for k in cls_metrics:
                logging_output[k] = cls_metrics[k]

        return loss, sample_size, logging_output

    def compute_classification_loss_indices(self, sample, net_output):
        logits = net_output[1]['cls_out'].view(-1, net_output[1]['cls_out'].size(-1))
        targets = sample['indices_labels'].view(-1)
        loss = F.nll_loss(
            F.log_softmax(logits, dim=-1, dtype=torch.float32),
            targets,
            reduction='sum',
            ignore_index=-1
        )
        preds = logits.argmax(dim=1)
        ncorrect = (preds == targets).sum()
        t0_p0 = (preds[targets == 0] == 0).sum()
        t0_p1 = (preds[targets == 0] == 1).sum()
        t1_p0 = (preds[targets == 1] == 0).sum()
        t1_p1 = (preds[targets == 1] == 1).sum()
        precision_0 = safe_divide(t0_p0, t0_p0 + t1_p0)
        recall_0 = safe_divide(t0_p0, t0_p0 + t0_p1)
        precision_1 = safe_divide(t1_p1, t0_p1 + t1_p1)
        recall_1 = safe_divide(t1_p1, t1_p0 + t1_p1)
        cls_metrics = {
            't0_p0': t0_p0,
            't0_p1': t0_p1,
            't1_p0': t1_p0,
            't1_p1': t1_p1,
            'precision_0': precision_0,
            'recall_0': recall_0,
            'precision_1': precision_1,
            'recall_1': recall_1,
        }
        return loss, ncorrect, cls_metrics


    def compute_classification_loss(self, sample, net_output):
        logits = net_output[1]['cls_out'].view(-1, net_output[1]['cls_out'].size(-1))
        targets = sample['cls_labels'].view(-1)
        loss = F.nll_loss(
            F.log_softmax(logits, dim=-1, dtype=torch.float32),
            targets,
            reduction='sum',
            ignore_index=-1
        )
        preds = logits.argmax(dim=1)
        ncorrect = (preds[targets != -1] == targets[targets != -1]).sum()
        t0_p0 = (preds[targets == 0] == 0).sum()
        t0_p1 = (preds[targets == 0] == 1).sum()
        t0_p2 = (preds[targets == 0] == 2).sum()

        t1_p0 = (preds[targets == 1] == 0).sum()
        t1_p1 = (preds[targets == 1] == 1).sum()
        t1_p2 = (preds[targets == 1] == 2).sum()

        t2_p0 = (preds[targets == 2] == 0).sum()
        t2_p1 = (preds[targets == 2] == 1).sum()
        t2_p2 = (preds[targets == 2] == 2).sum()

        precision_0 = safe_divide(t0_p0, t0_p0 + t1_p0 + t2_p0)
        recall_0 = safe_divide(t0_p0, t0_p0 + t0_p1 + t0_p2)
        precision_1 = safe_divide(t1_p1, t0_p1 + t1_p1 + t2_p1)
        recall_1 = safe_divide(t1_p1, t1_p0 + t1_p1 + t1_p2)
        precision_2 = safe_divide(t2_p2, t0_p2 + t1_p2 + t2_p2)
        recall_2 = safe_divide(t2_p2, t2_p0 + t2_p1 + t2_p2)

        cls_metrics = {
            't0_p0': t0_p0,
            't0_p1': t0_p1,
            't0_p2': t0_p2,
            't1_p0': t1_p0,
            't1_p1': t1_p1,
            't1_p2': t1_p2,
            't2_p0': t2_p0,
            't2_p1': t2_p1,
            't2_p2': t2_p2,
            'precision_0': precision_0,
            'recall_0': recall_0,
            'precision_1': precision_1,
            'recall_1': recall_1,
            'precision_2': precision_2,
            'recall_2': recall_2
        }
        return loss, ncorrect, cls_metrics

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        classification_loss_sum = sum(log.get('classification_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        ntokens_src = sum(log.get('ntokens_src', 0) for log in logging_outputs)
        n_indices = sum(log.get('n_indices', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

        metrics.log_scalar('classification_loss', classification_loss_sum / ntokens_src / math.log(2), ntokens_src, round=3)
        ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)

        if n_indices:
            metrics.log_scalar('accuracy', ncorrect * 1.0 / n_indices, n_indices, round=3)
        else:
            metrics.log_scalar('accuracy', ncorrect * 1.0 / ntokens_src, ntokens_src, round=3)

        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)

        for k in [
            't0_p0',
            't0_p1',
            't0_p2',
            't1_p0',
            't1_p1',
            't1_p2',
            't2_p0',
            't2_p1',
            't2_p2',
            'precision_0',
            'recall_0',
            'precision_1',
            'recall_1',
            'precision_2',
            'recall_2'
        ]:
            tmp = sum(log.get(k, 0) for log in logging_outputs)
            if k.startswith('t'):
                if n_indices:
                    metrics.log_scalar(k, tmp * 1.0 / n_indices, n_indices, round=3)
                else:
                    metrics.log_scalar(k, tmp * 1.0 / ntokens_src, ntokens_src, round=3)
            else:
                metrics.log_scalar(k, tmp * 1.0 / nsentences, nsentences, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
