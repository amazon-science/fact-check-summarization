# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

from tqdm import tqdm
import os
import glob
import json
from evaluate_hypo import count_lines_in_text_file
import argparse

import boto3
from fairseq.models.bart import BARTModel
import torch
from pathos.multiprocessing import ProcessPool
import subprocess
from spacy.lang.en.stop_words import STOP_WORDS
from data_prepro_clean import _format_source_answers_bpe
import random
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    TruncateLastElementDataset,
)
from fairseq import utils
from fairseq.sequence_scorer import SequenceScorer
from types import SimpleNamespace


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _run_q_gen_process_local(job_idx, *, input_source_file, input_ans_file, out_text_file,
                             offset, end, checkpoint_dir, ckp_file, bin_dir, args):
    bart = BARTModel.from_pretrained(
        checkpoint_dir,
        checkpoint_file=ckp_file,
        data_name_or_path=bin_dir
    )
    torch.cuda.set_device(torch.device("cuda:{}".format(job_idx)))
    bart.cuda()
    bart.eval()
    bart.half()

    special_token_id = 50009
    from fairseq.data.encoders.gpt2_bpe import get_encoder
    bpe = get_encoder(args.encoder_json, args.vocab_bpe)

    def _filter_ans_list(ans_list, black_set, num_ans_target):
        if len(ans_list) == 0:
            return []
        good_ans = []
        bad_ans = []
        for ans in ans_list:
            if ans.lower() in black_set:
                bad_ans.append(ans)
            else:
                good_ans.append(ans)

        if len(good_ans) >= num_ans_target:
            result = good_ans[:num_ans_target]
        elif len(ans_list) >= num_ans_target:
            result = good_ans + bad_ans
            result = result[:num_ans_target]
        else:
            result = ans_list + random.choices(ans_list, k=num_ans_target-len(ans_list))
        return result

    count = 0
    # bsz = 32
    bsz = args.bsz
    print("Local worker is processing {}-{}".format(offset, end))
    with torch.no_grad():
        with open(input_source_file, 'r') as source_f, \
                open(input_ans_file, 'r') as ans_f, \
                open(out_text_file, 'w') as out_text_f:
            for _ in range(offset):
                source_f.readline()
                ans_f.readline()
            source_line = source_f.readline()
            ans_line = ans_f.readline()
            while source_line:
                if offset + count >= end:
                    break
                source_item = json.loads(source_line.strip())
                ans_item = json.loads(ans_line.strip())

                assert len(source_item['summaries']) == len(ans_item)

                input_buffer = []
                hypo_id = 0
                # ans_dict = {}
                # filtered_ans_list = []
                for source_text, ans_list_hypo in zip(source_item['summaries'], ans_item):
                    # return 10 answers: try to avoid stop words; upsampling if not enough
                    # or return empty list if no answer available
                    if ans_list_hypo == []:
                        print("Answer span is empty!")
                        print(ans_item)
                    filtered_ans_list_hypo = _filter_ans_list(ans_list_hypo, STOP_WORDS, 10)
                    # filtered_ans_list.append(filtered_ans_list_hypo)

                    for answer in filtered_ans_list_hypo:
                        # if answer not in ans_dict:
                            # ans_dict[answer] = None
                        _, source_answer_bpe = _format_source_answers_bpe(bpe, source_text, answer, special_token_id)
                        input_buffer.append((hypo_id, answer, ' '.join(map(str, source_answer_bpe))))
                    hypo_id += 1

                all_hypos = []
                all_ids = []
                slines = []
                pa_ids = []
                # print("len of input_buffer {} = {}".format(job_idx, len(input_buffer)))
                for i in range(len(input_buffer)):
                    if i % bsz == 0 and i != 0:
                        hypotheses_batch, score_batch, unnormalized_score_batch = bart.sample(slines,
                                                                    beam=args.beam,
                                                                    lenpen=1.0,
                                                                    max_len_b=args.max_len,
                                                                    min_len=args.min_len,
                                                                    sampling=args.sampling,
                                                                    sampling_topk=args.sampling_topk,
                                                                    sampling_topp=args.sampling_topp,
                                                                    return_all=True,
                                                                    input_is_bpe=True
                                                                    )
                        assert len(hypotheses_batch) == len(score_batch) == len(unnormalized_score_batch), \
                            "lens not equal: {} and {} and {}".format(
                                len(hypotheses_batch), len(score_batch), len(unnormalized_score_batch)
                            )
                        assert len(hypotheses_batch) == len(slines), "slines={}, generated_score length={}".format(
                            slines, len(hypotheses_batch)
                        )

                        for t, s in zip(hypotheses_batch, score_batch):
                            all_hypos.append((t, s))

                        for id in pa_ids:
                            all_ids.append(id)

                        slines = []
                        pa_ids = []
                    slines.append(input_buffer[i][2])
                    pa_ids.append((input_buffer[i][0], input_buffer[i][1]))
                if slines != []:
                    hypotheses_batch, score_batch, unnormalized_score_batch = bart.sample(slines,
                                                                beam=args.beam,
                                                                lenpen=1.0,
                                                                max_len_b=args.max_len,
                                                                min_len=args.min_len,
                                                                sampling=args.sampling,
                                                                sampling_topk=args.sampling_topk,
                                                                sampling_topp=args.sampling_topp,
                                                                return_all=True,
                                                                input_is_bpe=True
                                                                )
                    assert len(hypotheses_batch) == len(score_batch) == len(unnormalized_score_batch), \
                        "lens not equal: {} and {} and {}".format(
                            len(hypotheses_batch), len(score_batch), len(unnormalized_score_batch)
                        )
                    assert len(hypotheses_batch) == len(slines), "slines={}, generated_score length={}".format(
                        slines, len(hypotheses_batch)
                    )

                    for t, s in zip(hypotheses_batch, score_batch):
                        all_hypos.append((t, s))

                    for id in pa_ids:
                        all_ids.append(id)

                # for id, hypo in zip(all_ids, all_hypos):
                #     ans_dict[id[1]] = {'questions': hypo[0], 'scores': hypo[1]}
                #
                # qa_item = {'question_dict': ans_dict, 'answers': filtered_ans_list}
                qa_item = []
                if all_ids != [] and all_hypos != []:
                    hypo_id = all_ids[0][0]
                    qa_list_hypo = []
                    for id, hypo in zip(all_ids, all_hypos):
                        if id[0] == hypo_id:
                            qa_list_hypo.append({'questions': hypo[0], 'q_scores': hypo[1], 'answer': id[1]})
                        else:
                            qa_item.append({'context': source_item['summaries'][hypo_id],
                                            'qa_list': qa_list_hypo})
                            qa_list_hypo = []
                            qa_list_hypo.append({'questions': hypo[0], 'q_scores': hypo[1], 'answer': id[1]})
                            hypo_id = id[0]
                    qa_item.append({'context': source_item['summaries'][hypo_id],
                                    'qa_list': qa_list_hypo})

                json.dump(qa_item, out_text_f)
                out_text_f.write('\n')

                source_line = source_f.readline()
                ans_line = ans_f.readline()
                count += 1
                if count % 100 == 0:
                    print("Generated {} lines from worker {}".format(count, job_idx))

        assert offset + count == end, "!worker ended at {}, should have been {}".format(
            offset + count,
            end
        )
        del bart
        torch.cuda.empty_cache()


def _sample_wrapper(model, sentences, beam=1, verbose=False, return_all=False,
               input_is_bpe=False, return_token_scores=False, **kwargs):

    if return_token_scores:
        hypotheses_batch, score_batch, unnormalized_score_batch, pos_scores, tokens = model.sample(
            sentences=sentences,
            beam=beam,
            verbose=verbose,
            return_all=return_all,
            input_is_bpe=input_is_bpe,
            return_token_scores=return_token_scores,
            **kwargs
        )
        return hypotheses_batch, score_batch, unnormalized_score_batch, pos_scores, tokens
    else:
        hypotheses_batch, score_batch, unnormalized_score_batch = model.sample(
            sentences=sentences,
            beam=beam,
            verbose=verbose,
            return_all=return_all,
            input_is_bpe=input_is_bpe,
            return_token_scores=return_token_scores,
            **kwargs
        )
        return hypotheses_batch, score_batch, unnormalized_score_batch, None, None


def _run_qa_gen_process_local_batch_lines(job_idx, *, input_source_file, out_text_file,
                             offset, end, checkpoint_dir, ckp_file, bin_dir, args):
    bart = BARTModel.from_pretrained(
        checkpoint_dir,
        checkpoint_file=ckp_file,
        data_name_or_path=bin_dir
    )
    torch.cuda.set_device(torch.device("cuda:{}".format(job_idx)))
    bart.cuda()
    bart.eval()
    bart.half()

    count = 1
    # bsz = 32
    bsz = args.bsz
    print("Local worker is processing {}-{}".format(offset, end))
    with torch.no_grad():
        with open(input_source_file, 'r') as source_f, \
                open(out_text_file, 'w') as out_text_f:
            for _ in range(offset):
                source_f.readline()
            source_line = source_f.readline()
            source_item = json.loads(source_line.strip())
            assert len(source_item['summaries']) == 1
            slines = [source_item['summaries'][0].strip()]
            while source_line:
                if offset + count >= end:
                    break
                if count % bsz == 0:
                    hypotheses_batch, score_batch, unnormalized_score_batch, pos_score_batch, tokens_batch = \
                        _sample_wrapper(
                            bart,
                            sentences=slines,
                            beam=args.beam,
                            lenpen=1.0,
                            max_len_b=args.max_len,
                            min_len=args.min_len,
                            sampling=args.sampling,
                            sampling_topk=args.sampling_topk,
                            sampling_topp=args.sampling_topp,
                            return_all=args.return_all,
                            input_is_bpe=False,
                            return_token_scores=args.return_token_scores,
                            diverse_beam_groups=args.diverse_beam_groups,
                            diverse_beam_strength=args.diverse_beam_strength,
                        )
                    assert len(hypotheses_batch) == len(score_batch) == len(unnormalized_score_batch), \
                        "lens not equal: {} and {} and {}".format(
                        len(hypotheses_batch), len(score_batch), len(unnormalized_score_batch)
                    )
                    assert len(hypotheses_batch) == len(slines), "slines={}, generated_score length={}".format(
                        slines, len(hypotheses_batch)
                    )
                    if args.return_token_scores:
                        for t, s, unnormalized_s, pos_s, toks, sline in zip(hypotheses_batch, score_batch,
                                                                           unnormalized_score_batch,
                                                                           pos_score_batch, tokens_batch, slines):
                            qa_item = [{
                                'context': sline,
                                'qa': t if type(t) is list else [t, ],
                                'norm_scores': s if type(s) is list else [s, ],
                                'unnorm_scores': unnormalized_s if type(unnormalized_s) is list else [unnormalized_s, ],
                                'pos_scores': [tmp.tolist() for tmp in pos_s] if args.return_all and args.beam > 1 \
                                    else [pos_s.tolist(), ],
                                'toks': [tmp.tolist() for tmp in toks] if args.return_all and args.beam > 1 else \
                                    [toks.tolist(), ]
                            }, ]
                            json.dump(qa_item, out_text_f)
                            out_text_f.write('\n')
                    else:
                        for t, s, unnormalized_s, sline in zip(hypotheses_batch, score_batch, unnormalized_score_batch,
                                                               slines):
                            qa_item = [{
                                'context': sline,
                                'qa': t if type(t) is list else [t, ],
                                'norm_scores': s if type(s) is list else [s, ],
                                'unnorm_scores':  unnormalized_s if type(unnormalized_s) is list else [unnormalized_s,]
                            },]
                            json.dump(qa_item, out_text_f)
                            out_text_f.write('\n')
                    out_text_f.flush()
                    slines = []
                source_line = source_f.readline()
                source_item = json.loads(source_line.strip())
                slines.append(source_item['summaries'][0].strip())
                count += 1
                # if count % 100 == 0:
                #     print("Generated {} lines from worker {}".format(count, job_idx))

            if slines != []:
                hypotheses_batch, score_batch, unnormalized_score_batch, pos_score_batch, tokens_batch = \
                    _sample_wrapper(
                        bart,
                        sentences=slines,
                        beam=args.beam,
                        lenpen=1.0,
                        max_len_b=args.max_len,
                        min_len=args.min_len,
                        sampling=args.sampling,
                        sampling_topk=args.sampling_topk,
                        sampling_topp=args.sampling_topp,
                        return_all=args.return_all,
                        input_is_bpe=False,
                        return_token_scores=args.return_token_scores,
                        diverse_beam_groups=args.diverse_beam_groups,
                        diverse_beam_strength=args.diverse_beam_strength,
                    )
                assert len(hypotheses_batch) == len(score_batch) == len(unnormalized_score_batch), \
                    "lens not equal: {} and {} and {}".format(
                        len(hypotheses_batch), len(score_batch), len(unnormalized_score_batch)
                    )
                assert len(hypotheses_batch) == len(slines), "slines={}, generated_score length={}".format(
                    slines, len(hypotheses_batch)
                )

                if args.return_token_scores:
                    for t, s, unnormalized_s, pos_s, toks, sline in zip(hypotheses_batch, score_batch,
                                                                        unnormalized_score_batch,
                                                                        pos_score_batch, tokens_batch, slines):
                        qa_item = [{
                            'context': sline,
                            'qa': t if type(t) is list else [t, ],
                            'norm_scores': s if type(s) is list else [s, ],
                            'unnorm_scores': unnormalized_s if type(unnormalized_s) is list else [unnormalized_s, ],
                            'pos_scores': [tmp.tolist() for tmp in pos_s] if args.return_all and args.beam > 1 else \
                                [pos_s.tolist(), ],
                            'toks': [tmp.tolist() for tmp in toks] if args.return_all and args.beam > 1 else \
                                [toks.tolist(), ]
                        }, ]
                        json.dump(qa_item, out_text_f)
                        out_text_f.write('\n')
                else:
                    for t, s, unnormalized_s, sline in zip(hypotheses_batch, score_batch, unnormalized_score_batch,
                                                           slines):
                        qa_item = [{
                            'context': sline,
                            'qa': t if type(t) is list else [t, ],
                            'norm_scores': s if type(s) is list else [s, ],
                            'unnorm_scores': unnormalized_s if type(unnormalized_s) is list else [unnormalized_s, ]
                        }, ]
                        json.dump(qa_item, out_text_f)
                        out_text_f.write('\n')
                out_text_f.flush()

        assert offset + count == end, "!worker ended at {}, should have been {}".format(
            offset + count,
            end
        )
        del bart
        torch.cuda.empty_cache()


def _run_qa_gen_process_local(job_idx, *, input_source_file, out_text_file,
                             offset, end, checkpoint_dir, ckp_file, bin_dir, args):
    bart = BARTModel.from_pretrained(
        checkpoint_dir,
        checkpoint_file=ckp_file,
        data_name_or_path=bin_dir
    )
    torch.cuda.set_device(torch.device("cuda:{}".format(job_idx)))
    bart.cuda()
    bart.eval()
    bart.half()

    special_token_id = 50009
    from fairseq.data.encoders.gpt2_bpe import get_encoder
    bpe = get_encoder(args.encoder_json, args.vocab_bpe)

    count = 0
    # bsz = 32
    bsz = args.bsz
    print("Local worker is processing {}-{}".format(offset, end))
    with torch.no_grad():
        with open(input_source_file, 'r') as source_f, \
                open(out_text_file, 'w') as out_text_f:
            for _ in range(offset):
                source_f.readline()
            source_line = source_f.readline()
            while source_line:
                if offset + count >= end:
                    break
                source_item = json.loads(source_line.strip())

                input_buffer = []
                hypo_id = 0
                # ans_dict = {}
                # filtered_ans_list = []
                for source_text in source_item['summaries']:
                    input_buffer.append((hypo_id, source_text))
                    hypo_id += 1

                all_hypos = []
                all_ids = []
                slines = []
                pa_ids = []
                # print("len of input_buffer {} = {}".format(job_idx, len(input_buffer)))
                for i in range(len(input_buffer)):
                    if i % bsz == 0 and i != 0:
                        hypotheses_batch, score_batch, unnormalized_score_batch, pos_score_batch, tokens_batch = \
                            _sample_wrapper(
                                bart,
                                sentences=slines,
                                beam=args.beam,
                                lenpen=1.0,
                                max_len_b=args.max_len,
                                min_len=args.min_len,
                                sampling=args.sampling,
                                sampling_topk=args.sampling_topk,
                                sampling_topp=args.sampling_topp,
                                return_all=args.return_all,
                                input_is_bpe=False,
                                return_token_scores=args.return_token_scores,
                                diverse_beam_groups=args.diverse_beam_groups,
                                diverse_beam_strength=args.diverse_beam_strength,
                            )
                        assert len(hypotheses_batch) == len(score_batch) == len(unnormalized_score_batch), \
                            "lens not equal: {} and {} and {}".format(
                            len(hypotheses_batch), len(score_batch), len(unnormalized_score_batch)
                        )
                        assert len(hypotheses_batch) == len(slines), "slines={}, generated_score length={}".format(
                            slines, len(hypotheses_batch)
                        )
                        if args.return_token_scores:
                            for t, s, unnormalized_s, pos_s, toks in zip(hypotheses_batch, score_batch,
                                                                         unnormalized_score_batch,
                                                                         pos_score_batch, tokens_batch):
                                all_hypos.append((t, s, unnormalized_s, pos_s, toks))
                        else:
                            for t, s, unnormalized_s in zip(hypotheses_batch, score_batch, unnormalized_score_batch):
                                all_hypos.append((t, s, unnormalized_s))

                        for id in pa_ids:
                            all_ids.append(id)

                        slines = []
                        pa_ids = []
                    slines.append(input_buffer[i][1])
                    pa_ids.append(input_buffer[i][0])
                if slines != []:
                    hypotheses_batch, score_batch, unnormalized_score_batch, pos_score_batch, tokens_batch = \
                        _sample_wrapper(
                            bart,
                            sentences=slines,
                            beam=args.beam,
                            lenpen=1.0,
                            max_len_b=args.max_len,
                            min_len=args.min_len,
                            sampling=args.sampling,
                            sampling_topk=args.sampling_topk,
                            sampling_topp=args.sampling_topp,
                            return_all=args.return_all,
                            input_is_bpe=False,
                            return_token_scores=args.return_token_scores,
                            diverse_beam_groups=args.diverse_beam_groups,
                            diverse_beam_strength=args.diverse_beam_strength,
                        )
                    assert len(hypotheses_batch) == len(score_batch) == len(unnormalized_score_batch), \
                        "lens not equal: {} and {} and {}".format(
                            len(hypotheses_batch), len(score_batch), len(unnormalized_score_batch)
                        )
                    assert len(hypotheses_batch) == len(slines), "slines={}, generated_score length={}".format(
                        slines, len(hypotheses_batch)
                    )

                    if args.return_token_scores:
                        for t, s, unnormalized_s, pos_s, toks in zip(hypotheses_batch, score_batch,
                                                                     unnormalized_score_batch,
                                                                     pos_score_batch, tokens_batch):
                            all_hypos.append((t, s, unnormalized_s, pos_s, toks))
                    else:
                        for t, s, unnormalized_s in zip(hypotheses_batch, score_batch, unnormalized_score_batch):
                            all_hypos.append((t, s, unnormalized_s))

                    for id in pa_ids:
                        all_ids.append(id)

                # for id, hypo in zip(all_ids, all_hypos):
                #     ans_dict[id[1]] = {'questions': hypo[0], 'scores': hypo[1]}
                #
                # qa_item = {'question_dict': ans_dict, 'answers': filtered_ans_list}
                qa_item = []
                if all_ids != [] and all_hypos != []:
                    if args.return_token_scores:
                        for id, hypo in zip(all_ids, all_hypos):
                            qa_item.append({'context': source_item['summaries'][id],
                                            'qa': hypo[0], 'norm_scores': hypo[1], 'unnorm_scores': hypo[2],
                                            'pos_scores': [tmp.tolist() for tmp in hypo[3]],
                                            'toks': [tmp.tolist() for tmp in hypo[4]]})
                    else:
                        for id, hypo in zip(all_ids, all_hypos):
                            qa_item.append({'context': source_item['summaries'][id],
                                            'qa': hypo[0], 'norm_scores': hypo[1], 'unnorm_scores': hypo[2]})

                json.dump(qa_item, out_text_f)
                out_text_f.write('\n')

                source_line = source_f.readline()
                count += 1
                # if count % 100 == 0:
                #     print("Generated {} lines from worker {}".format(count, job_idx))

        assert offset + count == end, "!worker ended at {}, should have been {}".format(
            offset + count,
            end
        )
        del bart
        torch.cuda.empty_cache()

def _run_qa_eval_process_local(job_idx, *, input_source_file, input_target_file, input_qas_file, out_text_file,
                             offset, end, checkpoint_dir, ckp_file, bin_dir, args):
    bart = BARTModel.from_pretrained(
        checkpoint_dir,
        checkpoint_file=ckp_file,
        data_name_or_path=bin_dir
    )
    torch.cuda.set_device(torch.device("cuda:{}".format(job_idx)))
    bart.cuda()
    bart.eval()
    bart.half()

    def batch_for_scorer(source_tokens_list, num_source_token_list, target_tokens_list, num_target_token_list, bsz):
        length = len(source_tokens_list)
        s = 0
        while s < length:
            e = s + bsz
            yield source_tokens_list[s:e], num_source_token_list[s:e], \
                  target_tokens_list[s:e], num_target_token_list[s:e]
            s = e

    special_token = 50259

    count = 0
    # bsz = 32
    bsz = args.bsz
    print("Local worker is processing {}-{}".format(offset, end))
    with torch.no_grad():
        with open(input_source_file, 'r') as source_f, \
                open(input_qas_file, 'r') as qas_f, \
                open(input_target_file, 'r') as target_f, \
                open(out_text_file, 'w') as out_text_f:
            for _ in range(offset):
                source_f.readline()
                target_f.readline()
                qas_f.readline()
            source_line = source_f.readline()
            target_line = target_f.readline()
            qas_line = qas_f.readline()
            while source_line:
                if offset + count >= end:
                    break

                max_source_tokens = 1024
                if args.prepend_target:
                    src_tokens = bart.encode(target_line.strip() + ' ' + source_line.strip(), no_bos=True,
                                          input_is_bpe=False)
                else:
                    src_tokens = bart.encode(source_line.strip(), no_bos=True, input_is_bpe=False)
                if len(src_tokens) > max_source_tokens:
                    src_tokens[max_source_tokens - 1] = src_tokens[-1]
                src_tokens = src_tokens if len(src_tokens) <= max_source_tokens else src_tokens[:max_source_tokens]

                qas_item = json.loads(qas_line.strip())

                qa_tensors = []
                for hypo_qas in qas_item:
                    for qa in hypo_qas['qas']:
                        if 'toks' in qa:
                            qa_tensors.append(torch.LongTensor(qa['toks']))
                        else:
                            q_tensor = bart.encode(qa['q'], no_bos=True, input_is_bpe=False)
                            q_tensor[-1] = special_token
                            a_tensor = bart.encode(qa['a'], no_bos=True, input_is_bpe=False)
                            qa_tensors.append(torch.cat((q_tensor, a_tensor)))

                num_src_tokens = src_tokens.numel()
                src_tokens_list = [src_tokens for _ in range(len(qa_tensors))]
                num_src_token_list = [num_src_tokens for _ in range(len(qa_tensors))]
                hypos = []
                for s_list, num_s_list, t_list, num_t_list in batch_for_scorer(src_tokens_list, num_src_token_list,
                                                                               qa_tensors,
                                                                               [x.numel() for x in qa_tensors], bsz):
                    if type(s_list) is not list:
                        s_list = [s_list]
                    if type(num_s_list) is not list:
                        num_s_list = [num_s_list]
                    if type(t_list) is not list:
                        t_list = [t_list]
                    if type(num_t_list) is not list:
                        num_t_list = [num_t_list]

                    dataset = LanguagePairDataset(s_list, num_s_list,
                                                  bart.task.source_dictionary,
                                                  t_list, num_t_list,
                                                  bart.task.target_dictionary,
                                                  shuffle=False)
                    sample = dataset.collater(dataset)
                    sample = utils.apply_to_sample(lambda tensor: tensor.cuda(), sample)
                    # print(sample)
                    generator = SequenceScorer(bart.task.target_dictionary, compute_alignment=False)
                    translations = bart.task.inference_step(
                        generator,
                        [bart.model],
                        sample,
                    )
                    translations = [v for _, v in sorted(zip(sample['id'].tolist(), translations))]
                    hypos += translations
                qa_id = 0
                for hypo_qas in qas_item:
                    for qa in hypo_qas['qas']:
                        hypo = hypos[qa_id]
                        qa['eval_ns'] = hypo[0]['score'].item()
                        qa['eval_uns'] = sum(hypo[0]['positional_scores']).item()
                        special_token_loc = (hypo[0]['tokens'] == special_token).nonzero()
                        ans_scores = hypo[0]['positional_scores'][special_token_loc+1:-1]
                        qa['eval_a_uns'] = sum(ans_scores).item() if ans_scores.numel() > 0 else 0.0
                        qa['eval_a_ns'] = qa['eval_a_uns'] * 1.0 / ans_scores.numel() if ans_scores.numel() > 0 else 0.0
                        qa['eval_pos_scores'] = hypo[0]['positional_scores'].tolist()
                        qa_id += 1
                        # print(hypo[0]['tokens'])
                        # print(hypo[0]['positional_scores'])
                json.dump(qas_item, out_text_f)
                out_text_f.write('\n')

                source_line = source_f.readline()
                target_line = target_f.readline()
                qas_line = qas_f.readline()
                count += 1
                # if count % 100 == 0:
                    # print("Generated {} lines from worker {}".format(count, job_idx))

        assert offset + count == end, "!worker ended at {}, should have been {}".format(
            offset + count,
            end
        )
        del bart
        torch.cuda.empty_cache()


def _run_qa_eval_gen_process_local(job_idx, *, input_source_file, input_target_file, input_qas_file, out_text_file,
                             offset, end, checkpoint_dir, ckp_file, bin_dir, args):
    bart = BARTModel.from_pretrained(
        checkpoint_dir,
        checkpoint_file=ckp_file,
        data_name_or_path=bin_dir
    )
    torch.cuda.set_device(torch.device("cuda:{}".format(job_idx)))
    bart.cuda()
    bart.eval()
    bart.half()

    def batch_for_scorer(source_tokens_list, num_source_token_list, target_tokens_list, num_target_token_list, bsz):
        length = len(source_tokens_list)
        s = 0
        while s < length:
            e = s + bsz
            yield source_tokens_list[s:e], num_source_token_list[s:e], \
                  target_tokens_list[s:e], num_target_token_list[s:e]
            s = e

    special_token = 50259

    count = 0
    # bsz = 32
    bsz = args.bsz
    print("Local worker is processing {}-{}".format(offset, end))
    with torch.no_grad():
        with open(input_source_file, 'r') as source_f, \
                open(input_qas_file, 'r') as qas_f, \
                open(input_target_file, 'r') as target_f, \
                open(out_text_file, 'w') as out_text_f:
            for _ in range(offset):
                source_f.readline()
                target_f.readline()
                qas_f.readline()
            source_line = source_f.readline()
            target_line = target_f.readline()
            qas_line = qas_f.readline()
            while source_line:
                if offset + count >= end:
                    break

                max_source_tokens = 1024
                if args.prepend_target:
                    src_tokens = bart.encode(target_line.strip() + ' ' + source_line.strip(), no_bos=True,
                                              input_is_bpe=False)
                else:
                    src_tokens = bart.encode(source_line.strip(), no_bos=True, input_is_bpe=False)

                if len(src_tokens) > max_source_tokens:
                    src_tokens[max_source_tokens - 1] = src_tokens[-1]
                src_tokens = src_tokens if len(src_tokens) <= max_source_tokens else src_tokens[:max_source_tokens]

                qas_item = json.loads(qas_line.strip())

                q_tensors = []
                for hypo_qas in qas_item:
                    for qa in hypo_qas['qas']:
                        q_tensor = bart.encode(qa['q'], no_bos=True, input_is_bpe=False)
                        q_tensor[-1] = special_token
                        q_tensors.append(q_tensor)

                num_src_tokens = src_tokens.numel()
                src_tokens_list = [src_tokens for _ in range(len(q_tensors))]
                num_src_token_list = [num_src_tokens for _ in range(len(q_tensors))]
                hypos = []
                for s_list, num_s_list, t_list, num_t_list in batch_for_scorer(src_tokens_list, num_src_token_list,
                                                                               q_tensors,
                                                                               [x.numel() for x in q_tensors], bsz):
                    if type(s_list) is not list:
                        s_list = [s_list]
                    if type(num_s_list) is not list:
                        num_s_list = [num_s_list]
                    if type(t_list) is not list:
                        t_list = [t_list]
                    if type(num_t_list) is not list:
                        num_t_list = [num_t_list]

                    dataset = LanguagePairDataset(s_list, num_s_list,
                                                  bart.task.source_dictionary,
                                                  t_list, num_t_list,
                                                  bart.task.target_dictionary,
                                                  shuffle=False,
                                                  input_feeding=False)
                    sample = dataset.collater(dataset)
                    sample = utils.apply_to_sample(lambda tensor: tensor.cuda(), sample)
                    # print(sample)
                    gen_args = SimpleNamespace(
                        beam=1,
                        max_len_b=50,
                    )
                    generator = bart.task.build_generator(gen_args)

                    translations = bart.task.inference_step(
                        generator,
                        [bart.model],
                        sample,
                        prefix_tokens=sample['target']
                    )
                    translations = [v for _, v in sorted(zip(sample['id'].tolist(), translations))]
                    hypos += translations
                qa_id = 0
                for hypo_qas in qas_item:
                    for qa in hypo_qas['qas']:
                        hypo = hypos[qa_id][0]
                        decoded_qa = bart.decode(hypo['tokens'])
                        q_a_split = decoded_qa.split(' strutConnector')
                        if len(q_a_split) == 2 and q_a_split[0] == qa['q']:
                            qa['eval_ans'] = q_a_split[1]
                        else:
                            print('Error in decoded qa: {} | {}'.format(q_a_split, qa['q']))
                            qa['eval_ans'] = ''
                        qa_id += 1
                        # print(hypo[0]['tokens'])
                        # print(hypo[0]['positional_scores'])
                json.dump(qas_item, out_text_f)
                out_text_f.write('\n')

                source_line = source_f.readline()
                target_line = target_f.readline()
                qas_line = qas_f.readline()
                count += 1
                if count % 100 == 0:
                    print("Generated {} lines from worker {}".format(count, job_idx))

        assert offset + count == end, "!worker ended at {}, should have been {}".format(
            offset + count,
            end
        )
        del bart
        torch.cuda.empty_cache()


def process_chunk_local(job_idx, *, input_file, out_text_file, offset, end, checkpoint_dir, ckp_file, bin_dir, args):
    bart = BARTModel.from_pretrained(
        checkpoint_dir,
        checkpoint_file=ckp_file,
        data_name_or_path=bin_dir
    )
    torch.cuda.set_device(torch.device("cuda:{}".format(job_idx)))
    bart.cuda()
    bart.eval()
    bart.half()

    count = 1
    # bsz = 32
    bsz = args.bsz
    print("Local worker is processing {}-{}".format(offset, end))
    with open(input_file, 'r') as f, \
            open(out_text_file, 'w') as out_text_f:
        for _ in range(offset):
            f.readline()
        line = f.readline()
        # f.seek(offset)
        # line = safe_readline(f)
        slines = [line.strip()]
        while line:
            # if end > 0 and f.tell() > end:
            if offset + count >= end:
                break
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch, score_batch, unnormalized_score_batch = bart.sample(slines,
                                                                 beam=args.beam,
                                                                 lenpen=1.0,
                                                                 max_len_b=args.max_len,
                                                                 min_len=args.min_len,
                                                                 sampling=args.sampling,
                                                                 sampling_topk=args.sampling_topk,
                                                                 sampling_topp=args.sampling_topp,
                                                                 return_all=args.return_all
                                                                 )
                assert len(hypotheses_batch) == len(score_batch) == len(unnormalized_score_batch), \
                    "lens not equal: {} and {} and {}".format(
                        len(hypotheses_batch), len(score_batch), len(unnormalized_score_batch)
                    )
                assert len(hypotheses_batch) == len(slines), "slines={}, generated_score length={}".format(
                    slines, len(hypotheses_batch)
                )

                for t, s, unnormalized_s in zip(hypotheses_batch, score_batch, unnormalized_score_batch):
                    d = {
                        'summaries': t if type(t) is list else [t,],
                        'scores': s if type(s) is list else [s,],
                        'unnorm_scores': unnormalized_s if type(unnormalized_s) is list else [unnormalized_s,]
                         }
                    json.dump(d, out_text_f)
                    out_text_f.write('\n')
                out_text_f.flush()
                slines = []
            line = f.readline()
            slines.append(line.strip())
            count += 1
            if count % 100 == 0:
                print("Generated {} lines from worker {}".format(count, job_idx))

        if slines != []:
            with torch.no_grad():
                hypotheses_batch, score_batch, unnormalized_score_batch = bart.sample(slines,
                                                            beam=args.beam,
                                                            lenpen=1.0,
                                                            max_len_b=args.max_len,
                                                            min_len=args.min_len,
                                                            sampling=args.sampling,
                                                            sampling_topk=args.sampling_topk,
                                                            sampling_topp=args.sampling_topp,
                                                            return_all=args.return_all
                                                            )
            for t, s, unnormalized_s in zip(hypotheses_batch, score_batch, unnormalized_score_batch):
                d = {
                    'summaries': t if type(t) is list else [t, ],
                    'scores': s if type(s) is list else [s, ],
                    'unnorm_scores': unnormalized_s if type(unnormalized_s) is list else [unnormalized_s, ]
                }
                json.dump(d, out_text_f)
                out_text_f.write('\n')
            out_text_f.flush()
        assert offset + count == end, "!worker ended at {}, should have been {}".format(
            offset + count,
            end
        )
    del bart
    torch.cuda.empty_cache()


def check_score_file(file_path):
    i = 1
    with open(file_path, 'r') as f:
        line = f.readline()
        while line:
            if int(line.split(',')[0]) != i:
                print(i, line)
                break
            i += 1
            line = f.readline()

class SMInference(object):
    def __init__(self, base_dir, output_suffix, num_workers, mode,
                 checkpoint_dir, ckp_file, bin_dir, args):
        self.content_type = 'application/json'
        self.accept_type = 'application/json'
        self.base_dir = base_dir
        self.output_suffix = output_suffix
        self.num_workers = num_workers
        self.mode = mode
        self.checkpoint_dir = checkpoint_dir
        self.ckp_file = ckp_file
        self.bin_dir = bin_dir
        self.args = args
        self.completed = []
        if args.task == 'gen_summary':
            for file in glob.glob(os.path.join(self.args.output_dir, '*.{}.hypo'.format(output_suffix))):
                filename = os.path.basename(file)
                self.completed.append(filename)
        elif args.task == 'gen_question':
            for file in glob.glob(os.path.join(self.args.output_dir, '*.{}.question'.format(output_suffix))):
                filename = os.path.basename(file)
                self.completed.append(filename)
        elif args.task == 'gen_qa':
            for file in glob.glob(os.path.join(self.args.output_dir, '*.{}.qas'.format(output_suffix))):
                filename = os.path.basename(file)
                self.completed.append(filename)
        elif args.task == 'qa_eval' or args.task == 'qa_eval_gen':
            for file in glob.glob(os.path.join(self.args.output_dir, '*.{}'.format(output_suffix))):
                filename = os.path.basename(file)
                self.completed.append(filename)


    def run_q_gen(self):
        source_files = sorted(list(glob.glob(os.path.join(self.args.base_dir, self.args.source_dir, self.args.input_file))))
        ans_files = sorted(list(
            glob.glob(os.path.join(self.args.base_dir, self.args.answer_dir, self.args.ans_file))))
        assert len(source_files) == len(ans_files)

        for source_file, ans_file in zip(source_files, ans_files):
            filename = os.path.basename(source_file)
            if filename + '.{}.question'.format(self.output_suffix) in self.completed:
                print("Skipping {} and {}: already done!".format(source_file, ans_file))
                continue

            n_lines_source = count_lines_in_text_file(source_file)
            n_lines_ans = count_lines_in_text_file(ans_file)
            assert n_lines_source == n_lines_ans

            output_prefix = filename + '.' + self.output_suffix
            print("Processing {} lines in {} and {}".format(n_lines_source, source_file, ans_file))
            step = n_lines_source // self.num_workers
            offsets = [i * step for i in range(self.num_workers)]
            offsets.append(n_lines_source)
            if self.mode == 'local':
                with ProcessPool(ncpus=self.num_workers) as pool:
                    process_func = lambda job_idx: _run_q_gen_process_local(
                        job_idx,
                        input_source_file=source_file,
                        input_ans_file=ans_file,
                        out_text_file=os.path.join(self.args.output_dir, "{}.question{}".format(output_prefix, job_idx)),
                        offset=offsets[job_idx],
                        end=offsets[job_idx + 1],
                        checkpoint_dir=self.checkpoint_dir,
                        ckp_file=self.ckp_file,
                        bin_dir=self.bin_dir,
                        args=self.args
                    )
                    pool_results = pool.uimap(process_func, list(range(self.num_workers)))
                    for res in pool_results:
                        print('Done with process {}'.format(res))
            self.concat_temp_files(output_prefix+'.question')
            print('Written to {}'.format(output_prefix + '.question'))
            if not self.args.iterate_files:
                break

    def run_qa_gen(self):
        source_files = sorted(list(glob.glob(os.path.join(self.args.base_dir, self.args.source_dir, args.input_file))))

        for source_file in source_files:
            filename = os.path.basename(source_file)
            if filename + '.{}.qas'.format(self.output_suffix) in self.completed:
                print("Skipping {}: already done!".format(source_file))
                continue

            n_lines_source = count_lines_in_text_file(source_file)

            output_prefix = filename + '.' + self.output_suffix
            print("Processing {} lines in {}".format(n_lines_source, source_file))
            step = n_lines_source // self.num_workers
            offsets = [i * step for i in range(self.num_workers)]
            offsets.append(n_lines_source)
            if self.mode == 'local':
                if self.args.batch_lines:
                    with ProcessPool(ncpus=self.num_workers) as pool:
                        process_func = lambda job_idx: _run_qa_gen_process_local_batch_lines(
                            job_idx,
                            input_source_file=source_file,
                            out_text_file=os.path.join(self.args.output_dir, "{}.qas{}".format(output_prefix, job_idx)),
                            offset=offsets[job_idx],
                            end=offsets[job_idx + 1],
                            checkpoint_dir=self.checkpoint_dir,
                            ckp_file=self.ckp_file,
                            bin_dir=self.bin_dir,
                            args=self.args
                        )
                        pool_results = pool.uimap(process_func, list(range(self.num_workers)))
                        for res in pool_results:
                            print('Done with process {}'.format(res))
                else:
                    with ProcessPool(ncpus=self.num_workers) as pool:
                        process_func = lambda job_idx: _run_qa_gen_process_local(
                            job_idx,
                            input_source_file=source_file,
                            out_text_file=os.path.join(self.args.output_dir, "{}.qas{}".format(output_prefix, job_idx)),
                            offset=offsets[job_idx],
                            end=offsets[job_idx + 1],
                            checkpoint_dir=self.checkpoint_dir,
                            ckp_file=self.ckp_file,
                            bin_dir=self.bin_dir,
                            args=self.args
                        )
                        pool_results = pool.uimap(process_func, list(range(self.num_workers)))
                        for res in pool_results:
                            print('Done with process {}'.format(res))
            self.concat_temp_files(output_prefix+'.qas')
            print('Written to {}'.format(output_prefix + '.qas'))
            if not self.args.iterate_files:
                break

    def run_qa_eval(self):
        source_files = sorted(list(glob.glob(os.path.join(self.args.base_dir, self.args.source_dir,
                                                          self.args.source_file))))
        target_files = sorted(list(glob.glob(os.path.join(self.args.base_dir, self.args.source_dir,
                                                          self.args.target_file))))
        qas_files = sorted(list(glob.glob(os.path.join(self.args.base_dir, self.args.qas_dir,
                                                          self.args.input_file))))
        assert len(source_files) == len(target_files) == len(qas_files)
        print("Entering run_qa_eval:")
        # print("source_files={}".format(source_files))
        # print("target_files={}".format(target_files))
        # print("qas_files={}".format(qas_files))

        for source_file, target_file, qas_file in zip(source_files, target_files, qas_files):
            filename = os.path.basename(source_file)
            if filename + '.{}'.format(self.output_suffix) in self.completed:
                print("Skipping {}: already done!".format(source_file))
                continue

            n_lines_source = count_lines_in_text_file(source_file)

            output_prefix = filename + '.' + self.output_suffix
            print("Processing {} lines in {}".format(n_lines_source, source_file))
            step = n_lines_source // self.num_workers
            offsets = [i * step for i in range(self.num_workers)]
            offsets.append(n_lines_source)
            if self.args.task == 'qa_eval':
                _func = _run_qa_eval_process_local
            elif self.args.task == 'qa_eval_gen':
                _func = _run_qa_eval_gen_process_local

            if self.mode == 'local':
                with ProcessPool(ncpus=self.num_workers) as pool:
                    process_func = lambda job_idx: _func(
                        job_idx,
                        input_source_file=source_file,
                        input_target_file=target_file,
                        input_qas_file=qas_file,
                        out_text_file=os.path.join(self.args.output_dir, "{}{}".format(output_prefix, job_idx)),
                        offset=offsets[job_idx],
                        end=offsets[job_idx + 1],
                        checkpoint_dir=self.checkpoint_dir,
                        ckp_file=self.ckp_file,
                        bin_dir=self.bin_dir,
                        args=self.args
                    )
                    pool_results = pool.uimap(process_func, list(range(self.num_workers)))
                    for res in pool_results:
                        print('Done with process {}'.format(res))
            self.concat_temp_files(output_prefix)
            print('Written to {}'.format(output_prefix))
            if not self.args.iterate_files:
                break

    def run(self):
        for h in tqdm(glob.glob(os.path.join(self.base_dir, args.input_file+'*'))):
            filename = os.path.basename(h)

            if filename + '.{}.hypo'.format(self.output_suffix) in self.completed:
                print("Skipping {}: already done!".format(filename))
                continue

            output_prefix = filename + '.' + self.output_suffix
            source_full_path = os.path.join(self.base_dir, filename)
            n_lines = count_lines_in_text_file(source_full_path)
            print("Processing {} lines in {}".format(n_lines, source_full_path))
            step = n_lines // self.num_workers
            offsets = [i * step for i in range(self.num_workers)]
            offsets.append(n_lines)
            if self.mode == 'local':
                with ProcessPool(ncpus=self.num_workers) as pool:
                    process_func = lambda job_idx: process_chunk_local(
                        job_idx,
                        input_file=source_full_path,
                        out_text_file=os.path.join(self.args.output_dir, "{}.hypo{}".format(output_prefix, job_idx)),
                        offset=offsets[job_idx],
                        end=offsets[job_idx + 1],
                        checkpoint_dir=self.checkpoint_dir,
                        ckp_file=self.ckp_file,
                        bin_dir=self.bin_dir,
                        args=self.args
                    )
                    pool_results = pool.uimap(process_func, list(range(self.num_workers)))
                    for res in pool_results:
                        print('Done with process {}'.format(res))

            self.concat_temp_files(output_prefix+'.hypo')
            print('Written to {}'.format(output_prefix + '.hypo'))
            if not self.args.iterate_files:
                break

    def concat_temp_files(self, final_file_name):
        temp_files = glob.glob(os.path.join(self.args.output_dir, final_file_name+'*'))
        if len(temp_files) == 0:
            print("No temporary file found in {} that matches {}".format(self.args.output_dir, final_file_name+'*'))
        elif len(temp_files) == 1:
            print("Only one temporary file: {}. Renaming it now!".format(temp_files[0]))
            os.rename(temp_files[0], os.path.join(self.args.output_dir, final_file_name))
        else:
            with open(os.path.join(self.args.output_dir, final_file_name), 'w') as f:
                for worker_id in range(0, self.num_workers):
                    temp_file_path = os.path.join(self.args.output_dir, final_file_name+str(worker_id))
                    print("Concatenating ", temp_file_path)
                    with open(temp_file_path, 'r') as f_in:
                        for line in f_in:
                            f.write(line)
                    os.remove(temp_file_path)


def _run_answer_process(job_idx, *, n_files, offsets, args):
    cmd = 'python answer_extraction.py --base_dir {} --output_dir {} --input_file {} ' \
          '--n_files {} --offset {} --end {}'.format(
        args.base_dir,
        args.output_dir,
        args.input_file,
        n_files,
        offsets[job_idx],
        offsets[job_idx+1]
    )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        encoding='utf-8',
        bufsize=0
    )

    while proc.poll() is None:
        print(proc.stdout.readline().strip())
    print(proc.stdout.read().strip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--mode', type=str, default="local")
    parser.add_argument('--task', type=str, default="gen_summary") # gen_answer
    parser.add_argument('--base_dir', type=str, default="/data/fairseq_bart/question_generation/distill_qa")
    parser.add_argument('--input_file', type=str, default="")
    parser.add_argument('--checkpoint_dir', type=str, default="/data/exps/fairseq_bart_question_generation/model")
    parser.add_argument('--ckp_file', type=str, default="checkpoint2.pt")
    parser.add_argument('--bin_dir', type=str, default="/data/fairseq_bart/question_generation/data_bin")
    parser.add_argument('--output_dir', type=str, default="")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument('--bsz', type=int, default=32, help='batch size for generation')
    parser.add_argument("--beam", type=int, default=6)
    parser.add_argument("--max_len", type=int, default=140)
    parser.add_argument("--min_len", type=int, default=55)
    parser.add_argument('--sampling', type=str2bool, nargs='?', const=True, default=False,
                       help='whether to use sampling or not')
    parser.add_argument('--sampling_topk', type=int, default=-1, help='sampling_topk, -1 to disable')
    parser.add_argument('--sampling_topp', type=float, default=-1.0, help='sampling_topp, -1.0 to disable')
    parser.add_argument('--diverse_beam_groups', default=-1, type=int, metavar='N',
                       help='number of groups for Diverse Beam Search')
    parser.add_argument('--diverse_beam_strength', default=0.5, type=float, metavar='N',
                       help='strength of diversity penalty for Diverse Beam Search')

    parser.add_argument('--iterate_files', type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--encoder_json', type=str, default="/home/ec2-user/fairseq/encoder.json")
    parser.add_argument('--vocab_bpe', type=str, default="/home/ec2-user/fairseq/vocab.bpe")

    # for q_gen, specify the subdirectories for generated hypothesis and answer spans, under the base_dir
    parser.add_argument('--source_dir', type=str, default="")
    parser.add_argument('--answer_dir', type=str, default="")

    # for qa_eval, specify the subdirectory for *.qas_filtered files, under the base_dir
    parser.add_argument('--qas_dir', type=str, default="")
    parser.add_argument('--source_file', type=str, default='train.source.split*')
    parser.add_argument('--target_file', type=str, default='train.target.split*')
    parser.add_argument('--ans_file', type=str, default='train.source.split*.hypo.ans')

    parser.add_argument('--return_all', type=str2bool, nargs='?', const=True, default=True,
                       help='whether to return all hypothesis or just the first one')
    parser.add_argument('--return_token_scores', type=str2bool, nargs='?', const=True, default=False,
                       help='whether to return token level scores and tokens in QAGen.')
    parser.add_argument('--batch_lines', type=str2bool, nargs='?', const=True, default=False,
                       help='whether to return all hypothesis or just the first one')
    parser.add_argument('--prepend_target', type=str2bool, nargs='?', const=True, default=True,
                       help='whether prepend ground truth summary to the source when evaluating qa score.')

    args, unparsed = parser.parse_known_args()

    if args.output_dir:
        if not os.path.isdir(args.output_dir):
            print("Output dir does not exisit. Creating: {}".format(args.output_dir))
            os.makedirs(args.output_dir)

    assert args.task in ['gen_summary', 'gen_answer', 'gen_question', 'gen_qa', 'qa_eval', 'qa_eval_gen',]
    if args.task in ['gen_summary', 'gen_question', 'gen_qa', 'qa_eval', 'qa_eval_gen',]:
        if args.task == 'qa_eval':
            output_suffix = 'source_eval'
            if not args.prepend_target:
                output_suffix += '_noprepend'
        elif args.task == 'qa_eval_gen':
            output_suffix = 'source_gen_eval'
            if not args.prepend_target:
                output_suffix += '_noprepend'
        elif args.sampling:
            output_suffix = 'sampling{}'.format(args.beam)
        else:
            output_suffix = 'beam{}'.format(args.beam)
        inference_object = SMInference(
            base_dir=args.base_dir,
            output_suffix=output_suffix,
            num_workers=args.num_workers,
            mode=args.mode,
            checkpoint_dir=args.checkpoint_dir,
            ckp_file=args.ckp_file,
            bin_dir=args.bin_dir,
            args=args
        )
        if args.task == 'gen_summary':
            inference_object.run()
        elif args.task == 'gen_question':
            inference_object.run_q_gen()
        elif args.task == 'gen_qa':
            inference_object.run_qa_gen()
        elif args.task == 'qa_eval' or args.task == 'qa_eval_gen':
            inference_object.run_qa_eval()

    elif args.task == 'gen_answer':
        input_files = list(glob.glob(os.path.join(args.base_dir, args.input_file)))
        print("Entering gen_answer, found files: {}".format(input_files))
        n_files = len(input_files)
        step = n_files // args.num_workers
        offsets = [i * step for i in range(args.num_workers)]
        offsets.append(n_files)

        with ProcessPool(ncpus=args.num_workers) as pool:
            process_func = lambda job_idx: _run_answer_process(
                job_idx,
                n_files=n_files,
                offsets=offsets,
                args=args
            )
            pool_results = pool.uimap(process_func, list(range(args.num_workers)))

            for res in pool_results:
                print('Done with process {}'.format(res))
