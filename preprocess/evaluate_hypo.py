# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

from tqdm import tqdm
from rouge_score import rouge_scorer
import scispacy
import spacy
from ner import entity_match
from data_prepro_clean import TRACKING_ENTITY_LIST
import os
import glob
import subprocess
import numpy as np
import json
import csv
import time
from pathos.multiprocessing import ProcessPool
import argparse
from collections import OrderedDict
import re
import string
import collections
from types import SimpleNamespace


def count_lines_in_text_file(filename):
    with open(filename, 'r') as f:
        line_count = 0
        for _ in f:
            line_count += 1
    return line_count


def ent_count_match(nlp, base, parent, is_scispacy=False):
    # perform NER on base, then match in parant:
    doc = nlp(base)
    ent_count_base = 0
    en_count_in_base_parent = 0
    if is_scispacy:
        for e in doc.ents:
            ent_count_base += 1
            # if e.text in source:
            match_result = entity_match(e.text, parent, 1)
            if match_result:
                en_count_in_base_parent += 1
    else:
        for e in doc.ents:
            if e[0].ent_type_ in TRACKING_ENTITY_LIST:
                ent_count_base += 1
                # if e.text in source:
                match_result = entity_match(e.text, parent, 2)
                if match_result:
                    en_count_in_base_parent += 1
    return ent_count_base, en_count_in_base_parent

def _run_hypo_eval_process(job_idx, *, source_file, target_file, hypo_file, offsets, metric_names):
    nlp = spacy.load("en_core_web_lg")
    result = {}
    for m in metric_names:
        result[m] = []

    count = 1
    offset = offsets[job_idx]
    end = offsets[job_idx + 1]
    print("Local worker is processing {}-{}".format(offset, end))

    with open(source_file, 'r') as s_f, \
            open(target_file, 'r') as t_f, \
            open(hypo_file, 'r') as h_f:
        for _ in range(offset):
            s_f.readline()
            t_f.readline()
            h_f.readline()
        sline = s_f.readline()
        tline = t_f.readline()
        hline = h_f.readline()
        while sline and tline:
            if offset + count >= end:
                break
            sline = sline.strip()
            tline = tline.strip()
            hline = hline.strip()
            ent_count_hypo, ent_count_hypo_source = ent_count_match(nlp, hline, sline)
            ent_count_hypo, ent_count_hypo_target = ent_count_match(nlp, hline, tline)
            ent_count_target, ent_count_target_hypo = ent_count_match(nlp, tline, hline)

            result['ent_count_hypo'].append(ent_count_hypo)
            result['ent_count_hypo_source'].append(ent_count_hypo_source)
            result['ent_count_target'].append(ent_count_target)
            result['ent_count_target_hypo'].append(ent_count_target_hypo)
            result['ent_count_hypo_target'].append(ent_count_hypo_target)
            if ent_count_hypo == 0:
                result['precision_source'].append(np.nan)
                result['precision_target'].append(np.nan)
            else:
                result['precision_source'].append(ent_count_hypo_source * 1.0 / ent_count_hypo)
                result['precision_target'].append(ent_count_hypo_target * 1.0 / ent_count_hypo)
            if ent_count_target == 0:
                result['recall'].append(np.nan)
            else:
                result['recall'].append(ent_count_target_hypo * 1.0 / ent_count_target)

        assert offset + count == end, "!worker ended at {}, should have been {}".format(
            offset + count,
            end
        )
        return result

def fix_empty_lines(filename):
    count_empty = 0
    with open(filename+'.temp', 'w') as f_out:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():
                    f_out.write(line)
                else:
                    f_out.write("EMPTY\n")
                    count_empty += 1

    print("fixed {} empty lines in {}".format(count_empty, filename))
    assert count_lines_in_text_file(filename) == count_lines_in_text_file(filename+'.temp')

    if count_empty > 0:
        # if os.path.exists(filename + ".tokenized"):
        #     os.remove(filename + ".tokenized")
        os.rename(filename, filename+'.orig')
        os.rename(filename+'.temp', filename)
    else:
        if os.path.exists(filename+'.temp'):
            os.remove(filename+'.temp')

    return count_empty


def evaluate_hypo(source_file, target_file, hypo_file, output_file, eval_rouge=True, rouge_package='files2rouge',
                  no_prec_recall=False):
    n_s = count_lines_in_text_file(source_file)
    n_t = count_lines_in_text_file(target_file)
    n_h = count_lines_in_text_file(hypo_file)

    rouge1, rouge2, rougeL = None, None, None

    assert n_s == n_t == n_h, \
        "Number of lines not consistent: {}, {}, {}".format(n_s, n_t, n_h)

    metric_names = [
        'ent_count_hypo', 'ent_count_hypo_source',
        'ent_count_target', 'ent_count_target_hypo',
        'ent_count_hypo_target',
        'precision_source',
        'precision_target',
        'recall',
    ]
    if eval_rouge and rouge_package == "rouge_scorer":
        metric_names += ['rouge1', 'rouge2', 'rougeL', ]

    if eval_rouge and rouge_package == "files2rouge":
        os.environ["CLASSPATH"] = "/home/ec2-user/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar"

        if not os.path.exists(target_file + ".tokenized"):
            # remove line seperator u'\u2028'
            with open(target_file, 'r') as f:
                lines = f.readlines()
            with open(target_file + '.tmp', 'w') as fout:
                for line in lines:
                    fout.write(" ".join(line.strip().split(u'\u2028')) + '\n')
            assert n_t == count_lines_in_text_file(target_file + '.tmp')
            print("Tokenizing:", target_file)
            cmd = "cat {} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > {}".format(
                target_file + '.tmp',
                target_file + ".tokenized"
            )
            # print(cmd)
            with subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True
            ) as p:
                stdout, stderr = p.communicate()
                print(stdout)

        if not os.path.exists(hypo_file + ".tokenized"):
            print("Tokenizing:", hypo_file)
            cmd = "cat {} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > {}".format(
                hypo_file,
                hypo_file + ".tokenized"
            )
            # print(cmd)
            with subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True
            ) as p:
                stdout, stderr = p.communicate()
                print(stdout)

        fix_empty_lines(hypo_file + ".tokenized")

        cmd = "files2rouge {} {}".format(
            hypo_file + ".tokenized",
            target_file + ".tokenized"
        )
        rouge1_re = re.compile(r"ROUGE-1 Average_F: ([0-9\\.]+)")
        rouge2_re = re.compile(r"ROUGE-2 Average_F: ([0-9\\.]+)")
        rougeL_re = re.compile(r"ROUGE-L Average_F: ([0-9\\.]+)")

        with subprocess.Popen(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
        ) as p:
            stdout, stderr = p.communicate()
            out_text = stdout.decode()
            print(out_text)
            rouge1 = rouge1_re.findall(out_text)[0]
            rouge2 = rouge2_re.findall(out_text)[0]
            rougeL = rougeL_re.findall(out_text)[0]

    result = {}
    for m in metric_names:
        result[m] = []

    if eval_rouge and rouge_package == "rouge_scorer":
        scorer = rouge_scorer.RougeScorer([name for name in metric_names if name.startswith("rouge")],
                                          use_stemmer=True)

    # step = n_h // num_workers
    # offsets = [i * step for i in range(num_workers)]
    # offsets.append(n_h)
    if no_prec_recall:
        return
    if args.scispacy:
        nlp = spacy.load("en_ner_bc5cdr_md")
        print("Using scispacy!")
    else:
        nlp = spacy.load("en_core_web_lg")
    with open(source_file, 'r') as s_f, \
            open(target_file, 'r') as t_f, \
            open(hypo_file, 'r') as h_f:
        for _ in tqdm(range(n_h)):
            sline = s_f.readline().strip()
            tline = t_f.readline().strip()
            hline = h_f.readline().strip()
            if eval_rouge and rouge_package == "rouge_scorer":
                rouge = scorer.score(tline, hline)
                for m in rouge.keys():
                    result[m].append(rouge[m].fmeasure)

            ent_count_hypo, ent_count_hypo_source = ent_count_match(nlp, hline, sline, args.scispacy)
            ent_count_hypo, ent_count_hypo_target = ent_count_match(nlp, hline, tline, args.scispacy)
            ent_count_target, ent_count_target_hypo = ent_count_match(nlp, tline, hline, args.scispacy)

            result['ent_count_hypo'].append(ent_count_hypo)
            result['ent_count_hypo_source'].append(ent_count_hypo_source)
            result['ent_count_target'].append(ent_count_target)
            result['ent_count_target_hypo'].append(ent_count_target_hypo)
            result['ent_count_hypo_target'].append(ent_count_hypo_target)
            if ent_count_hypo == 0:
                result['precision_source'].append(np.nan)
                result['precision_target'].append(np.nan)
            else:
                result['precision_source'].append(ent_count_hypo_source * 1.0 / ent_count_hypo)
                result['precision_target'].append(ent_count_hypo_target * 1.0 / ent_count_hypo)
            if ent_count_target == 0:
                result['recall'].append(np.nan)
            else:
                result['recall'].append(ent_count_target_hypo * 1.0 / ent_count_target)

    avg_metrics = {}
    for k in metric_names:
        avg_metrics[k] = np.nanmean(np.array(result[k]))
        print("average {}={}".format(k, avg_metrics[k]))

    if rouge1 and rouge2 and rougeL and not no_prec_recall:
        macro_recall = avg_metrics['ent_count_target_hypo'] / avg_metrics['ent_count_target']
        macro_prec_target = avg_metrics['ent_count_hypo_target'] / avg_metrics['ent_count_hypo']
        micro_recall = avg_metrics['recall']
        micro_prec_target = avg_metrics['precision_target']
        display_text = f"{rouge1} {rouge2} {rougeL} " \
                       f"{avg_metrics['ent_count_hypo']} {avg_metrics['ent_count_hypo_source']} " \
                       f"{avg_metrics['ent_count_hypo_source'] / avg_metrics['ent_count_hypo']} " \
                       f"{avg_metrics['ent_count_target']} {avg_metrics['ent_count_target_hypo']} " \
                       f"{macro_recall} " \
                       f"{avg_metrics['ent_count_hypo_target']} " \
                       f"{macro_prec_target} " \
                       f"{avg_metrics['precision_source']} {micro_prec_target} " \
                       f"{micro_recall} " \
                       f"{2 * macro_recall * macro_prec_target / (macro_recall + macro_prec_target)} " \
                       f"{2 * micro_recall * micro_prec_target / (micro_recall + micro_prec_target)}"
        print(display_text)

    with open(output_file, 'w') as outfile:
        for i in range(n_h):
            text = ""
            for k in metric_names:
                text += "{} ".format(result[k][i])
            outfile.write(text + '\n')
    print("Output saved: ", output_file)


def spacy_tokenize(input_file):
    nlp = spacy.load("en_core_web_lg")
    outfile = input_file + '.tokenized'
    with open(input_file, 'r') as f_in, \
            open(outfile, 'w') as f_out:
        for line in f_in:
            doc = nlp(line)
            f_out.write(" ".join([token.text for token in doc]) + '\n')
    print("Done!")


def _load_hypos(hypo_files, select_hypo_ind):
    hypos_out = []
    line_count = 0
    for lm_file in hypo_files:
        with open(lm_file, 'r') as lm_f:
            for line in lm_f:
                hypo_list = json.loads(line.strip())
                hypos_out.append(hypo_list[select_hypo_ind[line_count]]['context'].strip())
                line_count += 1
    print("Read {} hypos from {}".format(len(hypos_out), hypo_files))
    return hypos_out


def make_unlikelihood_dataset(args):
    print('Entering make_unlikelihood_dataset')
    assert args.unlike_select_ratio > 0.0 and args.unlike_select_ratio <= 1.0
    if args.score_type == 'lm':
        index_file_name = 'untarget.index'
        if args.separate_target_untarget:
            target_dir = os.path.join(args.base_dir, args.sub_dir, 'target-{}'.format(
                int(args.target_select_ratio * 100)))
            untarget_dir = os.path.join(args.base_dir, args.sub_dir, 'untarget-{}'.format(
                int(args.unlike_select_ratio * 100)))
        elif args.output_dir:
            out_dir = args.output_dir
        elif args.target_select_ratio > 0.0:
            out_dir = os.path.join(args.base_dir, args.sub_dir, 'lm-{}-{}'.format(int(args.unlike_select_ratio * 100),
                                                                                  int(args.target_select_ratio * 100)))
        else:
            out_dir = os.path.join(args.base_dir, args.sub_dir, 'lm-{}'.format(int(args.unlike_select_ratio * 100)))
        # if args.filter_ans_lm_score:
        #     metric_choice = 'ans_lm'
        # else:
        #     metric_choice = 'eval_ns-ns'
        assert args.metric_choice == 'ans_lm' or args.metric_choice == 'eval_ns-ns'
        metric_choice = args.metric_choice
        print("Making unlikelihood dataset using {}".format(metric_choice))

    elif args.score_type == 'f1':
        index_file_name = 'untarget_f1.index'
        # out_dir = os.path.join(args.base_dir, args.sub_dir, 'f1-{}'.format(int(args.unlike_select_ratio * 100)))
        metric_choice = 'lm_f1'
        if args.separate_target_untarget:
            target_dir = os.path.join(args.base_dir, args.sub_dir, 'f1-target-{}'.format(
                int(args.target_select_ratio * 100)))
            untarget_dir = os.path.join(args.base_dir, args.sub_dir, 'f1-untarget-{}'.format(
                int(args.unlike_select_ratio * 100)))
        elif args.output_dir:
            out_dir = args.output_dir
        elif args.target_select_ratio > 0.0:
            out_dir = os.path.join(args.base_dir, args.sub_dir, 'f1-{}-{}'.format(int(args.unlike_select_ratio * 100),
                                                                                  int(args.target_select_ratio * 100)))
        else:
            out_dir = os.path.join(args.base_dir, args.sub_dir, 'f1-{}'.format(int(args.unlike_select_ratio * 100)))
        print("Making unlikelihood dataset using {}".format(metric_choice))
    elif args.score_type == 'rouge':
        index_file_name = 'untarget_rouge.index'
        metric_choice = 'rouge'
        if args.separate_target_untarget:
            raise Exception("We do not support seperate target-untarget dataset for rouge!")
        elif args.output_dir:
            out_dir = args.output_dir
        else:
            out_dir = os.path.join(args.base_dir, args.sub_dir, 'rouge-{}-{}'.format(int(args.unlike_select_ratio * 100),
                                                                                  int(args.unlike_select_ratio * 100)))
        print("Making unlikelihood dataset using {}".format(metric_choice))
    else:
        raise Exception("Please specify score_type!")

    # read and sort scores from index file:
    target_scores = []
    if args.target_index_file:
        assert os.path.exists(args.target_index_file)
        count_1e5 = 0
        with open(args.target_index_file, 'r') as ind_f:
            for index_line in ind_f:
                index_dict = json.loads(index_line.strip())
                target_scores.append(index_dict['avg_value'][metric_choice])
                if target_scores[-1] == 1e5 or target_scores[-1] == -1e5:
                    count_1e5 += 1
        print("Read {} lines from {}, {} scores are 1e5".format(len(target_scores), args.target_index_file,
                                                                count_1e5))

    index_file = os.path.join(args.base_dir, args.sub_dir, index_file_name)
    assert os.path.exists(index_file)
    scores = []
    select_hypo_ind = []
    with open(index_file, 'r') as ind_f:
        for index_line in ind_f:
            index_dict = json.loads(index_line.strip())
            scores.append(index_dict['avg_value'][metric_choice])
            select_hypo_ind.append(index_dict['avg'][metric_choice])
    print("Read {} lines from {}".format(len(scores), index_file))
    additional_scores = []
    additional_select_hypo_ind = []
    if args.additional_index_file:
        with open(args.additional_index_file, 'r') as ind_f:
            for index_line in ind_f:
                index_dict = json.loads(index_line.strip())
                additional_scores.append(index_dict['avg_value'][metric_choice])
                additional_select_hypo_ind.append(index_dict['avg'][metric_choice])
        print("Read {} lines from {}".format(len(additional_scores), args.additional_index_file))

    # load hypos
    lm_files = sorted(list(
        glob.glob(os.path.join(args.base_dir, args.sub_dir, args.pattern))))
    hypos = _load_hypos(lm_files, select_hypo_ind)

    if additional_scores:
        assert args.additional_eval_patterns, "need to supply additional source_eval file patterns!"
        print("Loading additional hypos: {}".format(args.additional_eval_patterns))
        additional_hypos = _load_hypos(sorted(list(glob.glob(args.additional_eval_patterns))),
                                       additional_select_hypo_ind)
        merged_scores = []
        merged_hypos = []
        assert len(scores) == len(additional_scores) == len(hypos) == len(additional_hypos)
        if args.select_highest:
            for s, a_s, h, a_h in zip(scores, additional_scores, hypos, additional_hypos):
                if s > a_s:
                    merged_hypos.append(h)
                    merged_scores.append(s)
                else:
                    merged_hypos.append(a_h)
                    merged_scores.append(a_s)
        else:
            for s, a_s, h, a_h in zip(scores, additional_scores, hypos, additional_hypos):
                if s < a_s:
                    merged_hypos.append(h)
                    merged_scores.append(s)
                else:
                    merged_hypos.append(a_h)
                    merged_scores.append(a_s)
    else:
        merged_scores = scores
        merged_hypos = hypos

    if args.unlike_select_ratio == 1.0:
        selected_example_ind = list(range(len(merged_scores)))
    else:
        if not args.select_highest:
            sorted_example_ind = [i[0] for i in sorted(enumerate(merged_scores), key=lambda x: x[1])]
        else:
            sorted_example_ind = [i[0] for i in sorted(enumerate(merged_scores), key=lambda x: -x[1])]
        selected_example_ind = sorted_example_ind[:round(args.unlike_select_ratio * len(merged_scores))]

    selected_target_ind = []
    if target_scores:
        if args.target_select_ratio == 1.0:
            selected_target_ind = list(range(len(target_scores)))
        else:
            sorted_target_ind = [i[0] for i in sorted(enumerate(target_scores), key=lambda x: -x[1] if x[1] != 1e5 else x[1])]
            selected_target_ind = sorted_target_ind[:round(args.target_select_ratio * len(target_scores))]
        print("top 5 selected target indices:", [i for i in selected_target_ind[:5]])
        print("top 5 selected target scores:", [target_scores[i] for i in selected_target_ind[:5]])
        print("bottom 5 selected target indices:", [i for i in selected_target_ind[-5:]])
        print("bottom 5 selected target scores:", [target_scores[i] for i in selected_target_ind[-5:]])
        if not args.separate_target_untarget:
            selected_example_ind = list(set(selected_example_ind).intersection(set(selected_target_ind)))

    # write output files
    output_source_target = ""
    if args.separate_target_untarget:
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        if not os.path.exists(untarget_dir):
            os.mkdir(untarget_dir)
        output_untarget = os.path.join(untarget_dir, 'train.untarget')
        output_target = os.path.join(target_dir, 'train.target')
        output_source_untarget = os.path.join(untarget_dir, 'train.source')
        output_source_target = os.path.join(target_dir, 'train.source')
    else:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        output_untarget = os.path.join(out_dir, 'train.untarget')
        output_target = os.path.join(out_dir, 'train.target')
        output_source_untarget = os.path.join(out_dir, 'train.source')

    source_file = os.path.join(args.base_dir, 'train.source')
    with open(source_file, 'r') as f:
        source_lines = f.readlines()
    if not args.make_only_untarget:
        target_file = os.path.join(args.base_dir, 'train.target')
        with open(target_file, 'r') as f:
            target_lines = f.readlines()

        if args.separate_target_untarget:
            with open(output_untarget, 'w') as out_untarget_f, \
                    open(output_source_untarget, 'w') as out_source_f:
                for ind in selected_example_ind:
                    out_source_f.write(source_lines[ind])
                    out_untarget_f.write(merged_hypos[ind] + '\n')
            print('Wrote {} examples to {} and {}'.format(len(selected_example_ind), output_untarget,
                                                          output_source_untarget))

            with open(output_target, 'w') as out_target_f, \
                    open(output_source_target, 'w') as out_source_f:
                for ind in selected_target_ind:
                    out_source_f.write(source_lines[ind])
                    out_target_f.write(target_lines[ind])
            print('Wrote {} examples to {} and {}'.format(len(selected_target_ind), output_target,
                                                          output_source_target))
        else:
            with open(output_untarget, 'w') as out_untarget_f, \
                open(output_target, 'w') as out_target_f, \
                open(output_source_untarget, 'w') as out_source_f:
                for ind in selected_example_ind:
                    out_target_f.write(target_lines[ind])
                    out_source_f.write(source_lines[ind])
                    out_untarget_f.write(merged_hypos[ind] + '\n')
            print('Wrote {} examples to {} and {} and {}'.format(len(selected_example_ind), output_untarget,
                                                          output_target, output_source_untarget))
    else:
        with open(output_untarget, 'w') as out_untarget_f, \
            open(output_source_untarget, 'w') as out_source_f:
            for ind in selected_example_ind:
                out_source_f.write(source_lines[ind])
                out_untarget_f.write(merged_hypos[ind] + '\n')

        print('Wrote {} examples to {} and {}'.format(len(selected_example_ind), output_untarget,
                                                      output_source_untarget))


def convert_hypo_to_jsonl(args):
    for h in tqdm(glob.glob(os.path.join(args.base_dir, args.sub_dir, args.split + args.pattern))):
        hypo_filename = os.path.basename(h)
        print('Processing ', hypo_filename)
        out_file = h + '.hypo'
        with open(h, 'r') as f_in, \
            open(out_file, 'w') as f_out:
            for line in f_in:
                d = {
                    'summaries': [line.strip(),],
                    'scores': [0.0,],
                    'unnorm_scores': [0.0,]
                }
                json.dump(d, f_out)
                f_out.write('\n')


def _ans_lm_score(pos_scores, tokens, special_token=50259):
    ans_found = False
    score_sum = 0.0
    ignore_tokens = [0, 2, 4, 6, 328]
    for i, t in reversed(list(enumerate(tokens))):
        if t == special_token:
            ans_found = True
            return ans_found, score_sum
        if t not in ignore_tokens:
            score_sum += pos_scores[i]
    return ans_found, score_sum


def filter_qas_dataset_lm_score(args):
    qas_files = sorted(list(
        glob.glob(os.path.join(args.base_dir, args.sub_dir, args.pattern))))
    for qas_file in tqdm(qas_files):
        if args.filter_ans_lm_score:
            output_file = qas_file + '_filtered{}'.format(args.filter_ans_lm_threshold)
        else:
            output_file = qas_file + '_filtered'
        with open(qas_file, 'r') as qas_f, \
             open(output_file, 'w') as output_f:
            for line in qas_f:
                filtered_qa_dict_list = []
                qa_dict_list = json.loads(line.strip())
                for qa_dict in qa_dict_list:
                    filtered_qa_dict = {'context': qa_dict['context'], 'qas': []}
                    hypo_text_lower = qa_dict['context'].lower()
                    filtered_list = []
                    # make sure the question and answer can be extracted, and answer exists in hypo_text
                    if 'pos_scores' in qa_dict and 'toks' in qa_dict:
                        for qa, norm_score, unnorm_score, pos_score, tokens in zip(qa_dict['qa'],
                                                                                   qa_dict['norm_scores'],
                                                                                   qa_dict['unnorm_scores'],
                                                                                   qa_dict['pos_scores'],
                                                                                   qa_dict['toks']):
                            q_a_split = qa.split(' strutConnector')
                            if len(q_a_split) == 2 and q_a_split[1].lower() in hypo_text_lower:
                                filtered_list.append((q_a_split[0], q_a_split[1], norm_score, unnorm_score,
                                                      pos_score, tokens))
                    else:
                        for qa, norm_score, unnorm_score in zip(qa_dict['qa'], qa_dict['norm_scores'],
                                                                qa_dict['unnorm_scores']):
                            q_a_split = qa.split(' strutConnector')
                            if len(q_a_split) == 2 and q_a_split[1].lower() in hypo_text_lower:
                                filtered_list.append((q_a_split[0], q_a_split[1], norm_score, unnorm_score))
                    if not filtered_list:
                        filtered_qa_dict_list.append(filtered_qa_dict)
                        continue
                    if args.filter_ans_lm_score and 'pos_scores' in qa_dict and 'toks' in qa_dict: # filtering qa using answer lm scores:
                        for t in filtered_list:
                            ans_found, ans_score_sum = _ans_lm_score(t[4], t[5])
                            if  ans_score_sum >= args.filter_ans_lm_threshold:
                                filtered_qa_dict['qas'].append({'q': t[0], 'a': t[1], 'ns': t[2], 'uns': t[3],
                                                                'pos_s': t[4], 'toks': t[5]})
                    else:
                        filtered_list = sorted(filtered_list, key=lambda t: -t[3])
                        # form a ordered dictionary by going over the filtered list
                        seen_ans_dict = OrderedDict()
                        for tmp in filtered_list:
                            if tmp[1].lower() not in seen_ans_dict:
                                seen_ans_dict[tmp[1].lower()] = [tmp,]
                            else:
                                seen_ans_dict[tmp[1].lower()].append(tmp)
                        max_qas = 10
                        keep_adding = True
                        ans_question_set_dict = {}
                        while keep_adding and len(filtered_qa_dict['qas']) < max_qas:
                            keep_adding = False
                            for key, value in seen_ans_dict.items():
                                if value:
                                    tmp = value.pop(0)
                                    q, a, ns, uns = tmp[0], tmp[1], tmp[2], tmp[3]
                                    pos_s, toks = None, None
                                    if len(tmp) == 6:
                                        pos_s, toks = tmp[4], tmp[5]

                                    # if the question is repeated, don't add it.
                                    if a.lower() not in ans_question_set_dict:
                                        ans_question_set_dict[a.lower()] = set([q.lower()])
                                        if pos_s is None:
                                            filtered_qa_dict['qas'].append({'q': q, 'a': a, 'ns': ns, 'uns': uns})
                                        else:
                                            filtered_qa_dict['qas'].append({'q': q, 'a': a, 'ns': ns, 'uns': uns,
                                                                            'pos_s': pos_s, 'toks': toks})

                                        keep_adding = True
                                    elif q.lower() not in ans_question_set_dict[a.lower()]:
                                        ans_question_set_dict[a.lower()].add(q.lower())
                                        if pos_s is None:
                                            filtered_qa_dict['qas'].append({'q': q, 'a': a, 'ns': ns, 'uns': uns})
                                        else:
                                            filtered_qa_dict['qas'].append({'q': q, 'a': a, 'ns': ns, 'uns': uns,
                                                                            'pos_s': pos_s, 'toks': toks})
                                        keep_adding = True
                    filtered_qa_dict_list.append(filtered_qa_dict)
                json.dump(filtered_qa_dict_list, output_f)
                output_f.write('\n')


def select_unlikelihood_hypos_lm_score(args):
    assert os.path.exists(os.path.join(args.base_dir, args.sub_dir))

    def _compute_lm_for_hypos(hypo_list, metrics):
        return_hypo_ind_avg = {}
        return_hypo_ind_min = {}
        return_hypo_avg = {}
        return_hypo_min = {}
        prob_metric = 'eval_prob'
        return_hypo_ind_avg[prob_metric] = 0
        if args.select_highest:
            return_hypo_avg[prob_metric] = -1e5
        else:
            return_hypo_avg[prob_metric] = 1e5

        ans_lm_metric = 'ans_lm'
        return_hypo_ind_avg[ans_lm_metric] = 0
        if args.select_highest:
            return_hypo_avg[ans_lm_metric] = -1e5
        else:
            return_hypo_avg[ans_lm_metric] = 1e5
        for i, hypo in enumerate(hypo_list):
            sum_ans_lm_per_hypo = 0.0
            count_per_hypo = 0
            for qa in hypo['qas']:
                ans_found = False
                if 'pos_s' in qa:
                    ans_found, value = _ans_lm_score(np.array(qa['eval_pos_scores']) - np.array(qa['pos_s']), qa['toks'])
                if ans_found:
                    sum_ans_lm_per_hypo += value
                    count_per_hypo += 1
            if count_per_hypo > 0:
                avg_per_hypo = sum_ans_lm_per_hypo / count_per_hypo
                if not args.select_highest and avg_per_hypo < return_hypo_avg[ans_lm_metric]:
                    return_hypo_ind_avg[ans_lm_metric] = i
                    return_hypo_avg[ans_lm_metric] = avg_per_hypo
                if args.select_highest and avg_per_hypo > return_hypo_avg[ans_lm_metric]:
                    return_hypo_ind_avg[ans_lm_metric] = i
                    return_hypo_avg[ans_lm_metric] = avg_per_hypo

        for metric in metrics:
            return_hypo_ind_avg[metric] = 0
            return_hypo_ind_min[metric] = 0
            if args.select_highest:
                return_hypo_avg[metric] = -1e5
                # return_hypo_min[metric] = -1e5
            else:
                return_hypo_avg[metric] = 1e5
                # return_hypo_min[metric] = 1e5
            # min_avg_value = 1e5
            # min_min_value = 1e5
            for i, hypo in enumerate(hypo_list):
                sum_prob_per_hypo = 0.0
                sum_per_hypo = 0.0
                count_per_hypo = 0
                min_per_hypo = 1e5
                for qa in hypo['qas']:
                    metric_split = metric.split('-')
                    if len(metric_split) == 1:
                        value = qa[metric]
                    else:
                        value = qa[metric_split[0]] - qa[metric_split[1]]
                    if metric == 'eval_ns':
                        sum_prob_per_hypo += np.exp(value)
                    sum_per_hypo += value
                    if min_per_hypo > value:
                        min_per_hypo = value
                    count_per_hypo += 1
                if count_per_hypo > 0:
                    if metric == 'eval_ns':
                        avg_prob_per_hypo = sum_prob_per_hypo / count_per_hypo
                        if not args.select_highest and avg_prob_per_hypo < return_hypo_avg[prob_metric]:
                            return_hypo_avg[prob_metric] = avg_prob_per_hypo
                            return_hypo_ind_avg[prob_metric] = i
                        if args.select_highest and avg_prob_per_hypo > return_hypo_avg[prob_metric]:
                            return_hypo_avg[prob_metric] = avg_prob_per_hypo
                            return_hypo_ind_avg[prob_metric] = i

                    avg_per_hypo = sum_per_hypo / count_per_hypo
                    if not args.select_highest and avg_per_hypo < return_hypo_avg[metric]:
                        return_hypo_ind_avg[metric] = i
                        return_hypo_avg[metric] = avg_per_hypo
                    if args.select_highest and avg_per_hypo > return_hypo_avg[metric]:
                        return_hypo_ind_avg[metric] = i
                        return_hypo_avg[metric] = avg_per_hypo
                    # if min_per_hypo < return_hypo_min[metric]:
                    #     return_hypo_ind_min[metric] = i
                    #     return_hypo_min[metric] = min_per_hypo
        return return_hypo_ind_avg, return_hypo_ind_min, return_hypo_avg, return_hypo_min

    lm_files = sorted(list(
        glob.glob(os.path.join(args.base_dir, args.sub_dir, args.pattern))))
    metrics = ['eval_ns', 'eval_a_ns', 'eval_uns', 'eval_a_uns',
               'eval_ns-ns', 'eval_a_ns-ns', 'eval_uns-uns', 'eval_a_uns-uns']

    output_index_file = os.path.join(args.base_dir, args.sub_dir, 'untarget.index')
    avg_avg_scores = {}
    avg_min_scores = {}
    for metric in metrics:
        avg_avg_scores[metric] = 0.0
        avg_min_scores[metric] = 0.0
    avg_avg_scores['eval_prob'] = 0.0
    avg_avg_scores['ans_lm'] = 0.0
    count = 0
    # output_score_file = os.path.join(args.base_dir, args.sub_dir, args.split + '.qags')
    with open(output_index_file, 'w') as output_index_f:
        for lm_file in tqdm(lm_files):
            with open(lm_file, 'r') as lm_f:
                for line in lm_f:
                    hypo_list = json.loads(line.strip())
                    bad_hypo_ind_avg, bad_hypo_ind_min, min_avg_value, min_min_value = \
                        _compute_lm_for_hypos(hypo_list, metrics)
                    # print("min_avg_value:")
                    # print(min_avg_value)
                    # print("min_min_value")
                    # print(min_min_value)
                    if all([v != 1e5 for v in min_avg_value.values()]) and \
                            all([v != -1e5 for v in min_avg_value.values()]):
                        # all([v != 1e5 for v in min_min_value.values()]) and \
                        for metric in metrics:
                            avg_avg_scores[metric] += min_avg_value[metric]
                            # avg_min_scores[metric] += min_min_value[metric]
                        avg_avg_scores['eval_prob'] += min_avg_value['eval_prob']
                        avg_avg_scores['ans_lm'] += min_avg_value['ans_lm']
                        count += 1
                    json.dump({'avg': bad_hypo_ind_avg, 'min': bad_hypo_ind_min,
                               'avg_value': min_avg_value, 'min_value': min_min_value}, output_index_f)
                    output_index_f.write('\n')
    avg_out_text = ""
    # min_out_text = ""
    print("count = {}".format(count))
    for metric in metrics:
        if count > 0:
            avg_avg_scores[metric] /= count
            # avg_min_scores[metric] /= count
        avg_out_text += f"{avg_avg_scores[metric]} "
        # min_out_text += f"{avg_min_scores[metric]} "
        # avg_out_text += f"{metric}={avg_avg_scores[metric]} "
        # min_out_text += f"{metric}={avg_min_scores[metric]} "
    avg_out_text += f"{avg_avg_scores['eval_prob'] / count if count > 0 else 0.0} "
    avg_out_text += f"{avg_avg_scores['ans_lm'] / count if count > 0 else 0.0}"
    print("select_unlikelihood_hypos_lm_score Done! Index written to {}".format(output_index_file))
    print("avg lm scores: " + avg_out_text)
    if count == 0:
        print("Avg lm scores are not computed, probably because the return_token_scores option was set to False in running qa_gen in preprocess/sm_inference_asum.py")
    # print("min lm scores: " + min_out_text)


def compute_hypos_lm_score(args):
    '''
    compute the QUALS (eval_ns-ns) and save it for each hypothesis

    :param args:
    :return:
    '''
    assert os.path.exists(os.path.join(args.base_dir, args.sub_dir))
    lm_files = sorted(list(
        glob.glob(os.path.join(args.base_dir, args.sub_dir, args.pattern))))
    metrics = ['eval_ns-ns']

    for lm_file in tqdm(lm_files):
        output_index_file = lm_file + '.quals'
        with open(output_index_file, 'w') as output_index_f:
            with open(lm_file, 'r') as lm_f:
                for line in lm_f:
                    hypo_list = json.loads(line.strip())
                    hypo_avg = {}
                    for metric in metrics:
                        hypo_avg[metric] = []
                        for i, hypo in enumerate(hypo_list):
                            sum_per_hypo = 0.0
                            count_per_hypo = 0
                            for qa in hypo['qas']:
                                metric_split = metric.split('-')
                                if len(metric_split) == 1:
                                    value = qa[metric]
                                else:
                                    value = qa[metric_split[0]] - qa[metric_split[1]]
                                sum_per_hypo += value
                                count_per_hypo += 1
                            if count_per_hypo > 0:
                                hypo_avg[metric].append(sum_per_hypo / count_per_hypo)
                            else:
                                hypo_avg[metric].append(-1e5)

                    json.dump(hypo_avg, output_index_f)
                    output_index_f.write('\n')
        print("QUALS written to {}".format(output_index_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--sub_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default="")
    parser.add_argument('--pattern', type=str, default="")
    parser.add_argument('--no_prec_recall', default=False, action='store_true')
    parser.add_argument('--no_rouge', default=False, action='store_true')

    # for make_unlikelihood_dataset
    parser.add_argument('--unlike_select_ratio', type=float, default=-1.0,
                        help='ratio of unlikelihood examples to keep for training, must be (0, 1]')
    parser.add_argument('--target_select_ratio', type=float, default=-1.0,
                        help='ratio of unlikelihood examples to keep for training based on '
                             'filtering ground truth summary, must be (0, 1]')
    parser.add_argument('--target_index_file', type=str, default='',
                        help='untarget.index file based on target lm score filtering')
    parser.add_argument('--score_type', type=str, default='lm',
                        help='select unlikelihood examples based on "lm" or "f1" scores')

    # for filter_qas_dataset_lm_score
    parser.add_argument('--filter_ans_lm_score', default=False, action='store_true')
    parser.add_argument('--filter_ans_lm_threshold', type=float, default=-1.0,
                        help='threshold for selecting qa pairs based on answer lm score.')

    parser.add_argument('--select_highest', default=False, action='store_true')
    parser.add_argument('--make_only_untarget', default=False, action='store_true')
    parser.add_argument('--additional_eval_patterns', type=str, default="",
                        help='additional source_eval file full path patterns to merge with hypos in --pattern')
    parser.add_argument('--additional_index_file', type=str, default='',
                        help='additional untarget.index file based on target lm score filtering')
    parser.add_argument('--metric_choice', type=str, default='eval_ns-ns',
                        help='metric choice for making the dataset, options: eval_ns-ns, ans_lm')
    parser.add_argument('--separate_target_untarget', default=False, action='store_true')
    parser.add_argument('--output_dir', type=str, default="")

    parser.add_argument('--scispacy', default=False, action='store_true')

    args, unparsed = parser.parse_known_args()

    if args.mode == 'evaluate_summary':
        for h in tqdm(glob.glob(os.path.join(args.base_dir, args.sub_dir, args.split + args.pattern))):
            hypo_filename = os.path.basename(h)
            evaluate_hypo(
                source_file=os.path.join(args.base_dir, args.split + '.source'),
                target_file=os.path.join(args.base_dir, args.split + '.target'),
                hypo_file=os.path.join(args.base_dir, args.sub_dir, hypo_filename),
                output_file=os.path.join(args.base_dir, args.sub_dir, hypo_filename+'_eval'),
                eval_rouge=not args.no_rouge,
                rouge_package='files2rouge',
                no_prec_recall=args.no_prec_recall
            )
    if args.mode == 'filter_qas_dataset_lm_score':
        filter_qas_dataset_lm_score(args)
    if args.mode == 'make_unlikelihood_dataset':
        make_unlikelihood_dataset(args)
    if args.mode == 'convert_hypo_to_json':
        convert_hypo_to_jsonl(args)
    if args.mode == 'remove_ent_from_hypo':
        for h in tqdm(glob.glob(os.path.join(args.base_dir, args.sub_dir, args.split + args.pattern))):
            hypo_filename = os.path.basename(h)
            os.rename(h, h + '.ent')
            with open(h, 'w') as f_out, \
                open(h + '.ent', 'r') as f_in:
                for line_in in f_in:
                    line_split = line_in.split('strutConnector')
                    if len(line_split) > 1:
                        f_out.write(line_split[-1])
                    else:
                        f_out.write(line_in)
    if args.mode == 'compute_hypos_lm_score':
        compute_hypos_lm_score(args)
    if args.mode == 'select_unlikelihood_hypos_lm_score':
        select_unlikelihood_hypos_lm_score(args)
