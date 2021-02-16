# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

from tqdm import tqdm
import spacy
from ner import entity_match
from data_prepro_clean import TRACKING_ENTITY_LIST
import os
import glob
import subprocess
import numpy as np
import json
import argparse
import re


def count_lines_in_text_file(filename):
    with open(filename, 'r') as f:
        line_count = 0
        for _ in f:
            line_count += 1
    return line_count


def ent_count_match(nlp, base, parent):
    # perform NER on base, then match in parant:
    doc = nlp(base)
    ent_count_base = 0
    en_count_in_base_parent = 0
    for e in doc.ents:
        if e[0].ent_type_ in TRACKING_ENTITY_LIST:
            ent_count_base += 1
            # if e.text in source:
            match_result = entity_match(e.text, parent, 2)
            if match_result:
                en_count_in_base_parent += 1
    return ent_count_base, en_count_in_base_parent


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

    if eval_rouge and rouge_package == "files2rouge":

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

    if no_prec_recall:
        return
    nlp = spacy.load("en_core_web_lg")
    with open(source_file, 'r') as s_f, \
            open(target_file, 'r') as t_f, \
            open(hypo_file, 'r') as h_f:
        for _ in tqdm(range(n_h)):
            sline = s_f.readline().strip()
            tline = t_f.readline().strip()
            hline = h_f.readline().strip()

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


def convert_hypo_to_jsonl(args):
    for h in tqdm(glob.glob(os.path.join(args.base_dir + args.sub_dir, args.split + args.pattern))):
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

    parser.add_argument('--output_dir', type=str, default="")

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
