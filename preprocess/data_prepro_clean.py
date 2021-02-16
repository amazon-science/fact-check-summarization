# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import json
import os
from tqdm import tqdm
import glob
from ner import entity_match
import spacy
import hashlib
from subprocess import Popen, PIPE
import re
import argparse

SKIPTEXT = [
    "Share this with",
    "Email",
    "Facebook",
    "Messenger",
    "Twitter",
    "Pinterest",
    "WhatsApp",
    "LinkedIn",
    "Copy this link",
    "These are external links and will open in a new window",
            ]
SKIP_TEXT_LOWER = [t.lower() for t in SKIPTEXT]

TRACKING_ENTITY_LIST = ['PERSON', 'FAC', 'GPE', 'ORG', 'NORP', 'LOC', 'EVENT']


def fix_missing_period(line):
    dm_single_close_quote = u'\u2019'  # unicode
    dm_double_close_quote = u'\u201d'

    END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
                  ")"]  # acceptable ways to end a sentence

    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + "."


def select_example(nlp, intro, abstract, filter_level):
    if not (intro and abstract):
        return False
    if filter_level <= 1:
        return True

    doc = nlp(abstract)
    en_count_in_summary = 0
    select = True
    for e in doc.ents:
        if e[0].ent_type_ in TRACKING_ENTITY_LIST:
            en_count_in_summary += 1
            # if e.text in source:
            match_result = entity_match(e.text, intro, 2)
            if not match_result:
                select = False
                # print("ENTITY NOT FOUND: {}".format(e.text))
                # print(">>source>>", intro)
                # print(">>summary>>", abstract)
                break
    # if select and en_count_in_summary>0:
    if select:
        return True
    else:
        return False


def select_summary_sentences(nlp, intro, abstract, filter_level):
    if not (intro and abstract):
        return ""
    if filter_level <= 1:
        return abstract

    doc = nlp(abstract)
    en_count_in_summary = 0
    sentences_select = {}
    for sent in doc.sents:
        sentences_select[sent.text] = True
    for e in doc.ents:
        if e[0].ent_type_ in TRACKING_ENTITY_LIST:
            en_count_in_summary += 1
            # if e.text in source:
            match_result = entity_match(e.text, intro, 2)
            if not match_result:
                sentences_select[e.sent.text] = False
                # print("ENTITY NOT FOUND: {}".format(e.text))
                # print(">>source>>", intro)
                # print(">>summary>>", abstract)
                # break
    # if select and en_count_in_summary>0:
    result = []
    for sent in doc.sents:
        if sentences_select[sent.text]:
            result.append(sent.text)
    return " ".join(result)


def extract_text_xsum(input_file, filter_level):
    s_text, t_text = [], []
    intro, restbody = False, False
    try:
        with open(input_file, 'r') as input_f:
            for line in input_f:
                line = line.strip()
                if line:
                    if line == "[XSUM]INTRODUCTION[XSUM]":
                        intro = True
                        restbody = False
                        continue
                    if line == "[XSUM]RESTBODY[XSUM]":
                        intro = False
                        restbody = True
                        continue
                    if intro:
                        t_text.append(fix_missing_period(line))
                    elif restbody and line.lower() not in SKIP_TEXT_LOWER:
                        s_text.append(fix_missing_period(line))
                    elif restbody and filter_level == 0:
                        s_text.append(fix_missing_period(line))

        return " ".join(s_text), " ".join(t_text)
    except:
        return "", ""


def is_corrupt_summary_sent(sent):
    link_text = ["READ", "CLICK HERE"]
    for lt in link_text:
        if sent.startswith(lt):
            # print("======CORRUPT DETECTED===== {}".format(" ".join(sent)))
            return True
    return False


def extract_text_cnndm(input_file, filter_level):
    s_text, t_text = [], []
    flag = False
    try:
        with open(input_file, 'r') as input_f:
            for line in input_f:
                line = line.strip()
                if line:
                    if line == "@highlight":
                        flag = True
                        continue
                    if flag:
                        if is_corrupt_summary_sent(line):
                            # print("===CORRUPT DETECTED==={}-->{}".format(input_file, line))
                            if filter_level == 0:
                                print("CORRUPT ignored!")
                                t_text.append(fix_missing_period(line))
                        else:
                            t_text.append(fix_missing_period(line))
                    else:
                        s_text.append(fix_missing_period(line))
        return " ".join(s_text), " ".join(t_text)
    except:
        return "", ""


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()

def preprocess_cnndm(raw_dir="~/cnndm-data-dir", dataset_suffix='',
                     filter_level=0, output_dir="~/cnndm-data-dir/processed-data", write_inverse=False):
    '''
    preprocess cnndm dataset according to different filter levels.
    - filter_level=0: no special processing
    - filter_level=1: remove corruption text in summaries. (Undesirable texts included as a result
        of the imperfection in data collection. e.g. "CLICK HERE for all the latest Arsenal news.")
    - filter_level=2: entity hallucination filtering in addition to corruption text removal.
        A summary sentence is removed if it contains a named entity not in the source document.
    '''
    data_types = ['valid', 'test', 'train']
    output_directory = output_dir
    nlp = spacy.load("en_core_web_lg")

    corpus_mapping = {}
    for corpus_type in data_types:
        temp = []
        for line in open(os.path.join(raw_dir, 'mapping_' + corpus_type + '.txt')):
            temp.append(hashhex(line.strip()))
        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}

    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(os.path.join(raw_dir, 'raw_stories', '*.story')):
        real_name = f.split('/')[-1].split('.')[0]
        if (real_name in corpus_mapping['valid']):
            valid_files.append(f)
        elif (real_name in corpus_mapping['test']):
            test_files.append(f)
        elif (real_name in corpus_mapping['train']):
            train_files.append(f)

    split_dict = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for dtype in data_types:
        count = 0
        count_selected = 0
        print(dtype)
        if dtype == "valid":
            output_source = "val" + dataset_suffix + '.source'
            output_target = "val" + dataset_suffix + '.target'
        else:
            output_source = dtype + dataset_suffix + '.source'
            output_target = dtype + dataset_suffix + '.target'
        with open(os.path.join(output_directory, output_source), 'w') as source_f, \
            open(os.path.join(output_directory, output_target), 'w') as target_f:
            for orgfileid in split_dict[dtype]:
                count += 1
                source_text, target_text = extract_text_cnndm(os.path.join(raw_dir, 'raw_stories', orgfileid), filter_level)
                filtered_summary = select_summary_sentences(nlp, source_text, target_text, filter_level)
                if not filtered_summary:
                    if write_inverse:
                        source_f.write(source_text + '\n')
                        target_f.write(target_text + '\n')
                        count_selected += 1
                    continue
                if not write_inverse:
                    source_f.write(source_text + '\n')
                    target_f.write(filtered_summary + '\n')
                    count_selected += 1
        print("{} selected out of {}".format(count_selected, count))


def preprocess_newsroom(filter_level=0, raw_dir='~/newsroom-data-dir', output_dir='~/newsroom-data-dir/processed-data',
                        write_inverse=False):
    '''
    preprocess newsroom dataset according to different filter levels.
    - filter_level=0: no special processing
    - filter_level=1: remove corruption text in source articles and summaries. (e.g. Some source articles contain only
        captions for photos; some bad summaries such as "Collection of all USATODAY.com coverage of People,
        including articles, videos, photos, and quotes.")
    - filter_level=2: entity hallucination filtering in addition to corruption text removal.
        A summary sentence is removed if it contains a named entity not in the source document.
    '''

    from newsroom import jsonl

    def source_bad(source, is_print=False):
        if len(source.split()) < 50:
            if is_print:
                print(source)
            return True
        # if source.startswith("'Image ") or \
        # source.startswith('"Photo: ') or \
        if (source.startswith('Image ') and source[6] in "0123456789") or \
            source.startswith("Photo: ") or \
            '"params":' in source :
            if is_print:
                print(source)
            return True
        return False


    def summary_bad(summary, is_print=False):
        if len(summary.split()) < 8:
            if is_print:
                print(summary)
            return True
        if re.search(re.escape('on FoxNews.com'), summary, re.IGNORECASE) or \
            re.search(re.escape('from FoxNews.com'), summary, re.IGNORECASE) or \
            re.search(re.escape('Collection of all USATODAY.com'), summary, re.IGNORECASE) or \
            re.search(re.escape('washingtonpost.com'), summary, re.IGNORECASE):
            if is_print:
                print(summary)
            return True
        return False

    nlp = spacy.load("en_core_web_lg")
    split_types = ['val', 'train', 'test']
    for split_type in split_types:
        if split_type == 'train':
            in_file = os.path.join(raw_dir, 'train.dataset')
        elif split_type == 'val':
            in_file = os.path.join(raw_dir, 'dev.dataset')
        elif split_type == 'test':
            in_file = os.path.join(raw_dir, 'test.dataset')
        else:
            print("ERROR! split_type must be one of the following: train, val and test!")
        count = 0
        output_source = split_type + '.source'
        output_target = split_type + '.target'
        num_lines = sum(1 for _ in jsonl.open(in_file, gzip=True))
        with open(os.path.join(output_dir, output_source), 'w') as source_f, \
            open(os.path.join(output_dir, output_target), 'w') as target_f:
            with jsonl.open(in_file, gzip=True) as f:
                for entry in tqdm(f, total=num_lines):
                    if entry['summary'] and entry['text']:
                        summary = " ".join(entry['summary'].split('\n'))
                        if filter_level > 0:
                            if summary_bad(summary) or source_bad(entry['text']):
                                if write_inverse and filter_level <= 1:
                                    source_f.write(entry['text'].strip().encode('unicode-escape').decode().replace('\\\\', '\\') + '\n')
                                    target_f.write(summary.strip() + '\n')
                                    # target_f.write(summary.strip().encode('unicode-escape').decode().replace('\\\\', '\\') + '\n')
                                    # source_f.write(repr(entry['text'].strip()) + '\n')
                                    # target_f.write(repr(entry['summary'].strip()) + '\n')
                                    count += 1
                                continue
                            filtered_summary = select_summary_sentences(nlp, entry['text'], summary, filter_level)
                            if not filtered_summary:
                                if write_inverse:
                                    source_f.write(entry['text'].strip().encode('unicode-escape').decode().replace('\\\\', '\\') + '\n')
                                    target_f.write(summary.strip() + '\n')
                                    # target_f.write(summary.strip().encode('unicode-escape').decode().replace('\\\\', '\\') + '\n')
                                    count += 1
                                continue
                        if not write_inverse:
                            source_f.write(
                                entry['text'].strip().encode('unicode-escape').decode().replace('\\\\', '\\') + '\n')
                            target_f.write(summary.strip() + '\n')
                            # target_f.write(
                            #     summary.strip().encode('unicode-escape').decode().replace('\\\\', '\\') + '\n')
                            count += 1
        print("Wrote {} lines in {}".format(count, os.path.join(output_dir, output_source)))


def preprocess_xsum(raw_dir="~/xsum-data-dir", dataset_suffix='',
                    filter_level=0, output_dir="~/xsum-data-dir/processed-data"):
    '''
    preprocess xsum dataset according to different filter levels.
    - filter_level=0: no special processing
    - filter_level=1: remove corruption text in source articles and summaries. (Undesirable texts included as a result
        of the imperfection in data collection. e.g. "Share this with Email, Facebook, Messenger...".)
    - filter_level=2: entity hallucination filtering in addition to corruption text removal.
        A summary sentence is removed if it contains a named entity not in the source document.
    '''
    split_dict = json.loads(open(os.path.join(raw_dir, "XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json")).read())
    data_types = ["test", "validation", "train"]
    output_directory = output_dir
    nlp = spacy.load("en_core_web_lg")

    for dtype in data_types:
        count = 0
        count_selected = 0
        print(dtype)
        if dtype == "validation":
            output_source = "val" + dataset_suffix + '.source'
            output_target = "val" + dataset_suffix + '.target'
        else:
            output_source = dtype + dataset_suffix + '.source'
            output_target = dtype + dataset_suffix + '.target'
        with open(os.path.join(output_directory, output_source), 'w') as source_f, \
            open(os.path.join(output_directory, output_target), 'w') as target_f:
            for orgfileid in split_dict[dtype]:
                count += 1
                source_text, target_text = extract_text_xsum(os.path.join(raw_dir, "xsum-extracts-from-downloads", orgfileid+'.data'), filter_level)
                if select_example(nlp, source_text, target_text, filter_level):
                    source_f.write(source_text + '\n')
                    target_f.write(target_text + '\n')
                    count_selected += 1
        print("{} selected out of {}".format(count_selected, count))


def _format_source_answers_bpe(bpe, source, answer, special_token, max_len=1024):
    source_bpe = bpe.encode(source)
    answer_bpe = bpe.encode(answer)
    assert len(answer_bpe) < max_len - 3
    return bpe.decode(source_bpe[:max_len-len(answer_bpe)-2]), \
           source_bpe[:max_len-len(answer_bpe)-2] + [special_token, ] + answer_bpe


def bpe_tokenize(tokenizer_dir, input_dir,
                 output_dir,
                 split_name=None, source_target=None,
                 no_val=False):
    # splits = ['train', 'val'] if split_name is None else [split_name]
    if split_name is None:
        if no_val:
            splits = ['train',]
        else:
            splits = ['train', 'val']
    else:
        splits = [split_name]

    if source_target is None:
        langs = ['source', 'target']
    else:
        langs = source_target.split(',')

    for split in splits:
        for lang in langs:
            cmd = "python multiprocessing_bpe_encoder.py " \
                  "--encoder-json {} " \
                  "--vocab-bpe {} " \
                  "--inputs {} " \
                  "--outputs {} " \
                  "--workers 16 " \
                  "--keep-empty".format(os.path.join(tokenizer_dir, "encoder.json"),
                                        os.path.join(tokenizer_dir, "vocab.bpe"),
                                        os.path.join(input_dir, split+'.'+lang),
                                        os.path.join(output_dir, split+'.bpe.'+lang))
            process = Popen(cmd.split(), stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()
            print(stderr.decode())
            print(stdout.decode())

def binarize_dataset(dict_path,
                     input_dir,
                     output_dir,
                     target_lang='target',
                     only_target=False,
                     no_val=False):
    cmd = "python ../fairseq_cli/preprocess.py " \
          "--source-lang source " \
          "--target-lang {} " \
          "--only-target {} " \
          "--trainpref {} " \
          "--destdir {} " \
          "--workers 1 " \
          "--srcdict {} " \
          "--tgtdict {}".format(target_lang,
                                only_target,
                                os.path.join(input_dir, "train.bpe"),
                                output_dir,
                                os.path.join(dict_path, 'dict.txt'),
                                os.path.join(dict_path, 'dict.txt'))
    if not no_val:
        cmd += " --validpref {}".format(os.path.join(input_dir, "val.bpe"))

    process = Popen(cmd.split(), stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stderr.decode())
    print(stdout.decode())


def binarize_cls_labels(
        dict_path,
        input_dir,
        output_dir):
    cmd = "python ../fairseq_cli/preprocess.py " \
          "--source-lang source " \
          "--target-lang target " \
          "--only-source True " \
          "--only-cls True " \
          "--cls-suffix cls_labels " \
          "--trainpref {} " \
          "--validpref {} " \
          "--testpref {} " \
          "--destdir {} " \
          "--workers 16 " \
          "--srcdict {} " \
          "--tgtdict {}".format(os.path.join(input_dir, "train"),
                                os.path.join(input_dir, "val"),
                                os.path.join(input_dir, "test"),
                                output_dir,
                                os.path.join(dict_path, 'dict.txt'),
                                os.path.join(dict_path, 'dict.txt'))
    process = Popen(cmd.split(), stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stderr)
    print(stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--mode', type=str, default="binarize_cls_labels")
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--no_val', default=False, action='store_true')
    parser.add_argument('--bpe_source', default=False, action='store_true')
    parser.add_argument('--source_target', type=str, default="source,target")
    parser.add_argument('--pattern', type=str, default="")
    parser.add_argument('--file_path', type=str, default="")
    parser.add_argument('--filter_level', type=int, default=0)
    parser.add_argument('--tokenizer_dir', type=str)


    args, unparsed = parser.parse_known_args()
    if args.mode == 'binarize_cls_labels':
        binarize_cls_labels(dict_path=args.tokenizer_dir, input_dir=args.input_dir, output_dir=args.output_dir)
    if args.mode == 'preprocess_newsroom':
        preprocess_newsroom(raw_dir=args.input_dir, filter_level=args.filter_level, output_dir=args.output_dir)
    if args.mode == 'preprocess_xsum':
        preprocess_xsum(raw_dir=args.input_dir, filter_level=args.filter_level, output_dir=args.output_dir)
    if args.mode == 'preprocess_cnndm':
        preprocess_cnndm(raw_dir=args.input_dir, filter_level=args.filter_level, output_dir=args.output_dir)
    if args.mode == 'bpe_binarize':
        # remember to remove '\r' characters using '%s/\r//g' in vim
        bpe_tokenize(tokenizer_dir=args.tokenizer_dir, input_dir=args.input_dir, output_dir=args.input_dir, no_val=args.no_val,
                     source_target=args.source_target)
        binarize_dataset(dict_path=args.tokenizer_dir, input_dir=args.input_dir, output_dir=os.path.join(args.input_dir, 'data_bin'),
                         no_val=args.no_val, target_lang=args.source_target.split(',')[1])
    if args.mode == 'binarize':
        binarize_dataset(dict_path=args.tokenizer_dir, input_dir=args.input_dir,
                         output_dir=os.path.join(args.input_dir, 'data_bin'),
                         only_target=True)
