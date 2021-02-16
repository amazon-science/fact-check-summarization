# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import spacy
import math
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
import json
import re
from spacy.lang.en.stop_words import STOP_WORDS
import random

def entity_match(ent, source, level=2):
    if level == 0:
        # case sensitive match
        if ent in source:
            return [ent,]
        else:
            return []
    elif level == 1:
        # case insensitive match
        if re.search(re.escape(ent), source, re.IGNORECASE):
            return [ent,]
        else:
            return []
    elif level == 2:
        # split entity and match non-stop words
        ent_split = ent.split()
        result = []
        for l in range(len(ent_split), 1, -1):
            for start_i in range(len(ent_split) - l + 1):
                sub_ent = " ".join(ent_split[start_i:start_i+l])
                if re.search(re.escape(sub_ent), source, re.IGNORECASE):
                    result.append(sub_ent)
            if result:
                break
        if result:
            return result
        else:
            for token in ent_split:
                if token.lower() not in STOP_WORDS or token == "US":
                    if re.search(re.escape(token), source, re.IGNORECASE):
                        result.append(token)
            return result
    return []


def sub_or_swap_entity(summary, source, sub_candidates, constrained):
    for key, value in sub_candidates.items():
        random_draw_ent, ent_type = random.choice(value)
        ent_split = key.split()
        if not constrained:
            for l in range(len(ent_split), 1, -1):
                for start_i in range(len(ent_split) - l + 1):
                    sub_ent = " ".join(ent_split[start_i:start_i+l])
                    summary = re.sub(re.escape(sub_ent), random_draw_ent, summary, re.IGNORECASE)
                    source = re.sub(re.escape(sub_ent), random_draw_ent, source, re.IGNORECASE)
            for token in ent_split:
                if token.lower() not in STOP_WORDS or token == "US":
                    if ent_type == 'PERSON':  # replace only the last name:
                        random_draw_ent = random_draw_ent.split()[-1]
                    summary = re.sub(re.escape(token), random_draw_ent, summary, re.IGNORECASE)
                    source = re.sub(re.escape(token), random_draw_ent, source, re.IGNORECASE)
    return summary, source


def ner_count(nums, nlp, tokenizer, dataset, out_q):
    outdict = {}
    count_dict = {}
    for n in tqdm(nums):
        if n >= len(dataset):
            break
        example = dataset[n]
        summary = tokenizer.decode(example['summary'],
                                   clean_up_tokenization_spaces=True,
                                   skip_special_tokens=True)
        source = tokenizer.decode(example['source'],
                                  clean_up_tokenization_spaces=True,
                                  skip_special_tokens=True)
        doc = nlp(summary)
        en_count_in_summary = 0
        en_count_in_both = 0
        for e in doc.ents:
            if e[0].ent_type_ not in outdict:
                outdict[e[0].ent_type_] = set()
            outdict[e[0].ent_type_].add(e.text)
            en_count_in_summary += 1
            # if e.text in source:
            match_result = entity_match(e.text, source, 2)
            if match_result:
                en_count_in_both += 1
            else:
                print("ENTITY NOT FOUND in {}: {}".format(n,e.text))
                print(">>source>>", source)
                print(">>summary>>", summary)
        count_dict[n] = [en_count_in_summary, en_count_in_both]

    total_en_count_in_summary = 0
    total_en_count_in_both = 0
    count = 0
    for key, value in count_dict.items():
        total_en_count_in_summary += value[0]
        total_en_count_in_both += value[1]
        count += 1
    print("total examples={}, avg NE count in summary={},"
          " avg NE count in both={}".format(count,
                                            total_en_count_in_summary,
                                            total_en_count_in_both))
    for key, value in outdict.items():
        outdict[key] = list(value)

    out_q[nums[0]] = {'outdict': outdict, 'count': count,
                      'total_en_count_in_summary': total_en_count_in_summary,
                      'total_en_count_in_both': total_en_count_in_both}


def extract_entity_type_list(doc, tracking_entity_list):
    entity_d = {}
    for e in doc.ents:
        if e[0].ent_type_ in tracking_entity_list:
            if e[0].ent_type_ not in entity_d:
                entity_d[e[0].ent_type_] = [e.text, ]
            else:
                if not entity_match(e.text, " ".join(entity_d[e[0].ent_type_]), 2):
                    entity_d[e[0].ent_type_].append(e.text)
    return entity_d


def ner_sub_analysis(nums, nlp, dataset, out_q):
    tracking_entity_list = ['PERSON', 'FAC', 'GPE', 'ORG', 'NORP', 'LOC', 'EVENT']

    outdict = {}
    count_dict = {}
    for n in tqdm(nums):
        if n >= len(dataset):
            break
        example = dataset[n]
        summary = example['summary_text']
        source = example['source_text']
        doc_summary = nlp(summary)
        doc_source = nlp(source)
        source_entity_d = extract_entity_type_list(doc_source, tracking_entity_list)
        summary_entity_d = extract_entity_type_list(doc_summary, tracking_entity_list)

        count_num_ent_summary = 0
        count_num_ent_sub_not_available = 0
        count_num_ent_sub_available = 0
        sub_candidates = defaultdict(list)
        for k, v in summary_entity_d.items():
            count_num_ent_summary += len(v)
            if k in source_entity_d:
                for summary_ent in v:
                    found_sub = False
                    for source_ent in source_entity_d[k]:
                        if not entity_match(summary_ent, source_ent, 2):
                            found_sub = True
                            count_num_ent_sub_available += 1
                            sub_candidates[summary_ent].append(source_ent)
                    if not found_sub:
                        count_num_ent_sub_not_available += 1
            else:
                count_num_ent_sub_not_available += len(v)

        print(">>>ENTITY SUB FOUND in {}: {}".format(n, sub_candidates))
        print(">>source>>", source)
        print(">>summary>>", summary)
        count_dict[n] = [count_num_ent_summary, count_num_ent_sub_not_available, count_num_ent_sub_available]

    total_num_ent_summary = 0
    total_num_ent_sub_not_available = 0
    total_num_ent_sub_available = 0
    count = 0
    for key, value in count_dict.items():
        total_num_ent_summary += value[0]
        total_num_ent_sub_not_available += value[1]
        total_num_ent_sub_available += value[2]
        count += 1
    print("total examples={}, avg NE count in summary={},"
          " avg NE in summary with no substitution found={},"
          " avg valid NE substitutions per example={}".format(count,
                                                              total_num_ent_summary,
                                                              total_num_ent_sub_not_available,
                                                              total_num_ent_sub_available))
    for key, value in outdict.items():
        outdict[key] = list(value)

    out_q[nums[0]] = {'count': count,
                      'total_num_ent_summary': total_num_ent_summary,
                      'total_num_ent_sub_not_available': total_num_ent_sub_not_available,
                      'total_num_ent_sub_available': total_num_ent_sub_available}


if __name__ == '__main__':
    from src.lmdb_dataset import LMDBDataset

    data_path = ""
    tokenizer_path = ""
    dataset = LMDBDataset(data_path)
    length = len(dataset)
    nums = list(range(length))
    nprocs = 1

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

    # Load English tokenizer, tagger, parser, NER and word vectors
    nlp = spacy.load("en_core_web_sm")

    # Each process will get 'chunksize' nums and a queue to put his out
    # dict into
    manager = multiprocessing.Manager()
    chunksize = int(math.ceil(length / float(nprocs)))
    procs = []
    return_dict = manager.dict()

    for i in range(nprocs):
        p = multiprocessing.Process(
                target=ner_sub_analysis,
                args=(nums[chunksize * i:chunksize * i + 10], nlp, dataset, return_dict))
        procs.append(p)
        p.start()

    # Wait for all worker processes to finish
    for p in procs:
        p.join()
