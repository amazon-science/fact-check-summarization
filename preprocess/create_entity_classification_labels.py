# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import spacy
from evaluate_hypo import count_lines_in_text_file
from tqdm import tqdm
from data_prepro_clean import TRACKING_ENTITY_LIST
import os, re
from types import SimpleNamespace
from ner import entity_match
import argparse
from fairseq.data.encoders.gpt2_bpe import get_encoder


def subfinder(mylist, pattern, first_only=False):
    matches_indx = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches_indx.append((i, i+len(pattern)))
            if first_only:
                break
    return matches_indx


def update_labels(labels, matched_ranges):
    for range_i in matched_ranges:
        if labels[range_i[0]] == 0:
            labels[range_i[0]] = 1  # [B]eginning
            for i in range(range_i[0]+1, range_i[1]):
                labels[i] = 2  # [I]nside
    return labels


def update_bio_labels(labels, source, match_patterns, tokens, encoder, first_only=False):
    for pattern in match_patterns:
        found = re.findall(re.escape(pattern), source, re.IGNORECASE)
        for matched in found:
            matched_ids = encoder.encode(matched if matched[0] == ' ' else ' ' + matched)
            matched_ranges = subfinder(tokens, matched_ids, first_only)
            labels = update_labels(labels, matched_ranges)
    return labels


def create_ent_labels(source_file, target_file, out_file, tokenizer_dir, first_only=False):
    n_s = count_lines_in_text_file(source_file)
    n_t = count_lines_in_text_file(target_file)
    assert n_s == n_t, \
        "Number of lines not consistent: {}, {}".format(n_s, n_t)

    nlp = spacy.load("en_core_web_lg")
    entities_found = []

    encoder_args = SimpleNamespace(encoder_json=os.path.join(tokenizer_dir, "encoder.json"),
                                   vocab_bpe=os.path.join(tokenizer_dir, "vocab.bpe"),
                                   keep_empty=True)
    bpe = get_encoder(encoder_args.encoder_json, encoder_args.vocab_bpe)

    with open(source_file, 'r') as s_f, \
        open(target_file, 'r') as t_f, \
        open(out_file, 'w') as out_f:

        for _ in tqdm(range(n_s)):
            sline = s_f.readline().strip()
            tline = t_f.readline().strip()
            tokens = bpe.encode(sline)
            labels = [0] * len(tokens)

            doc = nlp(tline)
            entities_per_example = []
            for e in doc.ents:
                if e[0].ent_type_ in TRACKING_ENTITY_LIST:
                    entity_new = {'text': e.text, 'type': e[0].ent_type_}
                    # if e.text in source:
                    match_result = entity_match(e.text, sline, 2)
                    entity_new['match_result'] = match_result
                    labels = update_bio_labels(labels, sline, match_result, tokens, bpe, first_only=first_only)
                    entities_per_example.append(entity_new)
            out_f.write(" ".join([str(i) for i in labels]) + '\n')
            entities_found.append(entities_per_example)
    return entities_found


def create_ent_augmented_target(source_file, target_file, out_text_file, out_bpe_file, tokenizer_dir,
                                special_token=50009, max_len=1024):
    n_s = count_lines_in_text_file(source_file)
    n_t = count_lines_in_text_file(target_file)
    assert n_s == n_t, \
        "Number of lines not consistent: {}, {}".format(n_s, n_t)

    nlp = spacy.load("en_core_web_lg")

    encoder_args = SimpleNamespace(encoder_json=os.path.join(tokenizer_dir, "encoder.json"),
                                   vocab_bpe=os.path.join(tokenizer_dir, "vocab.bpe"),
                                   keep_empty=True)
    bpe = get_encoder(encoder_args.encoder_json, encoder_args.vocab_bpe)

    with open(source_file, 'r') as s_f, \
        open(target_file, 'r') as t_f, \
        open(out_bpe_file, 'w') as out_bpe_f, \
        open(out_text_file, 'w') as out_text_f:

        for _ in tqdm(range(n_s)):
            sline = s_f.readline().strip()
            tline = t_f.readline().strip()

            doc = nlp(tline)
            entities_per_example = []
            for e in doc.ents:
                if e[0].ent_type_ in TRACKING_ENTITY_LIST:
                    # if e.text in source:
                    match_result = entity_match(e.text, sline, 2)
                    if match_result:
                        entities_per_example.append(match_result[0])
            target_bpe = bpe.encode(tline)
            if entities_per_example:
                entity_bpe = bpe.encode(", ".join(entities_per_example))
                augmented_target_bpe = entity_bpe + [special_token, ] + target_bpe
            else:
                augmented_target_bpe = [special_token, ] + target_bpe
            out_text_f.write("{}".format(entities_per_example) + '\n')
            out_bpe_f.write(' '.join(map(str, augmented_target_bpe[:max_len-1])) + '\n')



def extract_ent_from_labels(tokens, labels):
    spans = []
    flag = False
    for l_i in range(len(labels)):
        if labels[l_i] == 1:
            flag = True
            length = 1
            while l_i + length < len(tokens) and labels[l_i+length] == 2:
                length += 1
            spans.append(tokens[l_i:l_i+length])
        elif labels[l_i] == 0 and flag:
            flag = False
        elif labels[l_i] == 2 and not flag:
            raise ValueError("Error! Wrong labels at {}: {}".format(l_i, labels))
    return spans


def sanity_check(entities, source_bpe_file, label_file, eval_file, tokenizer_dir):
    n_s = count_lines_in_text_file(source_bpe_file)
    n_l = count_lines_in_text_file(label_file)

    assert n_s == n_l == len(entities), \
        "Number of lines not consistent: {}, {}, {}, {}".format(n_s, n_l, len(entities))
    encoder_args = SimpleNamespace(encoder_json=os.path.join(tokenizer_dir, "encoder.json"),
                                   vocab_bpe=os.path.join(tokenizer_dir, "vocab.bpe"),
                                   keep_empty=True)
    bpe = get_encoder(encoder_args.encoder_json, encoder_args.vocab_bpe)

    with open(source_bpe_file, 'r') as s_f, \
        open(label_file, 'r') as l_f, \
        open(eval_file, 'w') as o_f:
        for i in tqdm(range(n_l)):
            sline = s_f.readline().strip()
            tokens = [int(t) for t in sline.split()]
            lline = l_f.readline().strip()
            labels = [int(t) for t in lline.split()]
            assert len(tokens) == len(labels), "Number of source tokens must equal that of labels!"
            entities_per_example = entities[i]
            ent_text = ""
            for e in entities_per_example:
                ent_text += e['text']
                ent_text += str(e['match_result'])
                ent_text += ", "
            spans = extract_ent_from_labels(tokens, labels)
            ent_text += "FROM LABELS==>"
            for span in spans:
                ent_text += bpe.decode(span).strip()
                ent_text += ', '
            ent_text += '\n'
            o_f.write(ent_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--tokenizer_dir', type=str, required=True)
    args, unparsed = parser.parse_known_args()

    assert args.type in ['cls_labels', 'entity_augment']
    base_dir = args.base_dir
    for split in ['val', 'train']:
        print("Creating labels for ", split)
        source_file = split + ".source"
        target_file = split + ".target"
        source_bpe_file = split + ".bpe.source"
        if args.type == 'cls_labels':
            out_file = split + ".cls_labels"
            entities_found = create_ent_labels(
                source_file=os.path.join(base_dir, source_file),
                target_file=os.path.join(base_dir, target_file),
                out_file=os.path.join(base_dir, out_file),
                tokenizer_dir=args.tokenizer_dir,
                first_only=False
            )
            check_file = split + ".ent_check"
            print("Labels created for {}. Doing sanity check now!".format(split))
            sanity_check(
                entities=entities_found,
                source_bpe_file=os.path.join(base_dir, source_bpe_file),
                label_file=os.path.join(base_dir, out_file),
                eval_file=os.path.join(base_dir, check_file),
                tokenizer_dir=args.tokenizer_dir
            )
        elif args.type == 'entity_augment':
            out_dir = os.path.join(base_dir, 'entity_augment')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_text_file = os.path.join(out_dir, split + '.ent_augment')
            out_bpe_file = os.path.join(out_dir, split + '.bpe.target')
            create_ent_augmented_target(source_file=os.path.join(base_dir, source_file),
                                        target_file=os.path.join(base_dir, target_file),
                                        out_text_file=out_text_file,
                                        out_bpe_file=out_bpe_file,
                                        tokenizer_dir=args.tokenizer_dir)
