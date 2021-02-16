#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

# Generate summaries:
python multi_gpu_generate.py --base_dir <processed-data-dir> --output_dir <output-dir> --output_suffix ent_beam4 --checkpoint_dir <checkpoint_dir> --bin_dir <processed-data-dir>/entity_augment/data_bin --input_file val.source  --max_len 60 --min_len 10 --beam 6 --bsz 32 --checkpoint_num 12345678 --num_workers 4

# Remove the named entities from the generated summaries from the JAENS model
python evaluate_hypo.py --mode remove_ent_from_hypo --base_dir <output-dir> --sub_dir <output-sub-dir> --split val --pattern .*.hypo

# evaluate the ROUGE and entity scores for the generated summaries
python evaluate_hypo.py --mode evaluate_summary --base_dir <output-dir> --sub_dir <output-sub-dir> --split val --pattern .*.hypo
