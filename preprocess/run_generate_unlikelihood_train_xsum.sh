#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

# split the training input into smaller sub files to avoid OOM errors:
split -l 10000 train.source train.source.split
split -l 10000 train.target train.target.split

# 0. Obtain the MLE summarization baseline by fine-tuning the BART model
# See README for standard BART fine-tuning

# 1. Use the current summarization model to sample summaries on the training data.
for i in {1..21}
do
  echo "Running $i times"
  python sm_inference_asum.py --task gen_summary --base_dir <processed-data-dir> --input_file train.source.split --num_workers <num-of-gpus> --bsz 30 --sampling True --sampling_topk 50 --beam 6 --max_len 60 --min_len 10 --checkpoint_dir <summarization-model-checkpoint-dir> --ckp_file checkpoint2.pt --bin_dir <processed-data-dir>/data_bin --output_dir <output-summaries-dir>
done

# 2. Generate question and answer pairs from summaries
for i in {1..21}
do
  echo "Running $i times"
  python sm_inference_asum.py --task gen_qa --base_dir <processed-data-dir> --source_dir <sub-dir-to-output-summaries> --output_dir <output-qas-dir> --num_workers <num-of-gpus> --bsz 10 --beam 60 --max_len 60 --min_len 8 --checkpoint_dir <QAGen-model-dir> --ckp_file checkpoint2.pt --bin_dir <processed-data-dir>/data_bin --diverse_beam_groups 60 --diverse_beam_strength 0.5 --input_file train.source.split*.hypo --return_token_scores True
done

# 3. Filter the generated question and answer for high quality pairs
python evaluate_hypo.py --mode filter_qas_dataset_lm_score --base_dir <processed-data-dir> --sub_dir <sub-dir-to-output-qas> --pattern train.target.split*.qas

# 4. Evaluate the generated question and answer pairs using the source document as input
for i in {1..21}
do
  echo "Running $i times"
  python sm_inference_asum.py --task qa_eval --base_dir <processed-data-dir> --output_dir <output-dir-to-qas-filtered> --num_workers <num-of-gpus> --bsz 60 --checkpoint_dir <QAGen-model-dir> --ckp_file checkpoint2.pt --bin_dir <processed-data-dir>/data_bin --qas_dir <sub-dir-to-output-qas> --source_file train.source.split* --target_file train.target.split* --input_file *.qas_filtered --prepend_target False
done

# 5. compute the lm scores for the ground truth training summaries
python evaluate_hypo.py --mode select_unlikelihood_hypos_lm_score --base_dir <processed-data-dir> --sub_dir <sub-dir-to-output-source_eval_noprepend> --pattern train.*.source_eval_noprepend

# In order to apply CONSEQ, we need to evaluate the QUALS of the ground truth summaries of the training set as well:

# 1a. Convert the ground truth target to jsonl format:
python evaluate_hypo.py --mode convert_hypo_to_json --base_dir <processed-data-dir> --sub_dir <sub-dir-to-train-target> --split train --pattern .target.split*

# 2a. Generate question and answer pairs from summaries
for i in {1..21}
do
  echo "Running $i times"
  python sm_inference_asum.py --task gen_qa --base_dir <processed-data-dir> --source_dir <sub-dir-to-train-target> --output_dir <output-qas-dir> --num_workers <num-of-gpus> --bsz 8 --beam 60 --max_len 60 --min_len 8 --checkpoint_dir <QAGen-model-dir> --ckp_file checkpoint2.pt --bin_dir <processed-data-dir>/data_bin --diverse_beam_groups 60 --diverse_beam_strength 0.5 --input_file train.target.split*.hypo --return_token_scores True --batch_lines True
done

# 3a. Filter the generated question and answer for high quality pairs
python evaluate_hypo.py --mode filter_qas_dataset_lm_score --base_dir <processed-data-dir> --sub_dir <sub-dir-to-output-qas> --pattern train.target.split*.qas

# 4a. Evaluate the generated question and answer pairs using the source document as input
for i in {1..21}
do
  echo "Running $i times"
  python sm_inference_asum.py --task qa_eval --base_dir <processed-data-dir> --output_dir <output-dir-to-qas-filtered> --num_workers <num-of-gpus> --bsz 60 --checkpoint_dir <QAGen-model-dir> --ckp_file checkpoint2.pt --bin_dir <processed-data-dir>/data_bin --qas_dir <sub-dir-to-output-qas> --source_file train.source.split* --target_file train.target.split* --input_file *.qas_filtered --prepend_target False
done

# 5a. compute the lm scores for the ground truth training summaries
python evaluate_hypo.py --mode select_unlikelihood_hypos_lm_score --base_dir <processed-data-dir> --sub_dir <sub-dir-to-output-source_eval_noprepend> --pattern train.*.source_eval_noprepend

# 6. make positive and negative training set for contrastive learning
ratio=0.3
targetRatio=0.3
ratio100=30
targetRatio100=30
type=lm
echo "$type-$ratio-$targetRatio"

# running the below command will create a sub-directory called $type-$ratio100-$targetRatio100 that contains the training data for contrastive learning (train.source for source documents; train.target for positive summaries and train.untarget for negative summaries.)
python evaluate_hypo.py --mode make_unlikelihood_dataset --base_dir <processed-data-dir> --sub_dir <sub-dir-to-qas-filtered> --pattern train.source.split*.source_eval_noprepend --unlike_select_ratio $ratio --score_type $type --target_select_ratio $targetRatio --target_index_file <path-to-untarget.index-of-ground-truth-summaries (result of step 5 above)> --metric_choice eval_ns-ns

# 7. Binarize the data for training
python data_prepro_clean.py --mode bpe_binarize --input_dir <path-to-qas-filtered>/"$type"-"$ratio100"-"$targetRatio100" --tokenizer_dir <bpe-dir> --no_val
python data_prepro_clean.py --mode binarize_untarget --input_dir <path-to-qas-filtered>/"$type"-"$ratio100"-"$targetRatio100" --tokenizer_dir <bpe-dir> --no_val

# 8. Re-use the binarized source inputs for positive and negative examples
ln -s <path-to-qas-filtered>/"$type"-"$ratio100"-"$targetRatio100"/data_bin/train.source-target.source.bin <path-to-qas-filtered>/"$type"-"$ratio100"-"$targetRatio100"/data_bin/train.source-untarget.source.bin
ln -s <path-to-qas-filtered>/"$type"-"$ratio100"-"$targetRatio100"/data_bin/train.source-target.source.idx <path-to-qas-filtered>/"$type"-"$ratio100"-"$targetRatio100"/data_bin/train.source-untarget.source.idx