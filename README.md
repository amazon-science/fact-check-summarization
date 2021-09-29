# Improving Factual Consistency of Abstractive Text Summarization
We provide the code for the papers:
 1. ["Entity-level Factual Consistency of Abstractive Text Summarization"](https://www.aclweb.org/anthology/2021.eacl-main.235/), EACL 2021.
    - We provide a set of new metrics to quantify the entity-level factual consistency of generated summaries. We also provide code for the two methods in our paper:
        - JAENS: joint entity and summary generation, and
        - Summary-worthy entity classification with summarization (multi-task learning)
 2. ["Improving Factual Consistency of Abstractive Summarization via Question Answering"](https://arxiv.org/abs/2105.04623), ACL-IJCNLP 2021
    - *QUALS*, a new automatic metric for factual consistency.
    - *CONSEQ*, a new contrastive learning algorithm for Seq2seq models to optimize sequence level objectives such as *QUALS*.

Our code is based on the [fairseq](https://github.com/pytorch/fairseq) library and we added support for model training on [Sagemaker](https://aws.amazon.com/sagemaker/).

## Requirements and setup

- `python==3.6`: `conda create -n entity_fact python=3.6`
- `pytorch==1.4.0`: `pip install torch==1.4.0 torchvision==0.5.0`
- run `pip install --editable ./`
- install `file2rouge` following instructions [here](https://github.com/pltrdy/files2rouge)
- download `en_core_web_lg`: `python -m spacy download en_core_web_lg`

## [Entity-level Factual Consistency of Abstractive Text Summarization](https://www.aclweb.org/anthology/2021.eacl-main.235/)
### Data preprocessing:
We provide three options to preprocess summarization data through the `filter_level` option.
- `filter_level=0`: no special processing
- `filter_level=1`: remove corruption text in source articles and summaries. 
(Undesirable texts included as a result of imperfect data collection. e.g. "Share this with Email, Facebook, Messenger".
Undesirable summaries such as "Collection of all USATODAY.com coverage of People, including articles, videos, photos, and quotes.")
- `filter_level=2`: entity hallucination filtering in addition to corruption text removal. A summary sentence is removed if it contains a named entity not in the source document.

#### XSUM:
1. Follow the instructions [here](https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset) to download and extract text from HTML files and establish the `xsum-extracts-from-downloads` directory.
2. Let `<xsum-data-dir>` be the directory that contains the `xsum-extracts-from-downloads` directory and `XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json`.
3. Run `python preprocess/data_prepro_clean.py --mode preprocess_xsum --input_dir <xsum-data-dir> --output_dir <xsum-data-dir>/processed-data --filter_level 0` with `filter_level` set to 0, 1, or 2.

#### CNNDM:
1. Download and unzip the stories directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. Put all .story files in a directory `<cnndm-data-dir>/raw_stories`.
2. Download the url files `mapping_train.txt`, `mapping_test.txt` and `mapping_valid.txt` from [here](https://github.com/nlpyang/BertSum/tree/master/urls) to `<cnndm-data-dir>`.
3. Run `python preprocess/data_prepro_clean.py --mode preprocess_cnndm --input_dir <cnndm-data-dir> --output_dir <cnndm-data-dir>/processed-data --filter_level 0` with `filter_level` set to 0, 1, or 2.

#### NEWSROOM:
1. Download the datasets following instructions from [here](https://github.com/lil-lab/newsroom).
2. Run `python preprocess/data_prepro_clean.py --mode preprocess_newsroom --input_dir <newsroom-data-dir> --output_dir <newsroom-data-dir>/processed-data --filter_level 0` with `filter_level` set to 0, 1, or 2.

#### Tokenize and binarize the data:
Download bpe encoder.json, vocabulary and fairseq dictionary to a directory, say `<bpe-dir>`; then tokenize and binarize the data.
```bash
wget -O <bpe-dir>/encoder.json 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -O <bpe-dir>/vocab.bpe 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N <bpe-dir>/dict.txt' https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

cd preprocess
python data_prepro_clean.py --mode bpe_binarize --input_dir <processed-data-dir> --tokenizer_dir <bpe-dir>
```

This generates the binary input files as well as the dictionaries under `<processed-data-dir>/data_bin` for fairseq.

### JAENS: Joint Entity and Summary Generation
The idea is to train the seq2seq model to generate `<summary-worthy named entities> <sep> <abstractive summary>`. 
#### 1. prepare the train/val data by generating entity augmented targets:
```bash
python preprocess/create_entity_classification_labels.py --base_dir <processed-data-dir> --type entity_augment --tokenizer_dir <bpe-dir>
```
Binarize the augmented targets:
`cd preprocess`
```
python data_prepro_clean.py --mode binarize --input_dir <processed-data-dir>/entity_augment --tokenizer_dir <bpe-dir>
```
Since we already binarized the source documents, we just need to create symbolic links to put all binary input files together for fairseq training:
```bash
ln -s <processed-data-dir>/data_bin/train.source-target.source.idx <processed-data-dir>/entity_augment/data_bin/train.source-target.source.idx
ln -s <processed-data-dir>/data_bin/train.source-target.source.bin <processed-data-dir>/entity_augment/data_bin/train.source-target.source.bin
ln -s <processed-data-dir>/data_bin/valid.source-target.source.bin <processed-data-dir>/entity_augment/data_bin/valid.source-target.source.bin
ln -s <processed-data-dir>/data_bin/valid.source-target.source.idx <processed-data-dir>/entity_augment/data_bin/valid.source-target.source.idx
```
#### 2. Fine-tune the BART-large model on the generated data:
Run the launch scripts `scripts/launch_xsum.py`, `scripts/launch_cnndm.py` or `scripts/launch_newsroom.py` to fine-tune the BART-large model.
Note you need to modify the following in the scripts:
- `hyperparameters`.
- `train_path`: location of the binary input files. e.g. `<processed-data-dir>/entity_augment/data_bin`.
- `init_path`: location of the pre-trained BART-large model checkpoint. **Please rename the checkpoint to `pretrained_model.pt`**
- `output_path`: location for the model outputs.

If training locally, you need to specify `ngpus` - the number of GPUS in the local machine. Example command:
```
python scripts/launch_xsum.py --datatype ner_filtered --epoch 8 --exp_type local
```
If training on Sagemaker, you need to specify the docker image name (`image_name`) as well as execution role (`role`). 
To create Sagemaker docker container and push to ECR:
```
./build_and_push.sh <YOUR ECR REPO>
```
To launch training job:
```
python scripts/launch_xsum.py --datatype ner_filtered --epoch 8 --exp_type sagemaker
```
#### 3. Generate the summaries from the fine-tuned models:
`preprocess/multi_gpu_generate.py` is used to generate summaries. 

Since the JAENS models generates the named entities before the summaries, we need to remove the named entities before evaluating the summaries. Example command:
```
python evaluate_hypo.py --mode remove_ent_from_hypo --base_dir <output-dir> --sub_dir <output-sub-dir> --split val --pattern .*.hypo
```

#### 4. To evaluate the generated summaries for ROUGE as well as entity level factual scores:
We use the tokenizer from [Stanford CoreNLP package](https://stanfordnlp.github.io/CoreNLP/download.html). Example command:
```bash
export CLASSPATH=path/to/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
python evaluate_hypo.py --mode evaluate_summary --base_dir <output-dir> --sub_dir <output-sub-dir> --split val --pattern .*.hypo
```
See `preprocess/run_*_eval.sh` for examples.

### Summary-worthy entity classification with summarization (multi-task learning)
We perform summary-worthy entity classification at a classification head on the encoder while keeping the seq2seq objective at the decoder. 
For training, we need to preprocess the input document by create B-I-O labels to identify summary-worthy entities:
```
python create_entity_classification_labels.py --base_dir <processed-data-dir> --type cls_labels --tokenizer_dir <bpe-dir>
python data_prepro_clean.py --mode binarize_cls_labels --input_dir <processed-data-dir> --output_dir <processed-data-dir>/data_bin --tokenizer_dir <bpe-dir>
```
Launch training jobs using scripts `scripts/launch_multitask_*.py`.

## [Improving Factual Consistency of Abstractive Summarization via Question Answering](https://arxiv.org/abs/2105.04623)

### *QUALS* (QUestion Answering with Language model score for Summarization)
To evaluate the QUALS of summaries (e.g. test.target) given original input (e.g. test.source), we execute the following steps in the `preprocess` sub-directory.
#### 0. Prepare summaries into jsonl format
```
python evaluate_hypo.py --mode convert_hypo_to_json --base_dir <processed-data-dir> --sub_dir <any-sub-directory-to-data> --split test --pattern .target
```
 
#### 1. Generating question and answer pairs from summaries
```
python sm_inference_asum.py --task gen_qa --base_dir <processed-data-dir> --source_dir <any-sub-directory-to-data> --output_dir <output-dir> --num_workers <num-of-gpus> --bsz 5 --beam 60 --max_len 60 --min_len 8 --checkpoint_dir <QAGen-model-dir> --ckp_file checkpoint2.pt --bin_dir <processed-data-dir>/data_bin --diverse_beam_groups 60 --diverse_beam_strength 0.5 --batch_lines True --input_file test.target.hypo --return_token_scores True
```
Here, we use diverse beam search to generate 60 question-answer pairs for each summary. The `batch_lines` option is set to `True` to batch `bsz` input summaries together for efficient generation. The QAGen model is trained by fine-tuning BART on the [SQuAD](https://www.aclweb.org/anthology/D16-1264.pdf) and [NewsQA](https://github.com/Maluuba/newsqa) datasets by concatenating the question-answer pairs using a separator. 

To train the QAGen model, place the `dev-v1.1.json` and `train-v1.1.json` of SQuAD and the `combined-newsqa-data-v1.json` of the NewsQA under `<squad-newsqa-dir>`. The following command generates the binarized input for fine-tuning BART using Fairseq.
```
python data_prepro_clean.py --mode newsqa_squad_prepro --input_dir <squad-newsqa-dir> --output_dir <squad-newsqa-dir>
```
You can also download our trained QAGen model from s3 by running:
```
aws s3 cp s3://fact-check-summarization/newsqa-squad-qagen-checkpoint/checkpoint2.pt <QAGen-model-dir>/
```
Alternatively, you can download [here](https://fact-check-summarization.s3.amazonaws.com/newsqa-squad-qagen-checkpoint/checkpoint2.pt) if you don't have awscli.

#### 2. Filter the generated question and answer for high quality pairs
```
python evaluate_hypo.py --mode filter_qas_dataset_lm_score --base_dir <processed-data-dir> --sub_dir <any-sub-directory-to-qas> --pattern test.target.hypo.beam60.qas
```

#### 3. Evaluate the generated question and answer pairs using the source document as input
```
python sm_inference_asum.py --task qa_eval --base_dir <processed-data-dir> --output_dir <output-dir> --num_workers <num-of-gpus> --bsz 30 --checkpoint_dir <QAGen-model-dir> --ckp_file checkpoint2.pt --bin_dir <processed-data-dir>/data_bin --qas_dir <sub-directory-to-qas-filtered> --source_file test.source --target_file test.target --input_file test.target.qas_filtered --prepend_target False
```
#### 4. Compute QUALS scores for each summary
```
python evaluate_hypo.py --mode compute_hypos_lm_score --base_dir <processed-data-dir> --sub_dir <sub-directory-to-qas-filtered> --pattern test.*.source_eval_noprepend
```

### *CONSEQ* (CONtrastive SEQ2seq learning)
To use *QUALS* to improve factual consistency of the summarization model using the *CONSEQ* algorithm, we follow the steps:
1. Obtain the MLE summarization baseline by fine-tuning the BART model. Note that in the ACL paper, we used the [corruption-filtered](#data-preprocessing) CNNDM and XSUM datasets (`filter_level=1`).
2. Use the MLE summarization model to sample summaries on the training data.
3. Evaluate the *QUALS* for the generated summaries as well as the ground truth summaries of the training data.
4. Form the positive and negative sets for contrastive learning.
5. Fine-tune the MLE summarization model using the positive and negative examples. Example scripts for launching training jobs locally or on Sagemaker are  `preprocess/run_generate_unlikelihood_train_cnndm.sh` and  `preprocess/run_generate_unlikelihood_train_xsum.sh`.

We provide an example script `preprocess/run_generate_unlikelihood_train_xsum.sh` to illustrate steps 2-4.
Note that
- To avoid running for a long time and encountering OOM errors and then restarting the whole process, we split the input files into smaller ones. 
We do this by splitting the source file by line (e.g. each sub-file has 10000 lines):
```
split -l 10000 train.source train.source.split   
```
- The script has to make repeated calls of `python sm_inference_asum.py --task gen_qa` to generate question-ansewr pairs, as many times as there are sub-files as a result of line splits. 
The python function automatically checks which sub-files have been processed (based on output files) so it always processes the next available sub-file. 
If all sub-files have been processed, it will simply do nothing so it's safe if it's called more times than there are available sub-files.
- Similarly, `sm_inference_asum.py --task qa_eval` needs to be repeated called to cover all sub-files.
- The speed of question-answer pairs generation depends on the batch size setting. Depending on the summary file and the `batch_lines` setting, batching is handled differently.
If the summary file contains only a single summary per input document, `batch_lines` should be set to `True` and `bsz` number of input lines are batched together as input to the QAGen model.
If the summary file contains multiple summaries per input document, `batch_lines` should be set to `False` and the batching is done using `bsz` within each input example (line).
For example, if there are 6 summaries per line in the summary file, we should set `batch_lines` to `False`; setting `bsz` to 7 will batch all the 7 summaries in a line together, which gives the best speed. (setting it higher won't improve speed since we do not do batching over different lines of input as `batch_lines` is `False`).
On CNN-DM, `bsz` of 7 would sometimes result in OOM errors with 16G GPU memory so I use 3 or 4; on XSUM, it is safe to use 7.
- `num_workers` should be the number of GPUs available on the machine. The lines in each input files will be distributed per GPU.
- Finally, run the following to concatenate the QUALS scores from the sub-files: `cat *.quals > train.source.source_eval_noprepend.quals` 


## Citations
```
@inproceedings{nan-etal-2021-entity,
    title = "Entity-level Factual Consistency of Abstractive Text Summarization",
    author = "Nan, Feng  and
      Nallapati, Ramesh  and
      Wang, Zhiguo  and
      Nogueira dos Santos, Cicero  and
      Zhu, Henghui  and
      Zhang, Dejiao  and
      McKeown, Kathleen  and
      Xiang, Bing",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.eacl-main.235",
    pages = "2727--2733",
}

@inproceedings{nan-etal-2021-improving,
    title = {Improving Factual Consistency of Abstractive Summarization via Question Answering},
    author = "Nan, Feng  and
      Nogueira dos Santos, Cicero  and
      Zhu, Henghui  and
      Ng, Patrick  and
      McKeown, Kathleen  and
      Nallapati, Ramesh  and
      Zhang, Dejiao  and
      Wang, Zhiguo  and
      Arnold, Andrew  and
      Xiang, Bing",
    booktitle = {Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP)},
    address = {Virtual},
    month = {August},
    url = {https://arxiv.org/abs/2105.04623},
    year = {2021}
}
```