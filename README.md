# Entity-level Factual Consistency of Abstractive Text Summarization
We provide the code for the paper ["Entity-level Factual Consistency of Abstractive Text Summarization"](https://arxiv.org/abs/2102.09130), by Feng Nan, Ramesh Nallapati, Zhiguo Wang, Cicero Nogueira dos Santos, Henghui Zhu, Dejiao Zhang, Kathleen McKeown and Bing Xiang, accepted to EACL 2021.

In this repo, we provide a set of new metrics to quantify the entity-level factual consistency of generated summaries. We also provide code for the two methods in our paper:
- JAENS: joint entity and summary generation, and
- Summary-worthy entity classification with summarization (multi-task learning)

Our code is based on the [fairseq](https://github.com/pytorch/fairseq) library and we added support for model training on [Sagemaker](https://aws.amazon.com/sagemaker/).

## Requirements and setup

- `python==3.6`: `conda create -n entity_fact python=3.6`
- `pytorch==1.4.0`: `pip install torch==1.4.0 torchvision==0.5.0`
- run `pip install --editable ./`
- install `file2rouge` following instructions [here](https://github.com/pltrdy/files2rouge)
- download `en_core_web_lg`: `python -m spacy download en_core_web_lg`

## Data preprocessing:
We provide three options to preprocess summarization data through the `filter_level` option.
- `filter_level=0`: no special processing
- `filter_level=1`: remove corruption text in source articles and summaries. 
(Undesirable texts included as a result of imperfect data collection. e.g. "Share this with Email, Facebook, Messenger".
Undesirable summaries such as "Collection of all USATODAY.com coverage of People, including articles, videos, photos, and quotes.")
- `filter_level=2`: entity hallucination filtering in addition to corruption text removal. A summary sentence is removed if it contains a named entity not in the source document.

### XSUM:
1. Follow the instructions [here](https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset) to download and extract text from HTML files and establish the `xsum-extracts-from-downloads` directory.
2. Let `<xsum-data-dir>` be the directory that contains the `xsum-extracts-from-downloads` directory and `XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json`.
3. Run `python preprocess/data_prepro_clean.py --mode preprocess_xsum --input_dir <xsum-data-dir> --output_dir <xsum-data-dir>/processed-data --filter_level 0` with `filter_level` set to 0, 1, or 2.

### CNNDM:
1. Download and unzip the stories directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. Put all .story files in a directory `<cnndm-data-dir>/raw_stories`.
2. Download the url files `mapping_train.txt`, `mapping_test.txt` and `mapping_valid.txt` from [here](https://github.com/nlpyang/BertSum/tree/master/urls) to `<cnndm-data-dir>`.
3. Run `python preprocess/data_prepro_clean.py --mode preprocess_cnndm --input_dir <cnndm-data-dir> --output_dir <cnndm-data-dir>/processed-data --filter_level 0` with `filter_level` set to 0, 1, or 2.

### NEWSROOM:
1. Download the datasets following instructions from [here](https://github.com/lil-lab/newsroom).
2. Run `python preprocess/data_prepro_clean.py --mode preprocess_newsroom --input_dir <newsroom-data-dir> --output_dir <newsroom-data-dir>/processed-data --filter_level 0` with `filter_level` set to 0, 1, or 2.

### Tokenize and binarize the data:
Download bpe encoder.json, vocabulary and fairseq dictionary to a directory, say `<bpe-dir>`; then tokenize and binarize the data.
```bash
wget -O <bpe-dir>/encoder.json 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -O <bpe-dir>/vocab.bpe 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N <bpe-dir>/dict.txt' https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

cd preprocess
python data_prepro_clean.py --mode bpe_binarize --input_dir <processed-data-dir> --tokenizer_dir <bpe-dir>
```

This generates the binary input files as well as the dictionaries under `<processed-data-dir>/data_bin` for fairseq.

## JAENS: Joint Entity and Summary Generation
The idea is to train the seq2seq model to generate `<summary-worthy named entities> <sep> <abstractive summary>`. 
### 1. prepare the train/val data by generating entity augmented targets:
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
### 2. Fine-tune the BART-large model on the generated data:
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
### 3. Generate the summaries from the fine-tuned models:
`preprocess/multi_gpu_generate.py` is used to generate summaries. 

Since the JAENS models generates the named entities before the summaries, we need to remove the named entities before evaluating the summaries. Example command:
```
python evaluate_hypo.py --mode remove_ent_from_hypo --base_dir <output-dir> --sub_dir <output-sub-dir> --split val --pattern .*.hypo
```

### 4. To evaluate the generated summaries for ROUGE as well as entity level factual scores:
We use the tokenizer from [Stanford CoreNLP package](https://stanfordnlp.github.io/CoreNLP/download.html). Example command:
```bash
export CLASSPATH=path/to/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
python evaluate_hypo.py --mode evaluate_summary --base_dir <output-dir> --sub_dir <output-sub-dir> --split val --pattern .*.hypo
```
See `preprocess/run_*_eval.sh` for examples.

## Summary-worthy entity classification with summarization (multi-task learning)
We perform summary-worthy entity classification at a classification head on the encoder while keeping the seq2seq objective at the decoder. 
For training, we need to preprocess the input document by create B-I-O labels to identify summary-worthy entities:
```
python create_entity_classification_labels.py --base_dir <processed-data-dir> --type cls_labels --tokenizer_dir <bpe-dir>
python data_prepro_clean.py --mode binarize_cls_labels --input_dir <processed-data-dir> --output_dir <processed-data-dir>/data_bin --tokenizer_dir <bpe-dir>
```
Launch training jobs using scripts `scripts/launch_multitask_*.py`.

## Citation
```angular2
@inproceedings{nan21eacl,
    title = {Entity-level Factual Consistency of Abstractive Text Summarization},
    author = {Feng Nan and Ramesh Nallapati and Zhiguo Wang and Cicero Nogueira dos Santos and Henghui Zhu and Dejiao Zhang and Kathleen McKeown and Bing Xiang},
    booktitle = {Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
    address = {Online},
    month = {April},
    url = {https://arxiv.org/abs/2102.09130},
    year = {2021}
}
```