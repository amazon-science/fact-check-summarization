#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT


from sagemaker.estimator import Estimator
from subprocess import Popen, PIPE
import sys
import argparse


def _write_screen_and_file(line, fptr):
    sys.stderr.write(line)

    fptr.write(line)
    fptr.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--datatype', type=str, required=True)
    parser.add_argument('--setting', type=str, required=True) # e.g.: lm-25
    parser.add_argument('--exp_type', type=str, default='local')

    args, unparsed = parser.parse_known_args()
    assert args.exp_type in ['local', 'sagemaker']

    job_name = "cnndm-unlike-{}".format(args.setting)
    job_name = job_name.replace('_', '-')

    hyperparameters = {
        "max-tokens": 1024,
        "task": "translation_with_unlikelihood",
        "lang-pairs": "source-target,source-untarget",
        "source-lang": "source",
        "target-lang": "target",
        "truncate-source": "True",
        "layernorm-embedding": "True",
        "share-all-embeddings": "True",
        "share-decoder-input-output-embed": "True",
        "reset-optimizer": "True",
        "reset-dataloader": "True",
        "reset-meters": "True",
        "required-batch-size-multiple": 1,
        "arch": "bart_large",
        "criterion": "label_smoothed_cross_entropy_with_unlikelihood",
        "label-smoothing": 0.1,
        "dropout": 0.1,
        "attention-dropout": 0.1,
        "weight-decay": 0.01,
        "optimizer": "adam",
        "adam-betas": "(0.9, 0.999)",
        "adam-eps": 1e-08,
        "clip-norm": 0.1,
        "lr-scheduler": "polynomial_decay",
        "lr": 3e-06,
        "max-epoch": 2,
        "warmup-updates": 0,
        "fp16": "True",
        "update-freq": 1,
        "skip-invalid-size-inputs-valid-test": "True",
        "num-workers": 4,
        "find-unused-parameters": "True",
        "log-format": "simple",
        "log-interval": 500,
        "disable-validation": "True",
        "no-last-checkpoints": "True",
        "use-hellinger-loss": 0,
        "valid-subset": 'train'
    }

    if args.exp_type == "local":
        train_instance_type = 'local'
        train_instance_count = 1

        train_path = "<data-bin location>"
        init_path = "<MLE finetuned BART checkpoint dir>"
        save_path = "<model output location>"

        ngpus = 4 # modify based on the number of GPUs on the local machine.
        cmd = ['python', 'train.py', ]
        cmd += ['--save_dir', save_path]
        cmd += ['--train', train_path]
        cmd += ['--pretrained_path', init_path]
        cmd += ['--ngpus', '{}'.format(ngpus)]
        for key, value in hyperparameters.items():
            key = key.replace('_', '-')
            cmd.append('--{}'.format(key))
            cmd.append(str(value))
        stdout_fptr = open(save_path + "/Job_0.stdout", 'wt', encoding='utf-8')
        process = Popen(cmd, stdout=PIPE,
                        stderr=open(save_path + "/Job_0.stderr", 'wt', encoding='utf-8'),
                        encoding='utf-8',
                        bufsize=0,
                        )
        while process.poll() is None:
            line = process.stdout.readline()
            _write_screen_and_file(line, stdout_fptr)
        line = process.stdout.read()

        # special log writing for job_idx == 0
        _write_screen_and_file(line, stdout_fptr)

        if process.returncode != 0:
            raise Exception('job 0 terminated with non-zero returncode')
    else:
        train_instance_type = 'ml.p3dn.24xlarge'
        train_instance_count = 1
        train_path = "<data-bin s3 location>"
        init_path = "<MLE finetuned BART checkpoint s3 dir>"
        image_name = "<docker-image-name>"
        output_path = "s3://path/to/output"
        role = "<sagemaker-execution-role>"
        estimator = Estimator(role=role,
                              train_instance_count=train_instance_count,
                              train_instance_type=train_instance_type,
                              train_volume_size=100,
                              image_name=image_name+':latest',
                              hyperparameters=hyperparameters,
                              base_job_name=job_name,
                              train_max_run=5 * 24 * 60 * 60,
                              output_path=output_path,
                              metric_definitions=[
                                  {'Name': 'train:loss', 'Regex': ' loss=([0-9\\.]+)'},
                              ],
                              )

        print("Start training")
        estimator.fit(
            inputs={
                "train": train_path,
                "init": init_path,
            },
            logs=True,
            wait=False,
        )