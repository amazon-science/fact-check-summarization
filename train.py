#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq_cli.train import cli_main
import argparse
import ast
import json
import os

def convert_args_dict_to_list(args_dict):
    args = []
    for key, value in args_dict.items():
        key = key.replace('_','-')
        args.append('--{}'.format(key))
        args.append(value)
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--save_dir', type=str, default="")
    parser.add_argument('--train', type=str, default="")
    parser.add_argument('--pretrained_path', type=str, default="")
    parser.add_argument('--ngpus', type=int, default=8)

    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument('--distributed_backend', type=str, default='NCCL', help='distributed backend (default: NCCL)')
    parser.add_argument('--hosts', type=str, default='["host0"]')
    parser.add_argument('--current_host', type=str, default="host0")

    args, unparsed = parser.parse_known_args()

    env_name_default_value = {
        "SM_CHANNEL_TRAIN": args.train,
        "SM_CHANNEL_INIT": args.pretrained_path,
        "SM_MODEL_DIR": args.save_dir,
        "SM_HOSTS": '["host0"]',
        "SM_CURRENT_HOST": "host0",
        "SM_NUM_GPUS": str(args.ngpus),
        "SM_NUM_CPUS": "16",
        "SM_CHANNEL_LABEL": "",
    }
    for env_name in env_name_default_value.keys():
        default_value = env_name_default_value[env_name]
        os.environ[env_name] = os.getenv(env_name, default_value)


    args_dict = vars(args)

    args_dict['save_dir'] = os.environ['SM_MODEL_DIR']
    args_dict['pretrained_path'] = os.environ['SM_CHANNEL_INIT']
    args_dict['train'] = os.environ['SM_CHANNEL_TRAIN']

    num_gpus = int(os.environ["SM_NUM_GPUS"])
    args_dict['hosts'] = ast.literal_eval(os.environ['SM_HOSTS'])
    args_dict['current_host'] = os.environ['SM_CURRENT_HOST']
    args_dict['distributed-world-size'] = str(len(args_dict['hosts']) * num_gpus)
    os.environ['WORLD_SIZE'] = str(len(args_dict['hosts']) * num_gpus)

    os.environ['RANK'] = str(args_dict['hosts'].index(args_dict['current_host']) * num_gpus)
    args_dict.pop('hosts', None)
    args_dict.pop('current_host', None)

    args_dict['restore-file'] = os.path.join(args_dict['pretrained_path'], 'pretrained_model.pt')
    args_dict.pop('pretrained_path', None)

    train_dir = args_dict['train']
    args_dict.pop('train', None)

    args_dict.pop('ngpus', None)

    try:
        prefix = '/opt/ml/'
        param_path = os.path.join(prefix, 'input/config/hyperparameters.json')
        # Read in any hyperparameters that the user passed with the training job
        with open(param_path, 'r') as tc:
            training_params = json.load(tc)

        for k,v in training_params.items():
            args_dict[k] = v
    except:
        print("hyperparameters.json not found! Probably running without Sagemaker!")
    training_args = [train_dir,] + convert_args_dict_to_list(args_dict) + unparsed

    cli_main(training_args)
