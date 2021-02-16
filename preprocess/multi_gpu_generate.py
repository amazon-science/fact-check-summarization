# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import os
import sys
import argparse

from pathos.multiprocessing import ProcessPool
from evaluate_hypo import count_lines_in_text_file
from tqdm import tqdm
import glob

"""
Run locally with multiple-GPUs

Stop job when an error occurs in any subprocesses
"""


def _write_screen_and_file(line, fptr):
    print(line, flush=True, end='')
    sys.stderr.write(line)

    fptr.write(line)
    fptr.flush()


def _run_process(job_idx, *, args, ckp_file, output_prefix, offsets):
    from fairseq.models.bart import BARTModel
    import torch

    bart = BARTModel.from_pretrained(
        args.checkpoint_dir,
        checkpoint_file=ckp_file,
        data_name_or_path=args.bin_dir
    )
    torch.cuda.set_device(torch.device("cuda:{}".format(job_idx)))
    bart.cuda()
    bart.eval()
    bart.half()

    count = 1
    # bsz = 32
    bsz = args.bsz
    offset = offsets[job_idx]
    end = offsets[job_idx + 1]
    input_file = os.path.join(args.base_dir, args.input_file)
    out_text_file = os.path.join(args.output_dir, "{}.hypo{}".format(output_prefix, job_idx))
    print("Local worker is processing {}-{}".format(offset, end))
    with open(input_file, 'r') as f, \
            open(out_text_file, 'w') as out_text_f:
        for _ in range(offset):
            f.readline()
        line = f.readline()
        slines = [line]
        while line:
            if offset + count >= end:
                break
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch, score_batch, _ = bart.sample(slines,
                                                                beam=args.beam,
                                                                lenpen=args.lenpen,
                                                                max_len_b=args.max_len,
                                                                min_len=args.min_len,
                                                                sampling=args.sampling,
                                                                sampling_topk=args.sampling_topk,
                                                                sampling_topp=args.sampling_topp,
                                                                )
                for hypothesis in hypotheses_batch:
                    out_text_f.write(hypothesis + '\n')
                out_text_f.flush()
                slines = []
            line = f.readline()
            slines.append(line)
            count += 1
        if slines != []:
            with torch.no_grad():
                hypotheses_batch, score_batch, _ = bart.sample(slines,
                                                            beam=args.beam,
                                                            lenpen=args.lenpen,
                                                            max_len_b=args.max_len,
                                                            min_len=args.min_len,
                                                            sampling=args.sampling,
                                                            sampling_topk=args.sampling_topk,
                                                            sampling_topp=args.sampling_topp,
                                                            )
            for hypothesis in hypotheses_batch:
                out_text_f.write(hypothesis + '\n')
            out_text_f.flush()
        assert offset + count == end, "!worker ended at {}, should have been {}".format(
            offset + count,
            end
        )
    del bart
    torch.cuda.empty_cache()


def concat_temp_files(args, final_file_name):
    temp_files = glob.glob(os.path.join(args.output_dir, final_file_name+'*'))
    if len(temp_files) == 0:
        print("No temporary file found in {} that matches {}".format(args.output_dir, final_file_name+'*'))
    elif len(temp_files) == 1:
        print("Only one temporary file: {}. Renaming it now!".format(temp_files[0]))
        os.rename(temp_files[0], os.path.join(args.output_dir, final_file_name))
    else:
        with open(os.path.join(args.output_dir, final_file_name), 'w') as f:
            for worker_id in range(0, args.num_workers):
                temp_file_path = os.path.join(args.output_dir, final_file_name+str(worker_id))
                print("Concatenating ", temp_file_path)
                with open(temp_file_path, 'r') as f_in:
                    for line in f_in:
                        f.write(line)
                os.remove(temp_file_path)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=str, help='location of test.source')
    parser.add_argument('--input_file', type=str, default='test.source')
    parser.add_argument('--output_dir', type=str, help='location to put generated hypothesis')
    parser.add_argument('--output_suffix', type=str, default='unfiltered')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, help='location of checkpoint')
    parser.add_argument('--checkpoint_num', type=str, default="12345678", help='checkpoint numbers')
    # parser.add_argument('--ckp_file', type=str, help='name of checkpoint')
    parser.add_argument('--bin_dir', type=str, help='path of the binary data')

    parser.add_argument('--bsz', type=int, default=32, help='batch size for generation')
    parser.add_argument('--max_len', type=int, default=140, help='max length for generation')
    parser.add_argument('--min_len', type=int, default=55, help='min length for generation')
    parser.add_argument('--beam', type=int, default=1, help='beam size')
    parser.add_argument('--lenpen', type=float, default=1.0, help='length penalty')
    parser.add_argument('--sampling', type=bool, default=False, help='use sampling')
    parser.add_argument('--sampling_topk', type=int, default=-1, help='sampling_topk, -1 to disable')
    parser.add_argument('--sampling_topp', type=int, default=-1.0, help='sampling_topp, -1.0 to disable')


    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir

        if os.path.exists(args.output_dir):
            print('directory already exists:', args.output_dir)
        else:
            print('output_dir created:', output_dir)
            os.mkdir(args.output_dir)
    else:
        raise Exception('Please specify the output directory')

    assert args.input_file[-6:] == 'source'

    already_completed = []
    for file in glob.glob(os.path.join(args.output_dir, '*.hypo')):
        filename = os.path.basename(file)
        already_completed.append(filename[:-len('.' + args.output_suffix + '.hypo')])

    source_full_path = os.path.join(args.base_dir, args.input_file)
    n_lines = count_lines_in_text_file(source_full_path)
    print("Processing {} lines in {}".format(n_lines, source_full_path))
    step = n_lines // args.num_workers
    offsets = [i * step for i in range(args.num_workers)]
    offsets.append(n_lines)

    for ckp in tqdm(glob.glob(os.path.join(args.checkpoint_dir, 'checkpoint[' + args.checkpoint_num + ']*.pt'))):
        ckp_file = os.path.basename(ckp)
        if args.input_file[:-6] + 'ckp' + ckp_file[10:-3] in already_completed:
            print("Skipping {}: already done!".format(ckp_file))
            continue
        output_prefix = args.input_file[:-6] + 'ckp' + ckp_file[10:-3] + '.' + args.output_suffix

        with ProcessPool(ncpus=args.num_workers) as pool:
            process_func = lambda job_idx: _run_process(
                job_idx,
                args=args,
                ckp_file=ckp_file,
                output_prefix=output_prefix,
                offsets=offsets
            )
            pool_results = pool.uimap(process_func, list(range(args.num_workers)))
            for res in pool_results:
                print('Done with process {}'.format(res))
        concat_temp_files(args, output_prefix + '.hypo')
        print('Written to {}'.format(output_prefix + '.hypo'))
        pool.close()
        pool.clear()

if __name__ == '__main__':
    main()
