# -*- coding: utf-8 -*-
# @Time    : 9/23/21 11:33 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : collect_summary.py

# collect summery of repeated experiment.

import argparse
import os
import numpy as np
import csv
from collections import defaultdict

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp-dir", type=str, default="./test", help="directory to dump experiments")
args = parser.parse_args()

csv_header = None
result = defaultdict(list)
# for each repeat experiment
for i in range(0, 10):
    cur_exp_dir = args.exp_dir + '/' + str(i)
    if os.path.isfile(cur_exp_dir + '/result.csv'):
        if csv_header is None:
            with open(cur_exp_dir + '/result.csv') as csv_fn:
                csv_header = ",".join(next(csv.reader(csv_fn)))
        try:
            print(cur_exp_dir)
            cur_res = np.loadtxt(cur_exp_dir + '/result.csv', delimiter=',', skiprows=1)
            for last_epoch in range(cur_res.shape[0] - 1, -1, -1):
                if cur_res[last_epoch, 0] < 5e-03: continue
                result[i].append(cur_res[last_epoch, :])
        except:
            pass

def write_result(result, exp_dir, csv_header, csv_fn='result_summary.csv'):
    result = np.array(result)
    print("result shape", result.shape)
    # get mean / std of the repeat experiments.
    result_mean = np.mean(result, axis=0)
    result_std = np.std(result, axis=0)
    result_max = np.max(result, axis=0)
    result_min = np.min(result, axis=0)

    if os.path.exists(exp_dir) == False:
        os.mkdir(exp_dir)
    np.savetxt(exp_dir + '/' + csv_fn, [result_mean, result_std, result_max, result_min], delimiter=',', header=csv_header, fmt='%.3f')

exp_dir = args.exp_dir
# last
result_list = [ result[i][0] for i in list(result.keys())]
write_result(result_list, exp_dir, csv_header, 'result_summary.csv')
# train best
result_list = [ min(result[i], key=lambda x: x[2]) for i in list(result.keys())]
write_result(result_list, exp_dir, csv_header, 'result_summary_tr_best.csv')
# test best
result_list = [ min(result[i], key=lambda x: x[4]) for i in list(result.keys())]
write_result(result_list, exp_dir, csv_header, 'result_summary_te_best.csv')
# Top 100 best iters in the 5 folds
result_list = [ r for i in list(result.keys()) for r in result[i]]
result_list = sorted(result_list, key=lambda x: x[2])
result_list = result_list[:100]
write_result(result_list, exp_dir, csv_header, 'result_summary_tr_b5fd.csv')