import torch
import torch.nn as nn
import math
import os,sys
import json
import numpy as np

def compute_list_mean(x):
    return sum(x)/len(x)

def compute_list_and(x):
    return int(sum(x)==len(x))

def evaluate_mcs(files, thred, clip_size):
    res_output = []
    for file_ in files:
        res = dict()
        with open(file_, "r") as f:
            for l in f:
                l = l.strip().split(',')
                if len(l) != 5:
                    # print(l, " This line is not score.")
                    continue

                key = l[0] + ";" + l[1]
                frames = l[2]
                j_score, f_score = l[3], l[4]

                if key not in res:
                    tmp = {}
                    tmp['frames'] = []
                    tmp['j_score'] = []
                    tmp['f_score'] = []
                    res[key] = tmp

                res[key]['frames'].append(frames)
                res[key]['j_score'].append(float(j_score))
                res[key]['f_score'].append(float(f_score))

        # [optional?] sort keys based on frame name
        keys = list(res.keys())
        for key in keys:
            data = res[key]['frames']
            order_ = sorted(range(len(data)), key=lambda k: data[k])
            res[key]['frames'] = list(np.array(res[key]['frames'])[order_])
            res[key]['j_score'] = list(np.array(res[key]['j_score'])[order_])
            res[key]['f_score'] = list(np.array(res[key]['f_score'])[order_])

        score = 0.
        num_count = 0
        # for each object - [vid;oid]
        for key in keys:
            data = res[key]['frames']
            j_score = res[key]['j_score']
            # get good binary label
            j_score_binary = [int(x > thred) for x in j_score]
            # split into chunks and compute (clip_size=10000 means sequence level)
            j_score_chunked_binary = [j_score_binary[i:i + clip_size] for i in range(0, len(j_score_binary), clip_size)]
            # compute scores
            j_score_chunked_binary = [compute_list_and(x) for x in j_score_chunked_binary]
            # for each video clip
            for i_, x in enumerate(j_score_chunked_binary):
                score += x
                num_count += 1

        print("The MCS Score of {} under thred {} is: {}.".format(file_.split('/')[-1], thred, score*100/num_count))
        res_output.append(score*100/num_count)

    return res_output


if __name__ == "__main__":
    # TODO: Get consistency score based on the [stdout.txt] download from ytb eval server
    files = ["your path for /stdout.txt"]
    thred = 0.9
    clip_size = 10000  # sequence level consistency
    res_output = evaluate_mcs(files, thred, clip_size)  # get mcs under target thred
    print(["%.2f" % x for x in res_output])
    print('Done.')
