import os
import shutil
import json

import numpy as np
from tqdm import tqdm
import pandas as pd


BASE_DIR = "./data_val/scans" # input dir to read json files
BASE_OUTPUT_DIR = "./data_val_preprocessed/scans" # output to save gt labels

labels_pd = pd.read_csv("scannetv2-labels.combined.tsv", sep="\t", header=0 )

scans = []
with open(f"scannetv2_val.txt", 'r') as fp:
    line = fp.readline().strip()
    while line:
        scans.append(line)
        line = fp.readline().strip()

for i, scan in enumerate(scans):

    print(f"Processing {scan} ({i+1}/{len(scans)})")

    with open(os.path.join(BASE_DIR, scan, f"{scan}_vh_clean.aggregation.json"), 'r') as fp:
        agg = json.load(fp)["segGroups"]

    with open(os.path.join(BASE_DIR, scan, f"{scan}_vh_clean_2.0.010000.segs.json"), 'r') as fp:
        segs = json.load(fp)["segIndices"]

    gt_inst = np.zeros(len(segs), dtype=np.int32)

    for instance in agg:

        segments = instance["segments"]
        label = instance["label"]
        instance_id = int(instance["id"])

        label_id = labels_pd[labels_pd["raw_category"] == label]["id"].iloc[0]

        occupid_indices = np.isin(segs, segments)

        gt_inst[occupid_indices] = label_id * 1000 + instance_id + 1 # note that there are 607 raw category, of which only 200 is used for benchmarking

    np.save(os.path.join(BASE_OUTPUT_DIR, scan, f"instance_gt.npy"), gt_inst)
    