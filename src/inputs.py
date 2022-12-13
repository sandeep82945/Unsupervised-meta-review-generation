# Copyright 2019 Megagon Labs, Inc. and the University of Edinburgh. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from collections import Counter
import commentjson
import csv
import gensim.downloader as api
import os
from nltk import word_tokenize as tokenizer
import random
from random import shuffle
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys


class Extraction:
    def __init__(self, opn, asp, att, pol, w2v, threshold=0.9):
        self.opn = opn
        self.asp = asp
        self.att = att
        self.pol = pol
        self.emb = self._get_emb(w2v)
        self.threshold = threshold
        self.is_valid = True

    def _get_emb(self, w2v):
        """ Get the avgerage w2v embedding """
        toks = tokenizer(self.opn.lower()) + tokenizer(self.asp.lower())
        self.is_valid = len(toks) > 0
        if not self.is_valid:
            return None
        embs = torch.stack([self._get_tok_emb(tok, w2v) for tok in toks], dim=0)
        return torch.mean(embs, dim=0)

    @staticmethod
    def _get_tok_emb(tok, w2v):
        """ Get the w2v embedding of a token """
        if tok not in w2v.vocab:
            return torch.zeros(w2v.vectors.shape[1])
        return torch.tensor(w2v.vectors[w2v.vocab[tok].index])


class Entity:
    def __init__(self, eid):
        self.eid = eid
        self.rids = []
        self.reviews = []
        self.exts = []

    def add_review(self, rid, review, exts, w2v, threshold):
        """ Add review into current entity
		Args:
			rid (int): review id
			review (str): review content
			exts (list(str)): extraction sequence
			w2v (glove2word2vec object): w2v object
			threhold (float): threshold for determine duplicate extraction
		"""
        self.rids.append(rid)
        self.reviews.append(review)
        cur_exts = []
        for ext in exts:
            if len(ext.strip()) < 1:
                continue
            opn, asp, att, pol = ext.split(",")
            ext_obj = Extraction(opn, asp, att, pol, w2v, threshold)
            if ext_obj.is_valid and ext_obj.emb is not None:
                cur_exts.append(ext_obj)
        self.exts.append(cur_exts)


class Input:
    def __init__(self, source_file, w2v, threshold):
        self.source_file = source_file
        self.threshold = threshold
        self.entities = self._read_file(source_file, w2v, threshold)

    def _read_file(self, source_file, w2v, threshold):
        """ Read reviews from file, compute w2v embedding for extractions.
		Args:
			source_file (str): source file path
			w2v (glove2word2vec object): w2v object
			threshold (float): threshold for determine duplicate extraction.
		"""
        num_lines = sum(1 for _ in open(source_file, "r"))
        # for _ in open(source_file, "r"):
        #   print("X ")
        print('\n',num_lines)
        pbar = tqdm(total=num_lines)
        entities = {}
        with open(source_file, "r") as file:
            reader = csv.reader(file, delimiter=",")
            next(reader)
            for row in reader:
                eid = row[0]
                rid = row[1]
                exts = row[3].split(";")
                #['brief i,seem,clarity,negative', 'derivation,think,clarity,negative']
                if eid not in entities:
                    entities[eid] = Entity(eid)
                entities[eid].add_review(rid, row[2], exts, w2v, threshold)
                # appends rid, review and extractions
                # review id, review text, extraction list
                print(exts)
              
                pbar.update(1)
        return list(entities.values())


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please specify prepare configuration file!"
    config_file = sys.argv[1]
    with open(config_file, "r") as file:
        configs = commentjson.loads(file.read())

    # Get all files
    source_files = [os.path.join("data", configs["p_name"], f) for f in configs["files"]]
    gold_file = os.path.join("data", configs["p_name"], configs["gold"])

    # Initialize w2v
    print("*****Load w2v ({}) matrix*****".format(configs["embedding"]))
    w2v = api.load(configs["embedding"])
    emb_dim = configs["embedding"][-3:]

    is_exact = True if configs["is_exact"] == "True" else False

    for source_file in source_files:
        print("*****Processing {}*****".format(source_file))
        inputs = Input(source_file, w2v, configs["threshold"])

        # print(inputs.entities)
        # print("*****Generating {}*****".format(source_file))
        # inputs.select(configs["num_review"], configs["top_k"], configs["attribute"], configs["sentiment"], is_exact,
        #               emb_dim, gold_file)
