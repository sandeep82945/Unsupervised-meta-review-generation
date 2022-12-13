import os
import sys
import json
import commentjson
from rouge import Rouge
import pandas as pd
import math


assert len(sys.argv) > 1, "Please specify prepare configuration file!"
config_file = sys.argv[1]
with open(config_file, "r") as file:
  configs = commentjson.loads(file.read())

gold_path = os.path.join("data", configs["p_name"], "golden_new.json")
pred_path = os.path.join("output", "default_op2text_default_default_beam-5-06-3-60_agg.csv")
target_path = os.path.join("data", configs["p_name"], "evaluation.csv")

with open(gold_path,"r") as f: 
  dic2 = json.load(f)

df = pd.read_csv(pred_path)
pred = df['pred']
eid = df['eid']

dic = {}
for i in range(len(list(df['eid']))):
  dic[eid[i].split('-')[0]] =''

for i in range(len(list(df['eid']))):
  dic[eid[i].split('-')[0]] = str(dic[eid[i].split('-')[0]]) + " "+ str(pred[i])

for i in range(len(list(df['eid']))):
  dic[eid[i].split('-')[0]] = dic[eid[i].split('-')[0]].replace(" ..",".")

for key in dic.keys():
  dic2[key]['pred'] = dic[key]

#print prediction.dic
with open(os.path.join("data", configs["p_name"], "prediction_dic.json"),'w') as outfile:
  json.dump(dic2,outfile,indent=4)

hyps, refs, ids = map(list, zip(*[[dic2[key]['pred'], dic2[key]['golden'],key] for key in dic2 if 'pred' in dic2[key].keys()]))
rouge = Rouge()
#scores = rouge.get_scores(hyps, refs)
# or
scores = rouge.get_scores(hyps, refs, avg=True)

#The rouge score is :-
print(scores)

# for i in scores:
#   print(scores[i]['f'])

class my_dictionary(dict): 
    # __init__ function 
    def __init__(self): 
        self = dict() 
    # Function to add key:value 
    def add(self, key, value): 
        self[key] = value 

score_dic = my_dictionary()

score_dic.add('threshold_min_clusters', [configs["threshold_min_clusters"]])
score_dic.add('threshold_deduplication', [configs["threshold_deduplication"]])
score_dic.add('threshold_word_similarity', [configs["threshold_word_similarity"]])
score_dic.add('threshold_sentiment', [configs["threshold_sentiment"]])
score_dic.add('threshold_cluster_similarity', [configs["threshold_cluster_similarity"]])

for i in scores:
  score_dic.add(str(i), [scores[i]['f']])

# print(score_dic)

df2 = pd.DataFrame(data=score_dic, index=None)
# print(df2)

with open(target_path, 'a', newline='') as f:
  df2.to_csv(f, index=False, header=f.tell()==0)

# df2.to_csv(target_path, index=False, mode='a', header=f.tell()==0, encoding='utf-8')