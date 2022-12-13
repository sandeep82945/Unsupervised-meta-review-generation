import numpy as np
import pandas as pd
from collections import Counter
import commentjson
import json
import csv
import gensim.downloader as api
import sys
import os
import nltk
from nltk import word_tokenize as tk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('brown')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import CountVectorizer 
from textblob import TextBlob
import torch
import torch.nn.functional as F
from tqdm import tqdm
import regex as re 
import argparse



assert len(sys.argv) > 1, "Please specify prepare configuration file!"
config_file = sys.argv[1]
with open(config_file, "r") as file:
    configs = commentjson.loads(file.read())

threshold_min_clusters = configs["threshold_min_clusters"]
threshold_deduplication = configs["threshold_deduplication"]
threshold_word_similarity = configs["threshold_word_similarity"]
threshold_sentiment = configs["threshold_sentiment"]
threshold_cluster_similarity = configs["threshold_cluster_similarity"]


# Initialize w2v
print("*****Load w2v ({}) matrix*****".format(configs["embedding"]))
w2v = api.load(configs["embedding"])
emb_dim = configs["embedding"][-3:]
is_exact = True if configs["is_exact"] == "True" else False


# Get all files
source_files = [os.path.join("data", configs["p_name"], f) for f in configs["files"]]
gold_file = os.path.join("data", configs["p_name"], configs["gold"])


def word_filter(words):
  words = tk(words.lower())
  # filtering out insignificant words/punctuations/text
  filter = ["and", "or", "you", "me", "are", "is", "the", "but", "because", "as", "when", "for", "in", "on", "upon", "since", "also", "with", "i"]
  filt_wrd = []
  for word in words:
    if (word not in stop_words) and (word not in filt_wrd) and (word not in filter) and not(re.search("\W",word)) and (len(word)>3):
      filt_wrd.append(word)
  return(filt_wrd)


# get embedding for filtered words
def get_emb(word):
  if word not in w2v.vocab:
    return torch.zeros(w2v.vectors.shape[1])
  return torch.tensor(w2v.vectors[w2v.vocab[word].index])

# get stack of embeddings
def emb_stack(word_list):
  embs = torch.stack([get_emb(word) for word in word_list], dim=0)
  return embs

# get mean of filtered words - centroid
def mean(embs):
  return torch.mean(embs, dim=0)

# cosine similarity
def cos(x1, x2, dim=0):
  return F.cosine_similarity(x1, x2, dim)



input_file = os.path.join("data", configs["p_name"], "AS_dic.json")
f0 = open(input_file,'r')
dic = json.load(f0)

substance = []
motivation = []
clarity = []
meaningful_comparison = []
originality = []
soundness = []
replicability = []


for keys in dic.keys():
    if dic[keys].get('substance') is not None:
        if(dic[keys]['substance'].get('positive') is not None):
            substance.extend(dic[keys]['substance']['positive'])
        if(dic[keys]['substance'].get('negative') is not None):
            substance.extend(dic[keys]['substance']['negative'])

    if dic[keys].get('motivation') is not None:
        if(dic[keys]['motivation'].get('positive') is not None):
            motivation.extend(dic[keys]['motivation']['positive'])
        if(dic[keys]['motivation'].get('negative') is not None):
            motivation.extend(dic[keys]['motivation']['negative'])
            
    if dic[keys].get('clarity') is not None:
        if(dic[keys]['clarity'].get('positive') is not None):
            clarity.extend(dic[keys]['clarity']['positive'])
        if(dic[keys]['clarity'].get('negative') is not None):
            clarity.extend(dic[keys]['clarity']['negative'])
    
    if dic[keys].get('meaningful-comparison') is not None:
        if(dic[keys]['meaningful-comparison'].get('positive') is not None):
            meaningful_comparison.extend(dic[keys]['meaningful-comparison']['positive'])
        if(dic[keys]['meaningful-comparison'].get('negative') is not None):
            meaningful_comparison.extend(dic[keys]['meaningful-comparison']['negative'])
    
    if dic[keys].get('originality') is not None:
        if(dic[keys]['originality'].get('positive') is not None):
            originality.extend(dic[keys]['originality']['positive'])
        if(dic[keys]['originality'].get('negative') is not None):
            originality.extend(dic[keys]['originality']['negative'])
            
    
    if dic[keys].get('soundness') is not None:
        if(dic[keys]['soundness'].get('positive') is not None):
            soundness.extend(dic[keys]['soundness']['positive'])
        if(dic[keys]['soundness'].get('negative') is not None):
            soundness.extend(dic[keys]['soundness']['negative'])
    
    
    if dic[keys].get('replicability') is not None:
        if(dic[keys]['replicability'].get('positive') is not None):
            replicability.extend(dic[keys]['replicability']['positive'])
        if(dic[keys]['replicability'].get('negative') is not None):
            replicability.extend(dic[keys]['replicability']['negative'])
            

sub = '. '.join(substance)
print("1")
mot = '. '.join(motivation)
print("2")
cla = '. '.join(clarity)
print("3")
cmp = '. '.join(meaningful_comparison)
print("4")
ori = '. '.join(originality)
print("5")
sou = '. '.join(soundness)
print("6")
rep = '. '.join(replicability) 
print("7")                        
                            
docs= [sub,mot,cla,cmp,ori,sou,rep]



class aspect_extract:
    def __init__(self,docs):
        self.docs = docs
    def get_tfidf(self,index):
        cv=CountVectorizer(stop_words='english', ngram_range=(index,index)) 
        #cv=CountVectorizer(ngram_range=(2,2)) 

        # this steps generates word counts for the words in your docs 
        word_count_vector=cv.fit_transform(self.docs)

        tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
        tfidf_transformer.fit(word_count_vector)
        # print idf values 
        df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"]) 
        # sort ascending 
        df_idf.sort_values(by=['idf_weights'])

        # count matrix 
        count_vector=cv.transform(docs) 

        # tf-idf scores 
        tf_idf_vector=tfidf_transformer.transform(count_vector)
        feature_names = cv.get_feature_names() 
        return tf_idf_vector,feature_names

    #get tfidf vector for first document


    def give_top_k(self,index,n,thres):
        tfidf,feature_names= self.get_tfidf(n)
        first_document_vector = tfidf[index]

        #print the scores 
        df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
        return list(df.sort_values(by=["tfidf"],ascending=False).index)[0:thres]

    def substract(self,test_list1,test_list2):
        res = [ ele for ele in test_list1 ]
        for a in test_list2:
            if a in test_list1:
                res.remove(a)
        return res

    def f(self,index,n,thres = 200):
        new_list = self.give_top_k(index,n,thres)
        for i in range(7):
            if(i!=index):
                new_list = self.substract(new_list, self.give_top_k(i,n,thres))
        return new_list





clusters = {}
asp_emb_dict = {}
words_dict = {}



asp = ["substance", "motivation", "clarity", "meaningful-comparison", "originality", "soundness", "replicability"]


# working on each aspect one by one
for i in range(len(asp)):

  aspect_extractor = aspect_extract(docs)
  word_list = aspect_extractor.f(i,1,150)

  #aspect does not have any suitable opinion
  if (len(word_list))==0:
    continue

  words_dict[asp[i]] = word_list
  asp_emb = [get_emb(word) for word in word_list]
  asp_emb_dict[asp[i]] = asp_emb

  centroid = mean(torch.stack(asp_emb, dim=0)) 
  clusters[asp[i]] = centroid



def asp_centroid_similarity(opinion, aspect, threshold=0.5):
  filt_cluster = word_filter(opinion)
  # print(filt_cluster)
  if len(filt_cluster)==0:
    return 0
  cos_sim = float((F.cosine_similarity(mean(emb_stack(filt_cluster)), clusters[aspect],dim=0)).tolist())
  # print(cos_sim)
  if cos_sim>threshold:
    return 1
  else:
    return 0



class SentimentAnalysis(object):
    """Class to get sentiment score based on analyzer."""

    def __init__(self, filename=os.path.join("data", configs["p_name"], "SentiWordNet_3.0.0.txt"), weighting='geometric'):
        """Initialize with filename and choice of weighting."""
        if weighting not in ('geometric', 'harmonic', 'average'):
            raise ValueError(
                'Allowed weighting options are geometric, harmonic, average')
        # parse file and build sentiwordnet dicts
        self.swn_pos = {'a': {}, 'v': {}, 'r': {}, 'n': {}}
        self.swn_all = {}
        self.build_swn(filename, weighting)

    def average(self, score_list):
        """Get arithmetic average of scores."""
        if(score_list):
            return sum(score_list) / float(len(score_list))
        else:
            return 0

    def geometric_weighted(self, score_list):
        """"Get geometric weighted sum of scores."""
        weighted_sum = 0
        num = 1
        for el in score_list:
            weighted_sum += (el * (1 / float(2**num)))
            num += 1
        return weighted_sum

    # another possible weighting instead of average
    def harmonic_weighted(self, score_list):
        """Get harmonic weighted sum of scores."""
        weighted_sum = 0
        num = 2
        for el in score_list:
            weighted_sum += (el * (1 / float(num)))
            num += 1
        return weighted_sum

    def build_swn(self, filename, weighting):
        """Build class's lookup based on SentiWordNet 3.0."""
        records = [line.split('\t') for line in open(filename)]
        # records[0] = ['a', '00001740', '0.125', '0', 'able#1', '(usually followed by `to\')]
        
        for rec in records:
            # has many words in 1 entry
            words = rec[4].split()
            pos = rec[0]
            for word_num in words:
                word = word_num.split('#')[0]
                sense_num = int(word_num.split('#')[1])

                # build a dictionary key'ed by sense number
                if word not in self.swn_pos[pos]:
                    self.swn_pos[pos][word] = {}
                self.swn_pos[pos][word][sense_num] = float(
                    rec[2]) - float(rec[3])
                if word not in self.swn_all:
                    self.swn_all[word] = {}
                self.swn_all[word][sense_num] = float(rec[2]) - float(rec[3])
        
#         with open('check.json','w') as f:
#             json.dump(self.swn_all, f, indent = 4)
#         in self.swn_pos
#          "a": {
#         "able": {
#             "1": 0.125,
#             "3": 0.125,
#             "2": 0.125,
#             "4": 0.25
#         },
        # convert innermost dicts to ordered lists of scores
    
        for pos in self.swn_pos.keys():
            for word in self.swn_pos[pos].keys():
                newlist = [self.swn_pos[pos][word][k] for k in sorted(
                    self.swn_pos[pos][word].keys())]
                if weighting == 'average':
                    self.swn_pos[pos][word] = self.average(newlist)
                if weighting == 'geometric':
                    self.swn_pos[pos][word] = self.geometric_weighted(newlist)
                if weighting == 'harmonic':
                    self.swn_pos[pos][word] = self.harmonic_weighted(newlist)

        for word in self.swn_all.keys():
            newlist = [self.swn_all[word][k] for k in sorted(
                self.swn_all[word].keys())]
            if weighting == 'average':
                self.swn_all[word] = self.average(newlist)
            if weighting == 'geometric':
                self.swn_all[word] = self.geometric_weighted(newlist)
            if weighting == 'harmonic':
                self.swn_all[word] = self.harmonic_weighted(newlist)

    def pos_short(self, pos):
        """Convert NLTK POS tags to SWN's POS tags."""
        if pos in set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
            return 'v'
        elif pos in set(['JJ', 'JJR', 'JJS']):
            return 'a'
        elif pos in set(['RB', 'RBR', 'RBS']):
            return 'r'
        elif pos in set(['NNS', 'NN', 'NNP', 'NNPS']):
            return 'n'
        else:
            return 'o' #was a before

    def score_word(self, word, pos):
        """Get sentiment score of word based on SWN and part of speech."""
        if(pos == 'v'):
            try:
                if(self.swn_pos[pos][word] !=0):
                    return word
            except KeyError:
                try:
                    if(self.swn_all[word] !=0):
                        return word
                except KeyError:
                    return ''
        else:
            return ''

    def return_word_score(self, word, pos):
        """Get sentiment score of word based on SWN and part of speech."""
        try:
            if(self.swn_pos[pos][word] !=0):
                return self.swn_pos[pos][word]
        except KeyError:
            try:
                if(self.swn_all[word] !=0):
                    return self.swn_all[word]
            except KeyError:
                return ''
    
    def score_sentence(self,sentence):
        tokens = nltk.tokenize.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        dic2 = {}
        for item in tagged:
            #if(self.pos_short(item[1]) in ['v','n','a']):
                #print(item[0],self.pos_short(item[1]))
            score = self.return_word_score(item[0], self.pos_short(item[1]))
            if(len(str(score))>0):
                dic2[item[0]] = score
        return dic2


t = SentimentAnalysis(filename=os.path.join("data", configs["p_name"], "SentiWordNet_3.0.0.txt"),weighting='geometric')
def senti_score(opinion, threshold=0.5):
  filt_cluster = word_filter(opinion)
  if len(filt_cluster)==0:
    return 0
  flag=0
  senti_dict = t.score_sentence((' ').join(filt_cluster))
  for score in senti_dict:
    if senti_dict[score]==None:
      senti_dict[score] = 0
    if float(senti_dict[score])>threshold:
      flag+=1
      print(float(senti_dict[score]))
  if flag>0:
    return 1
  else:
    return 0




for source_file in source_files:
  target_file = os.path.join("{}_{}_{}_{}_{}_{}_{}.csv".format(source_file[:-4], configs["num_review"], configs["top_k"], configs["attribute"], configs["sentiment"], emb_dim, int(10*configs["threshold"])))

  num_lines = sum(1 for _ in open(source_file, "r"))
  pbar = tqdm(total=num_lines)
  entities = []
  heads = ['eid', 'rid', 'review', 'input_text']
  with open(source_file, "r") as file:
      reader = csv.reader(file, delimiter=",")
      next(reader)
      for row in reader:
          entity = {}
          print("\n")
          eid = row[0]
          rid = row[1]
          exts = row[4].split("[SEP]")
          extractions = exts[2:]
          # for i in exts:
          #   ext = (" ").join(i.split(",")[0:2])
          #   extractions.append(ext)
          print(extractions)
          ext_fin = []

          if len(extractions)>threshold_min_clusters:
            ext = []
            ext_emb = []
            for i in extractions:
              filt_list = word_filter(i)
              if len(filt_list)!=0:
                ext.append(i)
                ext_emb.append(mean(emb_stack(filt_list)))
            # print(ext)
            # print(len(ext_emb))

            #1. deduplication
            new_ext=[ext[len(ext_emb)-1]]
            for i in range(0,len(ext_emb)-1):
              flag = 0
              for j in range(i+1,len(ext_emb)):
                # print(F.cosine_similarity(ext_emb[i], ext_emb[j], 0))
                if cos(ext_emb[i], ext_emb[j])>threshold_deduplication:
                  flag += 1
              if flag==0:
                new_ext.append(ext[i])
            # print(len(new_emb))
            print(new_ext)

            if len(new_ext)>threshold_min_clusters:
              #2. word by word similarity to centroid OR sentiment score analysis
              ext_fin = []
              aspect_category = (row[4].split("[SEP]")[0]).strip()
              for i in new_ext:
                for word in i.split(" "):
                  if asp_centroid_similarity(word, aspect_category, threshold_word_similarity) or senti_score(word, threshold_sentiment):
                    ext_fin.append(i)
                    break
              new_ext = ext_fin
              ext_fin = []
              print(new_ext)

            # #3. create a centroid of all the remaining clusters and then compare similarity of each cluster from that centroid
            # ext_fin = []
            # if len(new_ext)>0 and 1:
            #   centroid = mean(torch.stack([get_emb(word) for word in word_filter(((" ").join(new_ext)).lower())], dim=0))
            #   for i in new_ext:
            #     filt_list = word_filter(i)
            #     similarity = F.cosine_similarity((mean(emb_stack(filt_list))), centroid, 0)
            #     # print(similarity)
            #     if similarity>0.55:
            #       ext_fin.append(i)
            # else:
            #   ext_fin = new_ext

            if len(new_ext)>threshold_min_clusters:
              aspect_category = (row[4].split("[SEP]")[0]).strip()
              #4. similarity between opinion cluster and aspect category centroid 
              ext_fin = []
              for ext in new_ext:
                if asp_centroid_similarity(ext, aspect_category, threshold_cluster_similarity):
                  ext_fin.append(ext)
              new_ext = ext_fin
              ext_fin = []
              print(new_ext)

            ext_fin = new_ext

          else:
            ext_fin = extractions
            print(ext_fin)
          
          

          entity['eid'] = row[0]
          entity['rid'] = row[1]
          entity['review'] = row[2]
          inp_txt = (row[3].split(";")[0]).split(",")[-2:]
          inp_txt.extend(ext_fin)
          # print(inp_txt)
          entity['input_text'] = (" [SEP] ").join(inp_txt)
          # print(entity)
          entities.append(entity)
          print("\n")
          pbar.update(1)


  with open(target_file, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=heads)
    writer.writeheader()
    writer.writerows(entities)