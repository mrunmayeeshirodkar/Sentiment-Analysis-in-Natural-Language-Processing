# use this file to learn perceptron classifier 
# Expected: generate vanillamodel.txt and averagemodel.txt

import sys
import glob
import os
import collections
import string
import re
import json
import random

feature_dict = dict()
weight_pos_neg = dict()
weight_tru_dec = dict()
aweight_pos_neg = dict()
aweight_tru_dec = dict()
cweight_pos_neg = dict()
cweight_tru_dec = dict()
label_pos_neg = dict()
label_tru_dec = dict()


all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))
train_by_class = collections.defaultdict(list)
stopwords = {'below', 'because', 's', 'isn', 'in', 'thats', 't', 'whos', 'wheres', 'but', 'why', 'how', 'doing', 'theyd', 'whom', 'your', 'arent', 'some', 'shouldnt', 'didnt', 'here', 'didn', 'mustnt', 'wourld', 'lets', 'have', 'hows', 'her', 'us', 'hes', 'hadnt', 'is', 'can', 'they', 'aren', 'shes', 'has', 'against', 'down', 'hell', 'its', 'through', 'hadnt', 'hes', 'of', 'so', 'shouldn', 'am', 'its', 'shan', 'don', 'well', 'youll', 'the', 'youd', 'id', 'won', 'those', 'no', 'ought', 'few', 'o', 'are', 'haven', 'ourselves', 'been', 'cant', 'such', 'again', 'needn', 'where', 'theres', 'it', 'on', 'than', 'm', 'nor', 'im', 'them', 'wheres', 'other', 'shant', 'yourself', 'dosen', 'between', 're', 'with', 'same', 'ive', 'theyve', 'does', 'weren', 'll', 'whys', 'youve', 'own', 'cant', 'he', 'could', 'that', 'myself', 'youre', 'we', 'most', 'this', 'herself', 'each', 'shed', 'not', 'shant', 'hers', 't', 'werent', 'was', 'lets', 'once', 'shes', 'did', 'himself', 'doesnt', 'ain', 'heres', 'into', 'then', 'an', 'whos', 'd', 'very', 'yours', 'ill', 'him', 'only', 'or', 'cannot', 've', 'youre', 'whats', 'a', 'to', 'you', 'hasn', 'off', 'dont', 'wouldn', 'by', 'll', 'up', 'mightn', 'hasnt', 'what', 'she', 'too', 'couldn', 'return', 'as', 'wasnt', 'weve', 'thats', 'our', '###', 'havent', 'i', 'if', 'during', 'being', 'hed', 'above', 'under', 'isnt', 'which', 'and', 'theyll', 'were', 'their', 'about', 'before', 'whens', 'm', 'more', 'wasn', 'wed', 'whats', 'yourselves', 'any', 'theirs', 'while', 'theres', 'wont', 'couldnt', 'isnt', 'were', 'all', 'after', 'for', 'be', 'when', 'his', 'couldnt', 'im', 'over', 'heres', 'hows', 'were', 'should', 'y', 'didnt', 'theyve', 'hadn', 'from', 'would', 'themselves', 'these', 'there', 'arent', 'mustn', 'my', 'out', 'youd', 'ours', 'whens', 'at', 'mustnt', 'had', 'ma', 'until', 'whys', 'do', 'having', 'who', 'me', 'youve', 'further', 'both', 'itself'}

for f in all_files:
    class1, class2, fold, fname = f.split('/')[-4:]
    IpFile = open(f,"r")
    train_by_class[f] = re.sub('[^a-zA-Z0-9]',' ',IpFile.read().lower().replace("-"," ").replace("'"," ")).translate(None, string.digits).split()
    
keys = train_by_class.keys()
random.shuffle(keys)

def runvanillaperceptron(feature_dict):
    bias_pos_neg = 0
    cbias_tru_dec = 0
    cbias_pos_neg = 0
    bias_tru_dec = 0
    c = 1
    for num in range(0, 25):
        for key in feature_dict:
            val1 = 0
            val2 = 0
            for word in feature_dict[key]:
                val1 += weight_tru_dec[word] * feature_dict[key][word]
                val2 += weight_pos_neg[word] * feature_dict[key][word]
            val1 += bias_tru_dec
            val2 += bias_pos_neg
            
            if val1 * label_tru_dec[key] <= 0:
                for word in feature_dict[key]:
                    weight_tru_dec[word] += feature_dict[key][word] * label_tru_dec[key] 
                bias_tru_dec += label_tru_dec[key]
            
            if val2 * label_pos_neg[key] <= 0:
                for word in feature_dict[key]:
                    weight_pos_neg[word] += feature_dict[key][word] * label_pos_neg[key] 
                bias_pos_neg += label_pos_neg[key]
    customDict = {"weight_tru_dec" : weight_tru_dec,
                  "bias_tru_dec" : bias_tru_dec,
                  "weight_pos_neg" : weight_pos_neg,
                  "bias_pos_neg" : bias_pos_neg,
                 "feature_dict": my_dict,
                 "my_dict": feature_dict}
    
    with open("vanillamodel.txt", "w") as file:
        file.write(json.dumps(customDict, indent = 2))
        
def runaveragedperceptron(feature_dict):
    bias_pos_neg = 0
    cbias_tru_dec = 0
    cbias_pos_neg = 0
    bias_tru_dec = 0
    c = 1
    for num in range(0, 25):
        for key in feature_dict:
            val1 = 0
            val2 = 0
            for word in feature_dict[key]:
                val1 += aweight_tru_dec[word] * feature_dict[key][word]
                val2 += aweight_pos_neg[word] * feature_dict[key][word]
            val1 += bias_tru_dec
            val2 += bias_pos_neg
            
            if val1 * label_tru_dec[key] <= 0:
                for word in feature_dict[key]:
                    aweight_tru_dec[word] += feature_dict[key][word] * label_tru_dec[key] 
                    cweight_tru_dec[word] += feature_dict[key][word] * label_tru_dec[key] * c
                bias_tru_dec += label_tru_dec[key]
                cbias_tru_dec += label_tru_dec[key] * c
            
            if val2 * label_pos_neg[key] <= 0:
                for word in feature_dict[key]:
                    aweight_pos_neg[word] += feature_dict[key][word] * label_pos_neg[key] 
                    cweight_pos_neg[word] += feature_dict[key][word] * label_pos_neg[key] * c
                bias_pos_neg += label_pos_neg[key]
                cbias_pos_neg += label_pos_neg[key] * c
            c += 1
            
    cbias_tru_dec = bias_tru_dec - cbias_tru_dec/float(c)
    cbias_pos_neg = bias_pos_neg - cbias_pos_neg/float(c)
    
    for word in weight_pos_neg:
        cweight_tru_dec[word] = aweight_tru_dec[word] - cweight_tru_dec[word]/float(c)
        cweight_pos_neg[word] = aweight_pos_neg[word] - cweight_pos_neg[word]/float(c)
    customDict = {"weight_tru_dec" : cweight_tru_dec,
                  "bias_tru_dec" : cbias_tru_dec,
                  "weight_pos_neg" : cweight_pos_neg,
                  "bias_pos_neg" : cbias_pos_neg,
                 "feature_dict": my_dict,
                 "my_dict": feature_dict}
    
    with open("averagedmodel.txt", "w") as file:
        file.write(json.dumps(customDict, indent = 2))
    

def createDictionaries(train_by_class):
    for key in train_by_class:
        word_list = dict()
        if "truthful" in key:
            label_tru_dec[key] = 1
        else:
            label_tru_dec[key] = -1
        if "positive" in key:
            label_pos_neg[key] = 1
        else:
            label_pos_neg[key] = -1
        for word in train_by_class[key]:
            if word not in stopwords:
                word_list[word] = word_list.get(word, 0) + 1
                my_dict[word] = my_dict.get(word, 0) + 1
                if word not in weight_pos_neg:
                    weight_pos_neg[word] = 0
                    aweight_pos_neg[word] = 0
                    aweight_tru_dec[word] = 0
                    cweight_pos_neg[word] = 0
                    weight_tru_dec[word] = 0
                    cweight_tru_dec[word] = 0
        feature_dict[key] = word_list
    return my_dict
my_dict = dict()
my_dict = createDictionaries(train_by_class)
runvanillaperceptron(feature_dict)
runaveragedperceptron(feature_dict)

if __name__ == "__main__":
    model_file = "vanillamodel.txt"
    avg_model_file = "averagemodel.txt"
    
    input_path = str(sys.argv[1])