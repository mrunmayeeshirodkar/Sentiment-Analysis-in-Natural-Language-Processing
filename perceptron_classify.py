# use this file to classify using perceptron classifier 
# Expected: generate percepoutput.txt

import sys
import glob
import os
import collections
import json
import re
import string

all_files = glob.glob(os.path.join(sys.argv[2], '*/*/*/*.txt'))

test_by_class = collections.defaultdict(list)

stopwords = {'below', 'because', 's', 'isn', 'in', 'thats', 't', 'whos', 'wheres', 'but', 'why', 'how', 'doing', 'theyd', 'whom', 'your', 'arent', 'some', 'shouldnt', 'didnt', 'here', 'didn', 'mustnt', 'wourld', 'lets', 'have', 'hows', 'her', 'us', 'hes', 'hadnt', 'is', 'can', 'they', 'aren', 'shes', 'has', 'against', 'down', 'hell', 'its', 'through', 'hadnt', 'hes', 'of', 'so', 'shouldn', 'am', 'its', 'shan', 'don', 'well', 'youll', 'the', 'youd', 'id', 'won', 'those', 'no', 'ought', 'few', 'o', 'are', 'haven', 'ourselves', 'been', 'cant', 'such', 'again', 'needn', 'where', 'theres', 'it', 'on', 'than', 'm', 'nor', 'im', 'them', 'wheres', 'other', 'shant', 'yourself', 'dosen', 'between', 're', 'with', 'same', 'ive', 'theyve', 'does', 'weren', 'll', 'whys', 'youve', 'own', 'cant', 'he', 'could', 'that', 'myself', 'youre', 'we', 'most', 'this', 'herself', 'each', 'shed', 'not', 'shant', 'hers', 't', 'werent', 'was', 'lets', 'once', 'shes', 'did', 'himself', 'doesnt', 'ain', 'heres', 'into', 'then', 'an', 'whos', 'd', 'very', 'yours', 'ill', 'him', 'only', 'or', 'cannot', 've', 'youre', 'whats', 'a', 'to', 'you', 'hasn', 'off', 'dont', 'wouldn', 'by', 'll', 'up', 'mightn', 'hasnt', 'what', 'she', 'too', 'couldn', 'return', 'as', 'wasnt', 'weve', 'thats', 'our', '###', 'havent', 'i', 'if', 'during', 'being', 'hed', 'above', 'under', 'isnt', 'which', 'and', 'theyll', 'were', 'their', 'about', 'before', 'whens', 'm', 'more', 'wasn', 'wed', 'whats', 'yourselves', 'any', 'theirs', 'while', 'theres', 'wont', 'couldnt', 'isnt', 'were', 'all', 'after', 'for', 'be', 'when', 'his', 'couldnt', 'im', 'over', 'heres', 'hows', 'were', 'should', 'y', 'didnt', 'theyve', 'hadn', 'from', 'would', 'themselves', 'these', 'there', 'arent', 'mustn', 'my', 'out', 'youd', 'ours', 'whens', 'at', 'mustnt', 'had', 'ma', 'until', 'whys', 'do', 'having', 'who', 'me', 'youve', 'further', 'both', 'itself'}

for f in all_files:
    #class1, class2, fold, fname = f.split('/')[-4:]
    IpFile = open(f,"r")
    #print(f)
    test_by_class[f] = re.sub('[^a-zA-Z0-9]',' ',IpFile.read().lower().replace("-"," ").replace("'"," ")).translate(None, string.digits).split()

model_file = open(str(sys.argv[1]), "r")
model_json = json.load(model_file)
percepoutput = open("percepoutput.txt", "w")

weight_tru_dec = model_json['weight_tru_dec']
weight_pos_neg = model_json['weight_pos_neg']
bias_tru_dec = model_json['bias_tru_dec']
bias_pos_neg = model_json['bias_pos_neg']
my_dict = model_json['my_dict']

for key in test_by_class:
    val1 = 0
    val2 = 0
    for word in test_by_class[key]:
        if word in weight_tru_dec:
            if word in my_dict:
                val1 += weight_tru_dec[word] * my_dict[word]
            else:
                val1 += weight_tru_dec[word]
        if word in weight_pos_neg:
            if word in my_dict:
                val2 += weight_pos_neg[word] * my_dict[word]
            else:
                val2 += weight_pos_neg[word]
    val1 += bias_tru_dec
    val2 += bias_pos_neg
    if val1 > 0:
        percepoutput.write("truthful ")
    else:
        percepoutput.write("deceptive ")
    if val2 > 0:
        percepoutput.write("positive " + key + "\n")
    else:
        percepoutput.write("negative " + key + "\n")
        
percepoutput.close()
    
if __name__ == "__main__":
    model_file = str(sys.argv[1])
    output_file = "percepoutput.txt"
    input_path = str(sys.argv[2])