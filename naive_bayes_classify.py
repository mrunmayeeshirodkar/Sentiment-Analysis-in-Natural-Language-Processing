# use this file to classify using naive-bayes classifier 
# Expected: generate nboutput.txt

import sys
import glob
import os
import collections
import json
import re
import string
import math

all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

test_by_class = collections.defaultdict(list)

stopwords = {"i", "me", "it","were", "its","she", "her", "hers", "herself", "itself","out", "him","be", "been","did","he", "his", "himself", "doing", "being", "her", "himself","have", "has", "had", "having", "am", "is", "are", "was", "herself", "my", "mine", "a", "an", "the", "and", "during", "before", "after", "above","do", "does", "further", "then" ,"i", "me", "my","very", "yourselves", "myself", "we", "our", "ours",  "yours", "yourself","once", "on", "off","over","because", "as",  "of", "under","ourselves", "you", "your","here", "there", "when", "where","at", "by", "for", "why", "how","more", "most", "other","from","what","again", "which", "who", "whom", "up", "down","this", "that", "these", "those", "in", "some", "all", "any","they", "them", "their", "theirs", "themselves", "get", "gets", "getting", "given", "through","so", "than", "too","into","with","until","about", "against", "between", "while","below","but", "if", "or","both",  "own", "same", "to","each", "few", "such", "room", "able", "hotel", "edu", "each", "during", "hotels", "across", "aside", "ask", "try", "trying", "once", "twice"}

ClassPositive = 0.0
ClassNegative = 0.0
ClassTruthful = 0.0
ClassDeceptive = 0.0

model_file = open("nbmodel.txt", "r")
model_json = json.load(model_file)
nboutput = open("nboutput.txt", "w")

Cond_positive = model_json['Cond_positive']
Cond_negative = model_json['Cond_negative']
Cond_truthful = model_json['Cond_truthful']
Cond_deceptive = model_json['Cond_deceptive']

for f in all_files:
    class1, class2, fold, fname = f.split('/')[-4:]
    test_by_class[class1 + class2].append(f)
    IpFile = open(f,"r")
    text = IpFile.read().lower()
    text = re.sub('[^a-zA-Z0-9]',' ',text)
    text = text.translate(None, string.digits)
    ClassPositive = 0.0
    ClassNegative = 0.0
    ClassTruthful = 0.0
    ClassDeceptive = 0.0
    
    wordlist = text.split()
    for w in wordlist:
        w = w.lstrip().rstrip()
        if (w not in stopwords) and (w != ""):
            if w in Cond_positive:
                ClassPositive += math.log(model_json['Cond_positive'][w])
            if w in Cond_negative:
                ClassNegative += math.log(model_json['Cond_negative'][w])
            if w in Cond_truthful:
                ClassTruthful += math.log(model_json['Cond_truthful'][w])
            if w in Cond_deceptive:
                ClassDeceptive += math.log(model_json['Cond_deceptive'][w])
            ClassPositive += math.log(model_json['Prior_Prob_Positive'])
            ClassNegative += math.log(model_json['Prior_Prob_Negative'])
            ClassTruthful += math.log(model_json['Prior_Prob_Truthful'])
            ClassDeceptive += math.log(model_json['Prior_Prob_Deceptive'])
            
            if ClassTruthful > ClassDeceptive:
                nboutput.write("truthful ")
            else:
                nboutput.write("deceptive ")
            if ClassPositive > ClassNegative:
                nboutput.write("positive ")
            else:
                nboutput.write("negative ")
            nboutput.write(f)
            nboutput.write("\n")
model_file.close()
nboutput.close()
                
if __name__ == "main":
    model_file = "nbmodel.txt"
    output_file = "nboutput.txt"
    input_path = str(sys.argv[0])