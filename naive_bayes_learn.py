import sys
import glob
import os
import collections
import string
import re
import json

all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

train_by_class = collections.defaultdict(list)
stopwords = {"i", "me", "it","were", "its","she", "her", "hers", "herself", "itself","out", "him","be", "been","did","he", "his", "himself", "doing", "being", "her", "himself","have", "has", "had", "having", "am", "is", "are", "was", "herself", "my", "mine", "a", "an", "the", "and", "during", "before", "after", "above","do", "does", "further", "then" ,"i", "me", "my","very", "yourselves", "myself", "we", "our", "ours",  "yours", "yourself","once", "on", "off","over","because", "as",  "of", "under","ourselves", "you", "your","here", "there", "when", "where","at", "by", "for", "why", "how","more", "most", "other","from","what","again", "which", "who", "whom", "up", "down","this", "that", "these", "those", "in", "some", "all", "any","they", "them", "their", "theirs", "themselves", "get", "gets", "getting", "given", "through","so", "than", "too","into","with","until","about", "against", "between", "while","below","but", "if", "or","both",  "own", "same", "to","each", "few", "such", "room", "able", "hotel", "edu", "each", "during", "hotels", "across", "aside", "ask", "try", "trying", "once", "twice"}

#Variables 

positive = dict()
negative = dict()
deceptive = dict()
truthful = dict()

numPositive = 0.0
numNegative = 0.0
numDeceptive = 0.0
numTruthful = 0.0

numPosDoc = 0.0
numNegDoc = 0.0
numDecDoc = 0.0
numTruDoc = 0.0

PProbPositive = 0.0
PProbNegative = 0.0
PProbDeceptive = 0.0
PProbTruthful = 0.0

#class1 - deceptive and truthful
#class2 - positive and negative

for f in all_files:
    class1, class2, fold, fname = f.split('/')[-4:]
    train_by_class[class1 + class2].append(f)
    IpFile = open(f,"r")
    text = IpFile.read().lower()
    text = re.sub('[^a-zA-Z0-9]',' ',text)
    text = text.translate(None, string.digits)
    wordlist = text.split()
    if "positive" in f:
        numPosDoc += 1.0
    elif "negative" in f:
        numNegDoc += 1.0
    if "truthful" in f:
        numTruDoc += 1.0
    elif "deceptive" in f:
        numDecDoc += 1.0
    for w in wordlist:
        w = w.lstrip().rstrip()
        if (w not in stopwords) and (w != ""):
            if w not in positive:
                positive[w] = 1.0
            if w not in negative:
                negative[w] = 1.0
            if w not in truthful:
                truthful[w] = 1.0
            if w not in deceptive:
                deceptive[w] = 1.0
            if "positive" in f:
                numPositive += 1.0
                positive[w] += 1.0
            elif "negative" in f:
                numNegative += 1.0
                negative[w] += 1.0
            if "truthful" in f:
                numTruthful += 1.0
                truthful[w] += 1.0
            elif "deceptive" in f:
                numDeceptive += 1.0
                deceptive[w] += 1.0

for k in positive:
    positive[k] = positive[k] / (numPositive + float(len(positive)))
for k in negative:
    negative[k] = negative[k] / (numNegative + float(len(negative)))
for k in truthful:
    truthful[k] = truthful[k] / (numTruthful + float(len(truthful)))
for k in deceptive:
    deceptive[k] = deceptive[k] / (numDeceptive + float(len(deceptive)))

PProbPositive = numPosDoc / (numPosDoc + numNegDoc)
PProbNegative = numNegDoc / (numPosDoc + numNegDoc)
PProbTruthful = numTruDoc / (numTruDoc + numDecDoc)
PProbDeceptive = numDecDoc / (numTruDoc + numDecDoc)

customDict = {"Prior_Prob_Positive" : PProbPositive,
              "Prior_Prob_Negative" : PProbNegative,
              "Prior_Prob_Truthful" : PProbTruthful,
              "Prior_Prob_Deceptive" : PProbDeceptive,
              "Cond_positive" : positive,
              "Cond_negative" : negative,
              "Cond_truthful" : truthful,
              "Cond_deceptive" : deceptive}

with open("nbmodel.txt", "w") as file:
    file.write(json.dumps(customDict, indent = 2))

if __name__ == "main":
    model_file = "nbmodel.txt"
    input_path = str(sys.argv[0])