from Retrieval import Model
from Query import Query
from StanfordDataset import StanfordDataset
from nltk.tokenize import word_tokenize
import csv
import math

def computeAccuracy(topic,sd = StanfordDataset()):
    
    testPara = sd.getParagraph(topic)
    
    drm = Model(testPara,True,True)
    
    result = []
    res = [[0,0],[0,0],[0,0],[0,0]]
    devData =sd.getTopic(topic)
    for index in range(0,len(devData['paragraphs'])):
        p = devData['paragraphs'][index]
        for qNo in range(0,len(p['qas'])):
            pq = Query(p['qas'][qNo]['question'],True,False,True)
            index = 0
            if pq.aType == 'PERSON':
                index = 0
            elif pq.aType == 'DATE':
                index = 1
            elif pq.aType == 'LOCATION':
                index = 2
            else:
                index = 3
            res[index][0] += 1
            r = drm.query(pq)
            answers = []
            for ans in p['qas'][qNo]['answers']:
                answers.append(ans['text'].lower())
            r = r.lower()
            isMatch = False
            for rt in word_tokenize(r):
                
                if [rt in word_tokenize(ans) for ans in answers].count(True) > 0:
                    isMatch = True
                    res[index][1] += 1
                    break

          
            result.append((index, qNo, pq.question, r, str(answers),isMatch))
                
    noOfResult = len(result)
    correct = [r[5] for r in result].count(True)
    if noOfResult == 0:
        accuracy = -1
    else:
        accuracy = correct/noOfResult

    return {"Topic":topic,"No of Ques":noOfResult,"Correct Retrieval":correct,"OverallAccuracy":round(accuracy*100,2)}

def runAll():
    sd = StanfordDataset()

    toCSV = []
    total = len(sd.titles)
    index = 1
    tA = 0
    for title in sd.titles:
        print("Testing all questions for \"" + title + "\"")
        d=computeAccuracy(title)
        if d["No of Ques"] == 0:
            continue
        tA += d['OverallAccuracy']
        print(d)
        print(str(index) + "/" + str(total) + ":",d['OverallAccuracy'],"/",tA/index)
        toCSV.append(d)
        index += 1
    print("OverallAccuracy : ",tA/total)

    keys = toCSV[0].keys()
    with open('accuracy.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(toCSV)

    print("Written the accuracy measure in accuracy.csv file. Done")


runAll()
