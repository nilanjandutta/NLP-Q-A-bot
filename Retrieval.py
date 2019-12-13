from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.tree import Tree
from nltk import pos_tag,ne_chunk
from Time import Date
import json
import math
import re

class Model:
    def __init__(self,paragraphs,removeStopWord = False,useStemmer = False):
        self.idf = {}
        self.paragraphInfo = {}
        self.paragraphs = paragraphs
        self.totalParas = len(paragraphs)
        self.stopwords = stopwords.words('english')
        self.removeStopWord = removeStopWord
        self.useStemmer = useStemmer
        self.vData = None
        self.stem = lambda k:k.lower()
        if(useStemmer):
            ps = PorterStemmer()
            self.stem = ps.stem    
        # Initialize TF-IDF
        self.computeTFIDF()
        
   
    def getTermFrequencyCount(self,paragraph):
        sentences = sent_tokenize(paragraph)
        wordFrequency = {}
        for sent in sentences:
            for word in word_tokenize(sent):
                if self.removeStopWord == True:
                    if word.lower() in self.stopwords:
                        #Ignore stopwords
                        continue
                    if not re.match(r"[a-zA-Z0-9\-\_\\/\.\']+",word):
                        continue
                #Use of Stemmer
                if self.useStemmer:
                    word = self.stem(word)
                    
                if word in wordFrequency.keys():
                    wordFrequency[word] += 1
                else:
                    wordFrequency[word] = 1
        return wordFrequency
    
    def computeTFIDF(self):
        self.paragraphInfo = {}
        for index in range(0,len(self.paragraphs)):
            wordFrequency = self.getTermFrequencyCount(self.paragraphs[index])
            self.paragraphInfo[index] = {}
            self.paragraphInfo[index]['wF'] = wordFrequency
        
        wordParagraphFrequency = {}
        for index in range(0,len(self.paragraphInfo)):
            for word in self.paragraphInfo[index]['wF'].keys():
                if word in wordParagraphFrequency.keys():
                    wordParagraphFrequency[word] += 1
                else:
                    wordParagraphFrequency[word] = 1
        
        self.idf = {}
        for word in wordParagraphFrequency:
            # Adding Laplace smoothing by adding 1 to total number of documents
            self.idf[word] = math.log((self.totalParas+1)/wordParagraphFrequency[word])
        
        #Paragraph Vector
        for index in range(0,len(self.paragraphInfo)):
            self.paragraphInfo[index]['vector'] = {}
            for word in self.paragraphInfo[index]['wF'].keys():
                self.paragraphInfo[index]['vector'][word] = self.paragraphInfo[index]['wF'][word] * self.idf[word]
    

    def query(self,pQ):
        
        # Get relevant Paragraph
        relevantParagraph = self.getSimilarParagraph(pQ.qVector)

        # Get All sentences
        sentences = []
        for tup in relevantParagraph:
            if tup != None:
                p2 = self.paragraphs[tup[0]]
                sentences.extend(sent_tokenize(p2))
        
        if len(sentences) == 0:
            return "Oops! Unable to find answer"

        #unigram similarity
        relevantSentences = self.getMostRelevantSentences(sentences,pQ,1)

        aType = pQ.aType
        
        # Default Answer
        answer = relevantSentences[0][0]

        ps = PorterStemmer()
        if aType == "PERSON":
            ne = self.getNamedEntity([s[0] for s in relevantSentences])
            for entity in ne:
                if entity[0] == "PERSON":
                    answer = entity[1]
                    answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                    qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                    if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                        continue
                    break
        elif aType == "LOCATION":
            ne = self.getNamedEntity([s[0] for s in relevantSentences])
            for entity in ne:
                if entity[0] == "GPE":
                    answer = entity[1]
                    answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                    qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                    if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                        continue
                    break
        elif aType == "ORGANIZATION":
            ne = self.getNamedEntity([s[0] for s in relevantSentences])
            for entity in ne:
                if entity[0] == "ORGANIZATION":
                    answer = entity[1]
                    answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                    qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                    if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                        continue
                    break
        elif aType == "DATE":
            allDates = []
            for s in relevantSentences:
                allDates.extend(Date(s[0]))
            if len(allDates)>0:
                answer = allDates[0]
        elif aType in ["NN","NNP"]:
            candidateAnswers = []
            ne = self.getContinuousChunk([s[0] for s in relevantSentences])
            for entity in ne:
                if aType == "NN":
                    if entity[0] == "NN" or entity[0] == "NNS":
                        answer = entity[1]
                        answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                        qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                        # If any entity is already in question
                        if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                            continue
                        break
                elif aType == "NNP":
                    if entity[0] == "NNP" or entity[0] == "NNPS":
                        answer = entity[1]
                        answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                        qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                        # If any entity is already in question
                        if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                            continue
                        break
        elif aType == "DEFINITION":
            relevantSentences = self.getMostRelevantSentences(sentences,pQ,1)
            answer = relevantSentences[0][0]
        return answer

    #       pRanking(list) : List of tuple with top 3 paragraph with its
    #                        similarity coefficient
    def getSimilarParagraph(self,queryVector):    
        queryVectorDistance = 0
        for word in queryVector.keys():
            if word in self.idf.keys():
                queryVectorDistance += math.pow(queryVector[word]*self.idf[word],2)
        queryVectorDistance = math.pow(queryVectorDistance,0.5)
        if queryVectorDistance == 0:
            return [None]
        pRanking = []
        for index in range(0,len(self.paragraphInfo)):
            sim = self.computeSimilarity(self.paragraphInfo[index], queryVector, queryVectorDistance)
            pRanking.append((index,sim))
        
        return sorted(pRanking,key=lambda tup: (tup[1],tup[0]), reverse=True)[:3]
    

    #       sim(float)          : Cosine similarity coefficient
    def computeSimilarity(self, pInfo, queryVector, queryDistance):
        # Computing pVectorDistance
        pVectorDistance = 0
        for word in pInfo['wF'].keys():
            pVectorDistance += math.pow(pInfo['wF'][word]*self.idf[word],2)
        pVectorDistance = math.pow(pVectorDistance,0.5)
        if(pVectorDistance == 0):
            return 0

        # Computing dot product
        dotProduct = 0
        for word in queryVector.keys():
            if word in pInfo['wF']:
                q = queryVector[word]
                w = pInfo['wF'][word]
                idf = self.idf[word]
                dotProduct += q*w*idf*idf
        
        sim = dotProduct / (pVectorDistance * queryDistance)
        return sim
    

    #       relevantSentences(list) : List of tuple with sentence and their
    #                                 similarity coefficient
    def getMostRelevantSentences(self, sentences, pQ, nGram=3):
        relevantSentences = []
        for sent in sentences:
            sim = 0
            if(len(word_tokenize(pQ.question))>nGram+1):
                sim = self.sim_ngram_sentence(pQ.question,sent,nGram)
            else:
                sim = self.sim_sentence(pQ.qVector, sent)
            relevantSentences.append((sent,sim))
        
        return sorted(relevantSentences,key=lambda tup:(tup[1],tup[0]),reverse=True)
    

    #       sim(float)      : Ngram Similarity Coefficient
    def sim_ngram_sentence(self, question, sentence,nGram):
        #considering stop words as well
        ps = PorterStemmer()
        getToken = lambda question:[ ps.stem(w.lower()) for w in word_tokenize(question) ]
        getNGram = lambda tokens,n:[ " ".join([tokens[index+i] for i in range(0,n)]) for index in range(0,len(tokens)-n+1)]
        qToken = getToken(question)
        sToken = getToken(sentence)

        if(len(qToken) > nGram):
            q3gram = set(getNGram(qToken,nGram))
            s3gram = set(getNGram(sToken,nGram))
            if(len(s3gram) < nGram):
                return 0
            qLen = len(q3gram)
            sLen = len(s3gram)
            sim = len(q3gram.intersection(s3gram)) / len(q3gram.union(s3gram))
            return sim
        else:
            return 0
    
 #find similar words in query and passage vectors
    #       sim(float)          : Similarity Coefficient    
    def sim_sentence(self, queryVector, sentence):
        sentToken = word_tokenize(sentence)
        ps = PorterStemmer()
        for index in range(0,len(sentToken)):
            sentToken[index] = ps.stem(sentToken[index])
        sim = 0
        for word in queryVector.keys():
            w = ps.stem(word)
            if w in sentToken:
                sim += 1
        return sim/(len(sentToken)*len(queryVector.keys()))
    
    def getNamedEntity(self,answers):
        chunks = []
        for answer in answers:
            answerToken = word_tokenize(answer)
            nc = ne_chunk(pos_tag(answerToken))
            entity = {"label":None,"chunk":[]}
            for c_node in nc:
                if(type(c_node) == Tree):
                    if(entity["label"] == None):
                        entity["label"] = c_node.label()
                    entity["chunk"].extend([ token for (token,pos) in c_node.leaves()])
                else:
                    (token,pos) = c_node
                    if pos == "NNP":
                        entity["chunk"].append(token)
                    else:
                        if not len(entity["chunk"]) == 0:
                            chunks.append((entity["label"]," ".join(entity["chunk"])))
                            entity = {"label":None,"chunk":[]}
            if not len(entity["chunk"]) == 0:
                chunks.append((entity["label"]," ".join(entity["chunk"])))
        return chunks
    
    def getContinuousChunk(self,answers):
        chunks = []
        for answer in answers:
            answerToken = word_tokenize(answer)
            if(len(answerToken)==0):
                continue
            nc = pos_tag(answerToken)
            
            prevPos = nc[0][1]
            entity = {"pos":prevPos,"chunk":[]}
            for c_node in nc:
                (token,pos) = c_node
                if pos == prevPos:
                    prevPos = pos       
                    entity["chunk"].append(token)
                elif prevPos in ["DT","JJ"]:
                    prevPos = pos
                    entity["pos"] = pos
                    entity["chunk"].append(token)
                else:
                    if not len(entity["chunk"]) == 0:
                        chunks.append((entity["pos"]," ".join(entity["chunk"])))
                        entity = {"pos":pos,"chunk":[token]}
                        prevPos = pos
            if not len(entity["chunk"]) == 0:
                chunks.append((entity["pos"]," ".join(entity["chunk"])))
        return chunks
    
    def getqRev(self, pq):
        if self.vData == None:
            # For testing
            self.vData = json.loads(open("validatedata.py","r").readline())
        revMatrix = []
        for t in self.vData:
            sent = t["q"]
            revMatrix.append((t["a"],self.sim_sentence(pq.qVector,sent)))
        return sorted(revMatrix,key=lambda tup:(tup[1],tup[0]),reverse=True)[0][0]
        
    def __repr__(self):
        msg = "Total Paras " + str(self.totalParas) + "\n"
        msg += "Total Unique Word " + str(len(self.idf)) + "\n"
        msg += str(self.getMostSignificantWords())
        return msg
