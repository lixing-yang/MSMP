from strsimpy.qgram import QGram
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
import re
import math
import dataprocessing as dp

normal_levenshtein = NormalizedLevenshtein()

def calSim(q, r):
    if len(q) >=3 and len(r) >=3:
        qgram = QGram(3)
        dist = qgram.distance(q,r)
        similarity = (sum(qgram.get_profile(q).values())+sum(qgram.get_profile(r).values())-dist)/(sum(qgram.get_profile(q).values())+sum(qgram.get_profile(r).values()))
    else:
        similarity = 0
    return similarity

def mwPerc(exMW_i, exMW_j):
    m = len(exMW_i.intersection(exMW_j))
    n1 = len(exMW_i)
    n2 = len(exMW_j)
    mwPerc = 2*m/(n1+n2)
    return mwPerc

def nameNormalize(set_temp):
    set_clean = set()
    normalized = {'Inch':'inch','inches':'inch','"':'inch','-inch':'inch',' inch':'inch',
                  'Hertz':'hz','hertz':'hz','Hz':'hz','HZ':'hz',' hz':'hz','-hz':'hz',
                  '-':'','\(':'','\)':''
                  }
    for word in set_temp:
        for char in normalized.keys():
            word = re.sub(char, normalized[char], word)
        set_clean.update([word.upper()])
    return set_clean

def  TMWMSim(prod_i,prod_j):
    name_i = prod_i['title']
    name_j = prod_j['title']
    a = nameNormalize(set(name_i.split()))
    b = nameNormalize(set(name_j.split()))
    shared = a.intersection(b)
    cos = len(shared)/(math.sqrt(len(a)*len(b)))
    finalSim = 0
    if cos > 0.6:
        finalSim = 1
        return finalSim
    mw_i = dp.exMW_title(prod_i)
    mw_j = dp.exMW_title(prod_j)
    simMW = set()
    for i in mw_i:
        for j in mw_j:
            r_num = r'(\d+)'
            r_nonnum = r'(\D+)'
            nonnum_i = re.findall(r_nonnum,i)
            nonnum_j = re.findall(r_nonnum,j)
            if normal_levenshtein.similarity(nonnum_i, nonnum_j) > 0.9:
                if re.findall(r_num,i) != re.findall(r_num,j):
                    finalSim = -1
                    return finalSim
                else:
                    simMW.update([i,j])
    finalSim = 0.0 * cos + 1.0 * avgLvSim(a, b)
    if simMW:
        mwSim = avgLvSimMW(simMW)
        finalSim = 0.4*mwSim + 0.6*finalSim
    return finalSim

def avgLvSim(a, b):
    sum = 0
    denominator = 0
    for x in a:
        for y in b:
            denominator += (len(x)+len(y))
            sum += normal_levenshtein.similarity(x,y)*(len(x)+len(y))
    return sum/denominator

def avgLvSimMW(a):
    sum = 0
    denominator = 0
    for i in a:
        denominator += (len(i[0])+len(i[1]))
        sum += normal_levenshtein.similarity(i[0],i[1])*(len(i[0])+len(i[1]))
    return sum/denominator

