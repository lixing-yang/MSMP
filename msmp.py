import numpy as np
import random
from sympy import randprime
from scipy.optimize import fsolve
import dataprocessing as dp
import msm
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, fcluster

# LSH
def lsh(prod_set, t_value):
    nProd = len(prod_set)
    tv = prod_set
    t = t_value

    # Create Complete & Normalized Title Model Words Set
    set_title = set()
    for prod in tv:
        temp = dp.exMW_title(prod)
        set_title.update(temp)

    # Create Complete & Normalized Product Attributes Set
    set_attri = set()
    for prod in tv:
        temp = dp.exMW_kpv(prod=prod, keys=prod['featuresMap'].keys())
        set_attri.update(temp)

    # Minhashing
    # Obtain Binary Vectors
    comp_list = list(set_title.union(set_attri))
    binaries = []
    for prod in tv:
        p_title_set = dp.exMW_title(prod)
        p_features_set = dp.exMW_kpv(prod=prod, keys=prod['featuresMap'].keys())
        p_complete_set = p_title_set.union(p_features_set)
        binary = [0]*(len(comp_list))

        for i in range(len(comp_list)):
            if comp_list[i] in p_complete_set:
                binary[i] = 1
            else:
                binary[i] = 0

        binaries.append(binary)

    # Create Signature Matrix
    numHashes = int(len(comp_list)/2)
    randPrime = randprime(numHashes, numHashes+100)

    def pickRandomCoeffs(k):
        # Create a list of 'k' random values.
        randList = []

        while k > 0:
            # Get a random shingle ID.
            randIndex = random.randint(0, len(comp_list))

            # Ensure that each random number is unique.
            while randIndex in randList:
                randIndex = random.randint(0, len(comp_list))

                # Add the random number to the list.
            randList.append(randIndex)
            k = k - 1
        return randList

    coeffA = pickRandomCoeffs(numHashes)
    coeffB = pickRandomCoeffs(numHashes)

    signatures = []

    for prod in range(nProd):
        signature = []
        for i in range(numHashes):
            minHashCode = randPrime + 1
            for j in range(len(comp_list)):
                if binaries[prod][j] == 1:
                    hashCode = (coeffA[i] * j + coeffB[i]) % randPrime
                    if hashCode < minHashCode:
                        minHashCode = hashCode
            signature.append(minHashCode)
        signatures.append(signature)

    # LSH
    # Setting parameters
    def get_rbpair(b):
        return (1/b)**(1/(numHashes/b))-t

    root = fsolve(get_rbpair,np.array(numHashes/10))
    b = int(root[0])
    r = int(numHashes/b)

    # Creating bands for each signature vector
    def get_bands(signature):
        bands = []
        for i in range(b):
            band = []
            for j in range(r):
                band.append(signature[i*r+j])
            bands.append(band)
        return bands

    # Empty bucket list
    buckets = {}
    for i in range(nProd):
        buckets[i] = []

    # Assign to same buckets
    def to_buckets(id_1, id_2):
        bands_1 = get_bands(signatures[id_1])
        bands_2 = get_bands(signatures[id_2])
        for i in range(b):
            if bands_1[i] == bands_2[i]:
                buckets[id_1].append(id_2)
                return None
        return None

    # LSH
    for i in range(nProd):
        id_1 = i
        for id_2 in range(i+1, nProd):
            to_buckets(id_1, id_2)

    # No of duplicates found
    df = 0
    for key,item in buckets.items():
        for candidate in item:
            if tv[key]['shop'] != tv[candidate]['shop']:
                if 'brand' in tv[key].keys() and 'brand' in tv[candidate].keys():
                    if tv[key]['brand'] == tv[candidate]['brand']:
                        if tv[key]['modelID'] == tv[candidate]['modelID']:
                            df += 1
                else:
                    if tv[key]['modelID'] == tv[candidate]['modelID']:
                        df += 1

    # No of comparison made
    nc = 0
    for key,item in buckets.items():
        nc += len(item)

    # No of total duplicates
    dn = 0
    for prod in range(nProd):
        for prod_c in range(prod+1, nProd):
            if tv[prod]['shop'] != tv[prod_c]['shop']:
                if 'brand' in tv[prod].keys() and 'brand' in tv[prod_c].keys():
                    if tv[prod]['brand'] == tv[prod_c]['brand']:
                        if tv[prod]['modelID'] == tv[prod_c]['modelID']:
                            dn += 1
                else:
                    if tv[prod]['modelID'] == tv[prod_c]['modelID']:
                        dn += 1

    if df == 0:
        lsh_pq = 0
        lsh_pc = 0
        lsh_f1 = 0
    else:
        lsh_pq = df/nc
        lsh_pc = df/dn
        lsh_f1 = 2*lsh_pq*lsh_pc/(lsh_pq+lsh_pc)
    ttl_comparison = nProd*(nProd-1)/2
    lsh_foc = nc/ttl_comparison

    # get LSH measures
    lsh_measures ={'LSH_PQ':lsh_pq, 'LSH_PC':lsh_pc,'LSH_F1':lsh_f1, 'LSH_FOC':lsh_foc, 'Df':df, 'NC':nc, 'Dn':dn}

    # return candidate pairs in buckets
    return {'buckets':buckets,'result':lsh_measures}


# MSM
def MSM(tv_input, lsh_result):
    tv = tv_input
    buckets = lsh_result['buckets']

    nProd = len(tv_input)

    # Initialize Distance Matrix
    dist = np.full((nProd, nProd), 1000.0)
    for prod in range(nProd):
        dist[prod][prod] = 0

    def bool_updateMatrix(prod_i, prod_j):
        if prod_i['shop'] == prod_j['shop']:
            return False
        elif 'brand' in prod_i.keys() and 'brand' in prod_j.keys():
            if prod_i['brand'] != prod_j['brand']:
                return False
        return True

    # keySim threshold
    gamma = 0.756         # keySim threshold

    # MSM
    for i in range(nProd):
        if buckets[i]:
            for j in buckets[i]:
                if bool_updateMatrix(tv[i], tv[j]):
                    sim = 0
                    avgSim = 0
                    m = 0
                    w = 0
                    # Compare keys
                    nmk_i = set(tv[i]['featuresMap'].keys())
                    temp_i = set(tv[i]['featuresMap'].keys())
                    nmk_j = set(tv[j]['featuresMap'].keys())
                    temp_j = set(tv[j]['featuresMap'].keys())
                    for q in nmk_i:
                        for r in nmk_j:
                            keySim = msm.calSim(q, r)
                            if keySim > gamma:
                                valueSim = msm.calSim(tv[i]['featuresMap'][q], tv[j]['featuresMap'][r])
                                weight = keySim
                                sim = sim + weight*valueSim
                                m = m + 1
                                w = w + weight
                                if q in temp_i:
                                    temp_i.remove(q)
                                if r in temp_j:
                                    temp_j.remove(r)
                    if w > 0:
                        avgSim = sim/w
                    # mwPerc - Compare non-matching keys
                    exMW_i = dp.exMW_kpv(prod=tv[i], keys=temp_i)
                    exMW_j = dp.exMW_kpv(prod=tv[j], keys=temp_j)
                    if len(exMW_i) and len(exMW_j):
                        mwPerc = msm.mwPerc(exMW_i, exMW_j)
                    else:
                        mwPerc = 0
                    titleSim = msm.TMWMSim(tv[i], tv[j])
                    if titleSim == -1:
                        theta1 = m / min([len(tv[i]['featuresMap'].keys()), len(tv[j]['featuresMap'].keys())])
                        theta2 = 1 - theta1
                        hSim = theta1*avgSim+theta2*mwPerc
                    else:
                        theta1 = (1 - 0.65)*m/min([len(tv[i]['featuresMap'].keys()), len(tv[j]['featuresMap'].keys())])
                        theta2 = 1 - 0.65 - theta1
                        hSim = theta1*avgSim+theta2*mwPerc+0.65*titleSim

                    dist[i][j] = 1.0 - hSim
                    dist[j][i] = dist[i][j]
    return dist

# Clustering method
def clustering(prod_input, distance, epsilon):
    tv = prod_input
    dist = distance
    nProd = len(tv)

    # convert the redundant n*n square matrix form into a condensed nC2 array
    distArray = ssd.squareform(dist)
    clust = linkage(distArray, method='complete', metric='euclidean')
    results = fcluster(clust, t=epsilon, criterion='distance')

    # hash results into clusters
    largeprime = randprime(len(results), len(results) + 100)
    resultslist = list(results)
    clusters = {}
    for l in range(1, max(resultslist) + 1):
        clusters[l] = []
    for index in range(len(resultslist)):
        hcode = resultslist[index] % largeprime
        clusters[hcode].append(index)

    # MSM measures
    msm_tp = 0
    msm_fp = 0
    msm_fn = 0

    for kitem, item in clusters.items():
        if len(item) > 1:
            # number of true positive & false positive
            for cand in range(len(item)):
                for cand_o in range(cand + 1, len(item)):
                    if tv[item[cand]]['modelID'] == tv[item[cand_o]]['modelID']:
                        msm_tp += 1
                    else:
                        msm_fp += 1
    # number of total duplicates
    msm_dn = 0
    for prod in range(nProd):
        for prod_c in range(prod + 1, nProd):
            if tv[prod]['shop'] != tv[prod_c]['shop']:
                if 'brand' in tv[prod].keys() and 'brand' in tv[prod_c].keys():
                    if tv[prod]['brand'] == tv[prod_c]['brand']:
                        if tv[prod]['modelID'] == tv[prod_c]['modelID']:
                            msm_dn += 1
                else:
                    if tv[prod]['modelID'] == tv[prod_c]['modelID']:
                        msm_dn += 1
    # number of false negative
    msm_fn = msm_dn - msm_tp

    # MSM measures
    if msm_tp == 0:
        msm_p = 0
        msm_r = 0
        msm_f1 = 0
    else:
        msm_p = msm_tp / (msm_tp + msm_fp)
        msm_r = msm_tp / (msm_tp + msm_fn)
        msm_f1 = 2 * msm_p * msm_r / (msm_p + msm_r)


    msm_measures = {'MSM_P':msm_p,'MSM_R':msm_r,'MSM_F1':msm_f1,'TP':msm_tp, 'FP':msm_fp, 'FN':msm_fn}

    return msm_measures

