import json
import numpy as np
import msmp
import dataprocessing as dp


# Create data
with open('/Users/lixingyang/Desktop/Block 2/Computer Science - BA/Assignment/TVs-all-merged.json','r') as f:
    data = json.load(f)
prodID = 0
compData = {}
for key, val in data.items():
    for i in range(len(val)):
        compData[prodID] = val[i]
        prodID += 1

compDataList = dp.cleanData(list(compData.values()))

# threshold t for LSH
t_grid = list(np.arange(0.3, 0.9, 0.1))

prodKeys = [i for i in range(len(compDataList))]
nBootstrap = 5
LSH_measures = {}
MSMP_measures = {}

for t in t_grid:
    LSH_measures[t] = {}
    MSMP_measures[t] = {}

    # initialize lsh & msmp measures
    lsh_pq = 0
    lsh_pc = 0
    lsh_f1 = 0
    lsh_foc = 0
    msmp_tp = 0
    msmp_fp = 0
    msmp_fn = 0
    msmp_f1 = 0

    for n in range(nBootstrap):
        train = []
        test = []
        # bootstrap index
        idx = np.random.choice(prodKeys, replace = True, size = len(compDataList))
        # unique index sampled
        unidx = set(idx)
        # generate train & test data
        for item in range(len(compDataList)):
            if item in unidx:
                train.append(compDataList[item])
            else:
                test.append(compDataList[item])

        # LSH on training set
        train_lsh = msmp.lsh(train,t)

        # tune epsilon (similarity threshold) for training set
        epsilon_grid = list(np.arange(0.1, 1.00, 0.1))
        best_epsilon = 0
        temp = 0
        dist_msmp_train = msmp.MSM(train, train_lsh)
        for epsilon in epsilon_grid:
            train_msmp = msmp.clustering(train,dist_msmp_train,epsilon)
            if train_msmp['MSM_F1'] > temp:
                temp = train_msmp['MSM_F1']
                temp_msmp = train_msmp
                best_epsilon = epsilon

        # performance evaluation
        test_lsh = msmp.lsh(test,t)
        dist_msmp = msmp.MSM(test,test_lsh)
        test_msmp = msmp.clustering(test,dist_msmp,best_epsilon)

        lsh_pq += test_lsh['result']['LSH_PQ']
        lsh_pc += test_lsh['result']['LSH_PC']
        lsh_f1 += test_lsh['result']['LSH_F1']
        lsh_foc += test_lsh['result']['LSH_FOC']

        msmp_tp += test_msmp['TP']
        msmp_fp += test_msmp['FP']
        msmp_fn += test_msmp['FN']
        msmp_f1 += test_msmp['MSM_F1']

    # measures for different t-value
    LSH_measures[t]['LSH_PQ'] = lsh_pq / nBootstrap
    LSH_measures[t]['LSH_PC'] = lsh_pc / nBootstrap
    LSH_measures[t]['LSH_F1'] = lsh_f1 / nBootstrap
    LSH_measures[t]['LSH_FOC'] = lsh_foc / nBootstrap

    MSMP_measures[t]['TP'] = msmp_tp / nBootstrap
    MSMP_measures[t]['FP'] = msmp_fp / nBootstrap
    MSMP_measures[t]['FN'] = msmp_fn / nBootstrap
    MSMP_measures[t]['MSM_F1'] = msmp_f1 / nBootstrap
