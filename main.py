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
        print('train LSH:',train_lsh['result'])

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
        #print('best_epsilon:', best_epsilon)
        #print('train MSM',temp_msmp)

        # performance evaluation
        test_lsh = msmp.lsh(test,t)
        dist_msmp = msmp.MSM(test,test_lsh)
        test_msmp = msmp.clustering(test,dist_msmp,best_epsilon)

        #print('test MSM', test_msmp)

        lsh_pq += test_lsh['result']['LSH_PQ']
        lsh_pc += test_lsh['result']['LSH_PC']
        lsh_f1 += test_lsh['result']['LSH_F1']
        lsh_foc += test_lsh['result']['LSH_FOC']

        msmp_tp += test_msmp['TP']
        msmp_fp += test_msmp['FP']
        msmp_fn += test_msmp['FN']
        msmp_f1 += test_msmp['MSM_F1']

        print('Done for bootstrap:', n, 'with t=',t)

    # measures for different t-value
    LSH_measures[t]['LSH_PQ'] = lsh_pq / nBootstrap
    LSH_measures[t]['LSH_PC'] = lsh_pc / nBootstrap
    LSH_measures[t]['LSH_F1'] = lsh_f1 / nBootstrap
    LSH_measures[t]['LSH_FOC'] = lsh_foc / nBootstrap

    MSMP_measures[t]['TP'] = msmp_tp / nBootstrap
    MSMP_measures[t]['FP'] = msmp_fp / nBootstrap
    MSMP_measures[t]['FN'] = msmp_fn / nBootstrap
    MSMP_measures[t]['MSM_F1'] = msmp_f1 / nBootstrap


print(MSMP_measures)
print(LSH_measures)


with open('convert.txt', 'w') as convert_file:
    convert_file.write(json.dumps(MSMP_measures))
    convert_file.write(json.dumps(LSH_measures))

print('Finished!')


# Plotting
# LSH
#FOC
LSH = {"0.3": {"LSH_PQ": 0.0021220592308833102, "LSH_PC": 0.9731046443229976, "LSH_F1": 0.0042343623379863665, "LSH_FOC": 0.13705532836208018}, "0.4": {"LSH_PQ": 0.004842574202705847, "LSH_PC": 0.862785205350541, "LSH_F1": 0.009630442705177177, "LSH_FOC": 0.05388994702655062}, "0.5": {"LSH_PQ": 0.00970501474458267, "LSH_PC": 0.7638084381039237, "LSH_F1": 0.01916417269130864, "LSH_FOC": 0.021761229290042296}, "0.6000000000000001": {"LSH_PQ": 0.028313047190764523, "LSH_PC": 0.5613997290619077, "LSH_F1": 0.05382932603991357, "LSH_FOC": 0.006032426063795137}, "0.7000000000000002": {"LSH_PQ": 0.0770029405818876, "LSH_PC": 0.34108639587362993, "LSH_F1": 0.12488840105451209, "LSH_FOC": 0.0013622873465612864}, "0.8000000000000003": {"LSH_PQ": 0.18901669758812614, "LSH_PC": 0.17076943771859027, "LSH_F1": 0.1787351612544729, "LSH_FOC": 0.0002798553723015904}, "0.9000000000000001": {"LSH_PQ": 0.16428571428571428, "LSH_PC": 0.023142857142857142, "LSH_F1": 0.04057017543859649, "LSH_FOC": 3.376354382239043e-05}}
temp_foc  = []
MSM = {"0.3": {"TP": 15.6, "FP": 61.6, "FN": 36.2, "MSM_F1": 0.24075189852871431}, "0.4": {"TP": 17.8, "FP": 70.6, "FN": 35.6, "MSM_F1": 0.25357341448350224}, "0.5": {"TP": 19.0, "FP": 65.8, "FN": 30.2, "MSM_F1": 0.28006057725818206}, "0.6000000000000001": {"TP": 20.2, "FP": 43.6, "FN": 34.6, "MSM_F1": 0.3422685657270046}, "0.7000000000000002": {"TP": 15.8, "FP": 24.2, "FN": 36.8, "MSM_F1": 0.3365906503363809}, "0.8000000000000003": {"TP": 8.4, "FP": 4.8, "FN": 48.0, "MSM_F1": 0.2388920644572819}, "0.9000000000000001": {"TP": 1.2, "FP": 0.2, "FN": 51.0, "MSM_F1": 0.04427775013714008}}
temp_f1 = []
for metric in LSH.keys():
    temp_foc.append(LSH[metric]['LSH_FOC'])
for metric in MSM.keys():
    temp_f1.append(MSM[metric]['MSM_F1'])
import matplotlib.pyplot as plt
import numpy as np
x = np.array(temp_foc)
#MSM
y1 = np.array(temp_f1)

plt.plot(x,y1)
plt.xlabel('Fraction of comparisons')
plt.ylabel('F1')
plt.savefig('msm_f1.jpg')