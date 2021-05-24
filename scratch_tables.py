import pandas as pd

"""
@author : Shrikanth N C (nc.shrikanth@gmail.com)
Sampling Policy that `Wins' on most measures.
"""


def getRank(df, r):
    ranks = df[df['policy'] == r]['rank'].values.tolist()
    if len(ranks) != 1:
        return 'error'

    return ranks[0]


def updateMap(ruleScoreMap, df, add=True):
    selectionRule = df['policy'].values.tolist()

    for r in selectionRule:

        if r not in ruleScoreMap:
            ruleScoreMap[r] = 0

        rank = getRank(df, r)
        if add:
            ruleScoreMap[r] += rank
        else:
            ruleScoreMap[r] -= rank


def getMetricMedian(measuresCSVPath, metric, policy):
    df = pd.read_csv(measuresCSVPath +  metric + '.csv')
    medianValues = df[df['policy'].str.strip() == policy]['median'].values.tolist()

    if len(medianValues) == 1:
        mv = medianValues[0]

        rankValues = df[df['policy'].str.strip() == policy]['rank'].values.tolist()

        if rankValues[0] >= df['rank'].max() and metric in ['auc',  'recall', 'gm']:

            return str(mv), 1
        elif rankValues[0] <= df['rank'].min() and metric in ['pf', 'd2h', 'brier', 'ifa']:
            return str(mv), 1
        else:
            return str(mv), 0

    return 'error', None


def getMetricRank(measuresCSVPath, metric, policy):
    df = pd.read_csv(measuresCSVPath +  metric + '.csv')
    medianValues = df[df['policy'].str.strip() == policy]['rank'].values.tolist()

    if len(medianValues) == 1:
        return medianValues[0]

    return 'error'


def splitPolicyText(samplingPolicy):
    spArr = samplingPolicy.split('_')
    classifier = spArr[len(spArr) - 1]

    selRule = samplingPolicy.replace('_' + classifier, '')


    return selRule, classifier


def toPretty(policy):

    ppolicy = []

    for p in policy:

        if p.startswith('E_2'):
            ppolicy.append(p.replace('E_2','E+'))



        elif p.startswith('B_scikit-learn'):
            ppolicy.append(p.replace('B_scikit-learn', 'Bellwether'))

        elif p.startswith('T_scikit-learn'):
            ppolicy.append(p.replace('T_scikit-learn', 'TCA+'))

        elif p.startswith('E_B_scikit-learn_2'):
            ppolicy.append(p.replace('E_B_scikit-learn_2', 'E+_Bellwether'))


        elif p.startswith('E_F2_T2_scikit-learn'):
            ppolicy.append(p.replace('E_F2_T2_scikit-learn', 'E+_TCA+'))

        elif p.startswith('E'):
            ppolicy.append(p.replace('E', 'E'))

        else:
            ppolicy.append(p)

    return ppolicy




def generateTable(measuresCSVPath, fileName, sortedMap):

    aggRank = 1

    policy = []

    rank = []

    d2h = []

    auc = []

    ifa = []

    brier = []

    recall = []

    gscore = []

    pf = []

    classifiers = []

    frequency = []

    pastV = None
    for key, v in sortedMap.items():
        f = 0

        if pastV is None:
            pastV = v
        elif pastV != v:
            pastV = v
            aggRank += 1

        rawPolicy, classifier = splitPolicyText(key)
        policy.append(rawPolicy)
        classifiers.append(classifier)
        rank.append(aggRank)

        mv, win = getMetricMedian(measuresCSVPath, 'd2h', key)
        f += win
        d2h.append(mv)

        mv, win = getMetricMedian(measuresCSVPath, 'auc', key)
        f += win
        auc.append(mv)

        mv, win = getMetricMedian(measuresCSVPath, 'ifa', key)
        f += win
        ifa.append(mv)

        mv, win = getMetricMedian(measuresCSVPath, 'brier', key)
        f += win
        brier.append(mv)

        mv, win = getMetricMedian(measuresCSVPath, 'recall', key)
        f += win
        recall.append(mv)

        mv, win = getMetricMedian(measuresCSVPath, 'pf', key)
        f += win
        pf.append(mv)

        mv, win = getMetricMedian(measuresCSVPath, 'gm', key)
        f += win
        gscore.append(mv)

        frequency.append(f)

    df = pd.DataFrame()

    policy = toPretty(policy)

    df['Policy'] = policy
    df['Classifier'] = classifiers
    df['Wins'] = frequency

    df['Recall'] = recall

    df['PF'] = pf

    df['AUC'] = auc

    df['D2H'] = d2h

    df['Brier'] = brier

    df['GM'] = gscore

    df['IFA'] = ifa


    df = df.sort_values(['Wins', 'Recall', 'PF'], ascending=False)
    df.to_csv('scratch_'+fileName + '.csv', index=False)
    print("Generated ",' scratch_'+fileName+'.csv in the current working directory!')


def run(fileName, measuresCSVPath):

    ruleScoreMap = {}

    d2hDF = pd.read_csv(measuresCSVPath +  'd2h.csv')
    rocDF = pd.read_csv(measuresCSVPath +  'auc.csv')
    ifaDF = pd.read_csv(measuresCSVPath +  'ifa.csv')
    brierDF = pd.read_csv(measuresCSVPath +  'brier.csv')
    recallDF = pd.read_csv(measuresCSVPath +  'recall.csv')
    pfDF = pd.read_csv(measuresCSVPath +  'pf.csv')
    gscoreDF = pd.read_csv(measuresCSVPath +  'gm.csv')

    updateMap(ruleScoreMap, d2hDF, False)
    updateMap(ruleScoreMap, rocDF, True)
    updateMap(ruleScoreMap, ifaDF, False)
    updateMap(ruleScoreMap, brierDF, False)
    updateMap(ruleScoreMap, recallDF, True)
    updateMap(ruleScoreMap, pfDF, False)

    updateMap(ruleScoreMap, gscoreDF, True)

    sortedMap = {k.strip(): v for k, v in sorted(ruleScoreMap.items(), key=lambda item: item[1], reverse=True)}

    generateTable(measuresCSVPath, fileName, sortedMap)


if __name__ == '__main__':
    print("Generating")
    run('TableIV', './output/table4/scratch_table4_')

