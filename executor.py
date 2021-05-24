
import os
os.environ['OMP_NUM_THREADS'] = "1"
import time as goodtime
from identify_bell_project import *
import tca_plus
from transformation import *


from Util import *
from Constants import *

from multiprocessing import Process




import SMOTE
import feature_selector
import CFS



import data_manager


import metrices



from  sklearn.ensemble import *

from  sklearn.linear_model._logistic import *


from  sklearn.linear_model._stochastic_gradient import *

from  sklearn.naive_bayes import *
from  sklearn.neighbors._classification import *

from  sklearn.svm._classes import *
from  sklearn.tree._classes import *


import warnings
warnings.filterwarnings("ignore")



"""
@author : Shrikanth N C (nc.shrikanth@gmail.com)  
Evaluate various train approaches across various learners and records all measures
"""

RESULTS_FOLDER = 'output'


# TCA_DATA_FOLDER = 'TTD_TCA_DATA_FOLDER'

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

class MLUtil(object):

    def __init__(self):
        self.cores = 1

    def get_features(self,df):
        fs = feature_selector.featureSelector()
        df,_feature_nums,features = fs.cfs_bfs(df)
        return df,features

    def apply_pca(self, df):
        return df

    def apply_cfs(self,df):

        copyDF = df.copy(deep=True)

        y = copyDF.Buggy.values
        X = copyDF.drop(labels=['Buggy'], axis=1)

        X = X.values

        selected_cols = CFS.cfs(X, y)

        finalColumns = []

        for s in selected_cols:

            if s > -1:
                finalColumns.append(copyDF.columns.tolist()[s])

        finalColumns.append('Buggy')

        if 'loc' in finalColumns:
            finalColumns.remove('loc')


        return None, finalColumns


    def apply_normalize(self, df):
        """
        Not used
        :param df:
        :return:
        """

        return df

    def apply_smote(self,df):

        originalDF = df.copy(deep=True)

        try:
            cols = df.columns
            smt = SMOTE.smote(df)
            df = smt.run()
            df.columns = cols

        except:
            return originalDF

        return df




def getFirstChangesBetween(project, startDate, endDate):

    releases = project.getReleases()


    changeDF = None
    for r in releases:

        if r.getStartDate() >= startDate and r.getReleaseDate() <= endDate:

            if r.getChanges() is not None and len(r.getChanges()) > 1:

                if changeDF is None:
                    changeDF = r.getChanges().copy(deep=True)
                else:
                    changeDF = changeDF.append(r.getChanges().copy(deep=True))



    return changeDF





def toNominal(changes):

    releaseDF = changes
    releaseDF.loc[releaseDF['Buggy'] >= 1, 'Buggy'] = 1
    releaseDF.loc[releaseDF['Buggy'] <= 0, 'Buggy'] = 0
    d = {1: True, 0: False}
    releaseDF['Buggy'] = releaseDF['Buggy'].map(d)

    return changes





def validTrainChanges(changes):
    return changes is not None and len(changes[changes['Buggy'] > 0]) > 5 \
           and len(changes[changes['Buggy'] == 0]) > 5


def validTestChanges(changes):
    return changes is not None and len(changes) > 4 and len(changes[changes['Buggy'] > 0]) > 0 \
           and len(changes[changes['Buggy'] == 0]) > 0


def validate(trainChanges, testChanges):
    return validTrainChanges(trainChanges) and validTestChanges(testChanges)


def getReleaseObject(pname, releaseDate):
    project = data_manager.getProject(pname)
    allReleases = project.getReleases()

    for r in allReleases:
        if r.getReleaseDate() == releaseDate:
            return r

    return None

def computeMeasures(test_df, clf, timeRow, codeChurned):


    F = {}

    if getSimpleName(clf) == 'KNN' and  test_df.shape[0] < 6: #because knn needs 5 neighbors minimum
        return None

    test_y = test_df.Buggy
    test_X = test_df.drop(labels=['Buggy'], axis=1)

    testStart = goodtime.time()

    try:

        predicted = clf.predict(test_X)
    except Exception as e:
        predicted = None

    testDiff = goodtime.time() - testStart
    timeRow.append(testDiff)

    try:
        abcd = metrices.measures(test_y, predicted, codeChurned)
    except Exception as e:
        print('clf error --> ', 'clf = ', clf, 'e = ', e)
        abcd = None

    errorMessage = 'MEASURE_ERROR'

    if abcd is None:
        errorMessage = 'CLF_ERROR'

    try:
        F['f1'] = [abcd.calculate_f1_score()]
    except:
        F['f1'] = [errorMessage]

    try:
        F['precision'] = [abcd.calculate_precision()]
    except:
        F['precision'] = [errorMessage]

    try:
        F['recall'] = [abcd.calculate_recall()]
    except:
        F['recall'] = [errorMessage]



    try:
        F['pf'] = [abcd.get_pf()]
    except:
        F['pf'] = [errorMessage]


    try:
        F['g-score'] = [abcd.get_g_score()]
    except:
        F['g-score'] = [errorMessage]

    try:
        F['d2h'] = [abcd.calculate_d2h()]
    except:
        F['d2h'] = [errorMessage]

    # try:
    #     F['accuracy'] = [abcd.calculate_accuracy() * -1 ]
    # except:
    #     F['accuracy'] = [errorMessage]
    #
    #
    # print('warn')



    try:
        F['pci_20'] =  [1]
    except Exception as e:
        F['pci_20'] = [errorMessage]

    try:

        # tempPredicted = predicted
        # temptest_y = test_y.values.tolist()
        # initialFalseAlarm = 0
        #
        # for i in range(0, len(tempPredicted)):
        #
        #     if tempPredicted[i] == True and temptest_y[i] == False:
        #         initialFalseAlarm += 1
        #     elif not tempPredicted[i]:
        #         continue
        #     elif tempPredicted[i] == True and temptest_y[i] == True:
        #         break
        #
        # F['ifa'] = [initialFalseAlarm]

        F['ifa'] = [abcd.get_ifa()]
    except Exception as e:
    #     print('\t ',errorMessage)
        F['ifa'] = [errorMessage]

    try:
        F['ifa_roc'] = [abcd.get_ifa_roc()]
    except:
        F['ifa_roc'] = [errorMessage]

    try:
        F['roc_auc'] = [abcd.get_roc_auc_score()]
    except:
        F['roc_auc'] = [errorMessage]

    try:
        F['pd'] = [abcd.get_pd()]
    except:
        F['pd'] =[errorMessage]

    try:
        F['tp'] = [abcd.get_tp()]
    except:
        F['tp'] =[errorMessage]

    try:
        F['tn'] = [abcd.get_tn()]
    except:
        F['tn'] = [errorMessage]

    try:
        F['fp'] = [abcd.get_fp()]
    except:
        F['fp'] = [errorMessage]

    try:
        F['fn'] = [abcd.get_fn()]
    except:
        F['fn'] = [errorMessage]

    try:
        F['negopos'] = [abcd.negOverPos()]
    except:
        F['negopos'] = [errorMessage]

    try:
        F['balance'] = [abcd.balance()]
    except:
        F['balance'] = [errorMessage]

    try:
        F['brier'] = [abcd.brier()]
    except:
        F['brier'] = [errorMessage]

    try:
        pt = abcd.get_popt_20()
        F['popt20'] = [ pt ]
    except:
        F['popt20'] = [errorMessage]

    return F




if RESULTS_FOLDER.startswith('results_TTD_ALL_PAIRS') or 'results_TTD_RUNTIME' == RESULTS_FOLDER:
    CLASSIFIERS = [  LogisticRegression ]
else:
    CLASSIFIERS = [LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier,
                   GaussianNB, KNeighborsClassifier]


def getClassifiers():

    classifiers = [x() for x in CLASSIFIERS]
    return classifiers

def getFeatureSelectors():
    return [ 'CFS' ]


def getSimpleNames():

    return [ getSimpleName(clf) for clf in getClassifiers() ]

def getHeader():

    header = ['projectName', 'trainApproach', 'trainReleaseDate', 'testReleaseDate', 'train_changes', 'test_changes',
              'train_Bug_Per', 'test_Bug_Per', 'features_selected', 'classifier', 'featureSelector', 'SMOTE',
              'SMOTE_time', 'test_time', 'train_time', 'goal_score']

    header += METRICS_LIST

    return header

def getSimpleName(classifier, tune=False):

    if tune:
        PREFIX = 'TUNED_'
    else:
        PREFIX = ''

    if type(classifier) == VotingClassifier:
        s =   PREFIX + 'VT_'
        for e in classifier.get_params()['estimators']:

            s += getSimpleName(e[1]) + '_'

        return s


    return PREFIX + str(classifier.__class__.__name__)


def getCacheKey(fs, projectName, trainReleaseDate, testReleaseDate):

    return fs+"_"+projectName+"_"+str(trainReleaseDate)+"_"+str(testReleaseDate)


def getFileName(projectName):
        return './'+RESULTS_FOLDER+'/project_' + projectName + "_results.csv"



def dontDrop(changesDF, consider):

    for col in ['ns', 'nd', 'nf', 'entropy', 'ld', 'la', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']:
        if col not in consider:

            if col in changesDF.columns:
                changesDF = changesDF.drop(col, 1)

    return changesDF


def normalize_log(changesDF):
    """
        log Normalization
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for c in [c for c in changesDF.columns if changesDF[c].dtype in numerics]:

        if c != 'Buggy' and c != 'fix' and c != 'loc' and c != 'author_email':
            changesDF[c] = changesDF[c] + abs(changesDF[c].min()) + 0.00001

    for c in [c for c in changesDF.columns if changesDF[c].dtype in numerics]:

        if c != 'Buggy' and c != 'fix' and c != 'loc' and c != 'author_email':
            changesDF[c] = np.log10(changesDF[c])

    return changesDF


def splitFeatures(type):

    featArr = type.split('$')

    features = []
    for f in featArr:

        if len(f) > 1:
            features.append(f)

    # print("Returning ",features)
    return features


def customPreProcess(changesDF, type, tune):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for c in [c for c in changesDF.columns if changesDF[c].dtype in numerics]:
        if c != 'Buggy':
            changesDF[c] = changesDF[c] + abs(changesDF[c].min()) + 0.00001

    # changesDF['author_email'] = toMajorMinor(changesDF['author_email'].values.tolist())

    if type == 'diffusion':
        changesDF = dontDrop(changesDF, ['ns', 'nd', 'nf', 'entropy'])
    elif type == 'size':
        # changesDF = dontDrop(changesDF, ['ld', 'la', 'lt'])
        changesDF = dontDrop(changesDF, ['la', 'lt'])
    elif type == 'purpose':
        changesDF = dontDrop(changesDF, ['fix'])
    elif type == 'history':
        changesDF = dontDrop(changesDF, ['ndev', 'age', 'nuc'])
    elif type == 'experience':
        changesDF = dontDrop(changesDF, ['exp', 'rexp', 'sexp', 'author_email'])
    elif type == 'top':
        changesDF = dontDrop(changesDF, ['entropy', 'la', 'ld', 'lt', 'exp' ])
    elif type is None:
        # changesDF = changesDF.drop('author_email', 1)

        changesDF['la'] = changesDF['la'] / changesDF['lt']
        changesDF['ld'] = changesDF['ld'] / changesDF['lt']

        changesDF['lt'] = changesDF['lt'] / changesDF['nf']
        changesDF['nuc'] = changesDF['nuc'] / changesDF['nf']

        changesDF = changesDF.drop('nd', 1)
        changesDF = changesDF.drop('rexp', 1)

    elif type == 'i1':
        changesDF = dontDrop(changesDF, ['la', 'lt', 'entropy'])
    elif type == 'i2':
        changesDF = dontDrop(changesDF, ['la', 'lt', 'exp'])
    elif type == 'i3':
        changesDF = dontDrop(changesDF, ['la', 'lt', 'entropy', 'exp'])
    elif type == 'i4':
        changesDF = dontDrop(changesDF, ['la', 'lt', 'ndev'])
    elif type == 'i5':
        changesDF = dontDrop(changesDF, ['la', 'lt', 'entropy', 'ndev', 'exp'])
    elif '$' in type:
        changesDF = dontDrop(changesDF, splitFeatures(type))
    else:
        float("error")


    for c in [c for c in changesDF.columns if changesDF[c].dtype in numerics]:

        if c != 'Buggy' and c != 'fix' and c != 'loc' and c != 'author_email':
            changesDF[c] =  changesDF[c] + 0.0000001

    # if not tune:

    """
    log Normalization
    """
    for c in [c for c in changesDF.columns if changesDF[c].dtype in numerics]:

        if c != 'Buggy' and c != 'fix' and c != 'loc' and c!='author_email':

            changesDF[c] = np.log10(changesDF[c])

    return changesDF




def getTrue(labels):

    c = 0

    for x in labels:

        if x or x == 'True':
            c += 1

    return c


def getFalse(labels):
    c = 0

    for x in labels:

        if x == False or x == 'False':
            c += 1

    return c


smoteCopyMap = {}

current_time_ms = lambda: int(round(goodtime.time() * 1000))



def new_classifier(learner):

    return clone(learner)

def applyCFS(tune, trainApproach):

     return not trainApproach.startswith('T_')

def applySMOTE(tune, trainApproach):
    return True


def apply_custom_processing(tune, trainApproach):

    if trainApproach.startswith('T_'):
        return False
    else:
        return True

cfsMap = {}

train_cache = {}

def preprocess(projectName,trainReleaseDate, tune, trainApproach, train_changes, test_changes, tune_changes, learner, retainFew):

    if trainApproach.startswith('T_') or trainApproach.startswith('E_T2_') or trainApproach.startswith('E_F2_T2_'): # TCA plus

        tca_train_changes = MLUtil().apply_smote(train_changes)
        tca_train_changes = customPreProcess(tca_train_changes, retainFew, tune)

        test_changes =  customPreProcess(test_changes, retainFew, tune)

        if trainApproach.startswith('T_'):
            tca_trainChanges, tca_testChanges = tca_plus.apply_tcaplus(tca_train_changes.copy(deep=True), test_changes.copy(deep=True), 5)
            return tca_trainChanges, tca_testChanges, None, 't5', True
        elif trainApproach.startswith('E_T2_'):
            tca_trainChanges, tca_testChanges = tca_plus.apply_tcaplus(tca_train_changes.copy(deep=True),
                                                                       test_changes.copy(deep=True), 2)
            return tca_trainChanges, tca_testChanges, None, 't2', True
        elif trainApproach.startswith('E_F2_T2_'):
            tca_trainChanges, tca_testChanges = tca_plus.apply_tcaplus(tca_train_changes.copy(deep=True),
                                                                       test_changes.copy(deep=True), 2)
            return tca_trainChanges, tca_testChanges, None, 'tsize', True



    process_train_changes = True

    if trainApproach.endswith('_ALL_CHANGES') and trainApproach in train_cache:
        process_train_changes = False

    fselector = ''

    if apply_custom_processing(tune, trainApproach):

        if process_train_changes:
            train_changes = customPreProcess(train_changes, retainFew, tune)

        test_changes = customPreProcess(test_changes, retainFew, tune)
        if tune_changes is not None:
            tune_changes = customPreProcess(tune_changes, retainFew, tune)

        fselector = 'CUS'


    smote = False

    if applyCFS(tune, trainApproach):

        fselector += '_CFS'

        # cfs_key = projectName + '_' + trainApproach + "_" + trainReleaseDate + "_" + str(len(train_changes))+"_"+getSimpleName(learner, tune)

        cfs_key = projectName + '_' + trainApproach + "_" + trainReleaseDate + "_" + str(len(train_changes))+"_"+str(tune)

        if cfs_key in cfsMap:
            someDF, selected_cols = None, cfsMap[cfs_key]
        else:
            # if tune:
            #     selected_cols = ['la', 'lt', 'Buggy']
            #     cfsMap[cfs_key] = selected_cols
            #     print("<< warning only cfs >>")
            # else:
            someDF, selected_cols = MLUtil().apply_cfs(train_changes)
            cfsMap[cfs_key] = selected_cols


        if  getBugCount(train_changes) != getNonBugCount(train_changes): # applySMOTE(tune, trainApproach):
            smote = True
            train_changes = MLUtil().apply_smote(train_changes)

    else:
        selected_cols = train_changes.columns.tolist()

        # if 'loc' in selected_cols:
        #     selected_cols.remove('loc')

    if process_train_changes:
        train_changes = train_changes[selected_cols]

    test_changes = test_changes[selected_cols]

    if tune_changes is not None:
        tune_changes = tune_changes[selected_cols]

    if not process_train_changes:
        train_changes = train_cache[trainApproach]
        # print("taking from cache ",trainApproach)
    else:
        if trainApproach.endswith('_ALL_CHANGES'):
            train_cache[trainApproach] = train_changes
            # print("putting in cache")



    return train_changes, test_changes, tune_changes, fselector, smote







def writeRowEntry(test_changes_processed, clf, trainX, trainY, train_start, projectName, trainApproach, trainReleaseDate, testReleaseDate,
                  train_changes_processed, returnResults, classifierName, fselector, smote, goal_score):

    train_time = goodtime.time() - train_start

    test_start = goodtime.time()
    F = computeMeasures(test_changes_processed, clf, [], [1 for x in range(0, len(test_changes_processed))])
    test_time = goodtime.time() - test_start

    metricsReport = F

    featuresSelectedStr = ''

    for sc in trainX.columns.tolist():
        featuresSelectedStr += '$' + str(sc)

    result = [projectName, trainApproach, trainReleaseDate, testReleaseDate,
              len(train_changes_processed),
              len(test_changes_processed),
              percentage(len(train_changes_processed[train_changes_processed['Buggy'] > 0]),
                         len(train_changes_processed)),
              percentage(len(test_changes_processed[test_changes_processed['Buggy'] > 0]),
                         len(test_changes_processed)),
              featuresSelectedStr, classifierName, fselector, str(smote), 0, test_time, train_time, goal_score]

    if metricsReport is not None:
        for key in metricsReport.keys():
            result += metricsReport[key]
    else:
        for m in METRICS_LIST:
            result.append(str('UNABLE'))

    if UNIT_TEST == False:

        if returnResults:
            return metricsReport['balance']
        else:
            writeRow(getFileName(projectName), result)

    else:
        print("***************** ", classifierName, fselector, train_changes_processed.columns.tolist(),
              test_changes_processed.columns.tolist(), " Results **************************")
        if metricsReport is not None:

            print('\tprecision', metricsReport['precision'])
            print('\trecall', metricsReport['recall'])
            print('\tpf', metricsReport['pf'])
            print('\troc_auc', metricsReport['roc_auc'])
            print('\td2h', metricsReport['d2h'])
            print('*** \n ')
            return metricsReport['precision'], metricsReport['recall'], metricsReport['pf'], []
        else:

            print(metricsReport, getTrue(trainY), getFalse(trainY))
            return None, None, None, None



def tca_sample(train_changes):

    return train_changes.head(150).copy(deep=True).append(
        train_changes.tail(150).copy(deep=True)).copy(deep=True)


def early_sample(train_changes):

    buggyChangesDF = train_changes[train_changes['Buggy'] == True]
    nonBuggyChangesDF = train_changes[train_changes['Buggy'] == False]

    sample_size = min(25, min(len(buggyChangesDF), len(nonBuggyChangesDF)))

    return buggyChangesDF.sample(sample_size).copy(deep=True).append(
        nonBuggyChangesDF.sample(sample_size).copy(deep=True)).copy(deep=True)

fitCache = {}

def performPredictionRunner(projectName, originalTrainChanges, originalTestChanges, trainReleaseDate, testReleaseDate, trainApproach,
                            testReleaseStartDate=None, retainFew=None, tuneChanges=None, returnResults=False):
    for i in range(1, 2):

        if validTrainChanges(originalTrainChanges) == False or \
                validTestChanges(originalTestChanges) == False:

            return


        project_object = data_manager.getProject(projectName)
        tune_date = None

        for dummy_flag in [  False  ]:

            for learner in getClassifiers():

                    # if dummy_flag and getSimpleName(learner) not in [ str(RidgeClassifier.__name__), str(LogisticRegression.__name__) ]:
                    #     if DEBUG:
                    #         print("continue for ", learner)
                    #     continue

                    process_start =  goodtime.time()

                    train_changes_copy = originalTrainChanges.copy(deep=True)
                    test_changes_copy = toNominal(originalTestChanges.copy(deep=True))
                    tune_changes_copy = None

                    train_changes_processed, test_changes_processed, tune_changes_processed, fselector, smote = preprocess(
                        projectName,
                        trainReleaseDate, dummy_flag,
                        trainApproach,
                        train_changes_copy,
                        test_changes_copy, tune_changes_copy, learner, retainFew)

                    clfkey = trainApproach + '_' + getSimpleName(learner, dummy_flag) + '_' + str(trainReleaseDate)

                    # if  clfkey not in fitCache or \
                    #         (dummy_flag and clfkey in fitCache and
                    #          (float(testReleaseStartDate) - float(fitCache[clfkey][2])   > 6 * one_month)):
                    #
                    if  clfkey not in fitCache  :

                    # if (dummy_flag and clfkey in fitCache and (float(testReleaseStartDate) - float(fitCache[clfkey][1])   > 4 * one_month)):

                        if dummy_flag:

                            temp_train = originalTrainChanges.copy(deep=True)
                            temp_tune = early_sample(project_object.getAllChanges().head(150))

                            temp_train_processed, temp_tune_processed, x, y, z = preprocess(
                                    projectName,
                                    trainReleaseDate, dummy_flag,
                                    trainApproach,
                                temp_train,
                                temp_tune, None, learner, retainFew)





                            if DEBUG:
                                print("Received = ", best_params, preprocessor)

                            tune_date = testReleaseStartDate



                        else:
                            classifier = new_classifier(learner)
                            best_params = classifier
                            preprocessor = None
                            tune_date = None


                        fitCache[clfkey] = [best_params, preprocessor, tune_date]

                    temp_preprocessor = fitCache[clfkey][1]

                    if temp_preprocessor is not None:
                        train_changes_processed = transform(train_changes_processed, temp_preprocessor)
                        test_changes_processed = transform(test_changes_processed, temp_preprocessor)

                    trainY = train_changes_processed.Buggy
                    trainX = train_changes_processed.drop(labels=['Buggy'], axis=1)

                    for i in range(0,1):

                        clf = new_classifier(fitCache[clfkey][0])

                        if dummy_flag:
                            clf.set_params(**fitCache[clfkey][0].get_params())

                        if DEBUG:
                            print('\t', i, dummy_flag, clf)

                        if dummy_flag:
                            classifierName = 'dummy_V2'
                        else:
                            classifierName = getSimpleName(clf, dummy_flag)

                        try:
                            clf.fit(trainX, trainY)
                        except Exception as e:
                            continue

                        spl = ''
                        if tune_changes_processed is not None:
                            spl = len(tune_changes_processed)

                        writeRowEntry(test_changes_processed, clf, trainX, trainY, process_start, projectName, trainApproach,
                                      trainReleaseDate, testReleaseDate,
                                      train_changes_processed, returnResults, classifierName, fselector, smote, str('')+"_"+str(i))
                        # printLocal()



def getBugCount(  xDF ):

    return len(xDF[xDF['Buggy'] == True])

def getNonBugCount(  xDF):

    return len(xDF[xDF['Buggy'] == False])










def run_performance(projectName):

    print("Cores  = ", os.cpu_count())

    if projectName == 'homebrew-cask':

        project_obj = getProject(projectName)

        releaseList = project_obj.getReleases()

        testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)

        early_changes = early_sample(project_obj.getAllChanges().head(150))

        for testReleaseObj in testReleaseList:

            all_changes = getFirstChangesBetween(project_obj, 0, testReleaseObj.getStartDate())

            performPredictionRunner(projectName, all_changes.copy(deep=True), testReleaseObj.getChanges(),
                                    str(len(all_changes)), testReleaseObj.getReleaseDate(), 'ALL')

            performPredictionRunner(projectName, early_changes.copy(deep=True), testReleaseObj.getChanges(),
                                    str(len(early_changes)), testReleaseObj.getReleaseDate(), 'E')
    else:
        print("skipping ", projectName)


def print_time():

    all_time = 0.0
    early_time = 0.0

    for p in getProjectNames():


            all_df = pd.read_csv('./results_TTD_ALL_PAIRS/project_'+p+'_results.csv')
            all_df = all_df [ all_df['classifier'] == 'LogisticRegression']

            early_df = pd.read_csv('./results_TTD_ALL_PAIRS_Early/project_' + p + '_results.csv')
            early_df = early_df[early_df['classifier'] == 'LogisticRegression']

            all_time += sum(all_df['total_time'].values.tolist())

            early_time += sum(early_df['test_time'].values.tolist())
            early_time += sum(early_df['train_time'].values.tolist())

            print(all_time, early_time)



    print("all time = ", all_time)
    print("early time = ", early_time)


def bell_project_index(bell_project):

    return bell_project


def run_train_time():

    bellproject = getProject('homebrew-cask').getAllChanges()
    targetProject = getProject('camel')

    for testReleaseObj in targetProject.getReleases():

        if len(testReleaseObj.getChanges()) > 25:
            testChanges = testReleaseObj.getChanges().copy(deep=True)
            break

    t1 = []
    t2 = []
    t3 = []
    amtt = []

    for amt in [50, 100, 200, 400, 800, 1600, 3200, 6400]:

        amtt.append(amt)

        test_changes = testChanges.copy(deep=True)
        train_changes = bellproject.head(amt).copy(deep=True)

        current_time_ms = goodtime.time()
        performPredictionRunner('camel', train_changes.copy(deep=True), test_changes.copy(deep=True),
                                str(len(train_changes)),
                                testReleaseObj.getReleaseDate(), 'T_' + str(bell_project_index('homebrew-cask')))
        t1.append(goodtime.time() - current_time_ms)

        current_time_ms = goodtime.time()

        performPredictionRunner('camel', train_changes.copy(deep=True), testReleaseObj.getChanges(),
                                str(len(train_changes)), testReleaseObj.getReleaseDate(), 'B')
        t2.append(goodtime.time() - current_time_ms)
        current_time_ms = goodtime.time()

        train_changes = early_sample(train_changes.head(150))
        performPredictionRunner('camel', train_changes.copy(deep=True), test_changes.copy(deep=True),
                                str(len(train_changes)),
                                testReleaseObj.getReleaseDate(), 'E_B' + str(bell_project_index('homebrew-cask')))

        t3.append(goodtime.time() - current_time_ms)
        print(amtt, t1, t2, t3)

    print(amtt, t1, t2, t3)

    df = pd.DataFrame()

    df['amt'] = amtt
    df['t1']= t1
    df['t2'] = t2
    df['t3'] = t3

    df.to_csv('runtime.csv', index=False)



def run_tca_multiple_bell(projectName):

    for bell_project in BELLWETHER_PROJECTS:
        run_tca(projectName, bell_project)

def run_tca(projectName, bell_project):

    project_obj = getProject(projectName)
    releaseList = project_obj.getReleases()
    testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)

    bell_project_changes = getProject(bell_project).getAllChanges()

    if len(bell_project_changes) <= 300:
        print(bell_project, ' has less than 300 changes')
        return

    more_train_changes = tca_sample(bell_project_changes)
    early_trainChanges = early_sample(bell_project_changes.copy(deep=True).head(150))


    for testReleaseObj in testReleaseList:

        try:

            testChanges = testReleaseObj.getChanges().copy(deep=True)

            performPredictionRunner(projectName, more_train_changes.copy(deep=True), testChanges.copy(deep=True),
                                    str(len(more_train_changes)),
                                    testReleaseObj.getReleaseDate(), 'TCA+')


            performPredictionRunner(projectName, early_trainChanges.copy(deep=True), testChanges.copy(deep=True), str(len(early_trainChanges)),
                                    testReleaseObj.getReleaseDate(), 'E+_TCA+' ,
                                    None, 'size')

            # performPredictionRunner(projectName, early_trainChanges.copy(deep=True), testChanges.copy(deep=True), str(len(early_trainChanges)),
            #                         testReleaseObj.getReleaseDate(), 'E_T2_' + str(bell_project_index(bell_project)))

        except Exception as e:
            print('TCAPLUS EXCEPTION : ', e)

def validTrainRegion(trainingRegion):
    return trainingRegion is not None and len(trainingRegion) > 2





def convert(v, metric):
    vv= []

    if v is not None:

        for vvv in v:

            try:
                converted = float(vvv)
                if str(converted).lower() != 'nan':
                    vv.append(converted)
                elif metric in ['g-score', 'gm']:
                    vv.append(0)
            except:
                continue

    return vv


def run_multiple_bellwether(projectName, type=None):

    for bellwether_project in BELLWETHER_PROJECTS:

        if projectName == bellwether_project:
            continue

        project_obj = data_manager.getProject(projectName)
        releaseList = project_obj.getReleases()
        testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)
        bell_changes = getProject(bellwether_project).getAllChanges()
        early_changes = early_sample(bell_changes.head(150))

        for testReleaseObj in testReleaseList:

            try:

                performPredictionRunner(projectName, bell_changes.copy(deep=True), testReleaseObj.getChanges(),
                                        str(len(bell_changes)), testReleaseObj.getReleaseDate(),
                                        'B_' + str(bell_project_index(bellwether_project)))

                performPredictionRunner(projectName, early_changes.copy(deep=True), testReleaseObj.getChanges(),
                                        str(len(early_changes)), testReleaseObj.getReleaseDate(),
                                        'E_B_' + str(bell_project_index(bellwether_project)) + '_2', None, 'size')

                # if type is None:
                #     performPredictionRunner(projectName, bell_changes.copy(deep=True), testReleaseObj.getChanges(),
                #                         str(len(bell_changes)), testReleaseObj.getReleaseDate(), 'B_'+str(bell_project_index(bellwether_project)), None, type)
                #     # performPredictionRunner(projectName, early_changes.copy(deep=True), testReleaseObj.getChanges(),
                #     #                         str(len(early_changes)), testReleaseObj.getReleaseDate(),
                #     #                         'E_B_' +str(bell_project_index(bellwether_project)), None, type)
                # else:
                #     # performPredictionRunner(projectName, bell_changes.copy(deep=True), testReleaseObj.getChanges(),
                #     #                         str(len(bell_changes)), testReleaseObj.getReleaseDate(),
                #     #                         'B_' + str(bell_project_index(bellwether_project))+'_2', None, type)
                #     performPredictionRunner(projectName, early_changes.copy(deep=True), testReleaseObj.getChanges(),
                #                             str(len(early_changes)), testReleaseObj.getReleaseDate(),
                #                             'E_B_' + str(bell_project_index(bellwether_project))+'_2', None, type)
            except Exception as e:
                print('Exception @ run_multiple_bellwether ', projectName, bellwether_project)

def run_bellwether(projectName):

    if projectName == BELLWETHER_PROJECT:
        return

    project_obj = data_manager.getProject(projectName)
    releaseList = project_obj.getReleases()
    testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)
    bell_changes = getProject(BELLWETHER_PROJECT).getAllChanges()

    for testReleaseObj in testReleaseList:

        performPredictionRunner(projectName, bell_changes.copy(deep=True), testReleaseObj.getChanges(),
                                str(len(bell_changes)), testReleaseObj.getReleaseDate(), 'B')

def run_early_bellwether(projectName):

    if projectName == BELLWETHER_PROJECT:
        return

    project_obj = data_manager.getProject(projectName)
    releaseList = project_obj.getReleases()
    testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)

    early_bell_changes = early_sample(getProject(BELLWETHER_PROJECT).getAllChanges().head(150))

    for testReleaseObj in testReleaseList:
        performPredictionRunner(projectName, early_bell_changes.copy(deep=True), testReleaseObj.getChanges(),
                                str(len(early_bell_changes)), testReleaseObj.getReleaseDate(), 'E_B')





def test_all_releases(projectName):

    print("Appending prediction evaluations for ", projectName, ' to ',
          str(os.getcwd()) + "/output/" + projectName + '.csv')

    writeRow(getFileName(projectName), getHeader())

    """
    WPDP
    """
    run_early(projectName)
    run_early_2(projectName)

    """
    CPDP
    """
    run_bellwether(projectName)
    run_early_bellwether_2(projectName)
    run_tca(projectName, BELLWETHER_PROJECT)





def resultExists(p):
    filePath = './'+RESULTS_FOLDER+'/project_' + p + "_results.csv"
    return path.exists(filePath)
    # return False

def run_all(projectName):

    project_obj = getProject(projectName)

    releaseList = project_obj.getReleases()

    testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)

    print("releases ", testReleaseList)

    for testReleaseObj in testReleaseList:
        print(testReleaseObj.getReleaseDate())
        train_changes = getFirstChangesBetween(project_obj, 0, testReleaseObj.getStartDate())

        performPredictionRunner(projectName, train_changes, testReleaseObj.getChanges(),
                                str(len(train_changes)),  testReleaseObj.getReleaseDate(), 'ALL')

def run_early(projectName):

    project_obj = data_manager.getProject(projectName)
    releaseList = project_obj.getReleases()
    testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)
    train_changes = project_obj.getAllChanges().head(150)
    early_changes = early_sample(train_changes)

    for testReleaseObj in testReleaseList:

        performPredictionRunner(projectName, early_changes.copy(deep=True), testReleaseObj.getChanges(),
                                str(len(train_changes)), testReleaseObj.getReleaseDate(), 'E')

def run_early_2(projectName):

    project_obj = data_manager.getProject(projectName)
    releaseList = project_obj.getReleases()
    testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)
    train_changes = project_obj.getAllChanges().head(150)
    early_changes = early_sample(train_changes)

    for testReleaseObj in testReleaseList:
        # projectName, originalTrainChanges, originalTestChanges, trainReleaseDate, testReleaseDate, trainApproach,
        # testReleaseStartDate = None, retainFew = None, tuneChanges = None, returnResults = False

        performPredictionRunner(projectName, early_changes.copy(deep=True), testReleaseObj.getChanges(),
                                str(len(train_changes)), testReleaseObj.getReleaseDate(), 'E+', None, 'size')

def run_early_bellwether_2(projectName):

    if projectName == BELLWETHER_PROJECT:
        return

    project_obj = data_manager.getProject(projectName)
    releaseList = project_obj.getReleases()
    testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)

    early_bell_changes = early_sample(getProject(BELLWETHER_PROJECT).getAllChanges().head(150))

    for testReleaseObj in testReleaseList:
        performPredictionRunner(projectName, early_bell_changes.copy(deep=True), testReleaseObj.getChanges(),
                                str(len(early_bell_changes)), testReleaseObj.getReleaseDate(), 'E_B_+' , None, 'size')


def removeExistingFiles():

    print("Attemping to remove existing results")

    for table in ['table4', 'table5']:
        for metric in METRICS_LIST:
            filetoremove = './output/' + table + '/z' + metric + '.txt'
            try:
                if path.exists(filetoremove):
                    try:
                        os.remove(filetoremove)
                        print('Existing ', filetoremove,  ' removed!')
                    except Exception as e:
                        print('[ERROR] : Unable to remove ',filetoremove, str(e))
            except:
                continue

def get_common_releases(df, samplingPolicies):

    releaseList = []


    for samplingPolicy in samplingPolicies:

        samplingReleaseList  = df[ df['trainApproach'] == samplingPolicy ]['testReleaseDate'].values.tolist()
        # print(samplingPolicy, len(samplingReleaseList))
        if samplingReleaseList is None or len(samplingReleaseList) == 0:
            return []
        else:
            releaseList.append(samplingReleaseList)

    testReleaseSet = None

    for releases in releaseList:

        if testReleaseSet is None:

            testReleaseSet = list(set(releases))
            continue

        else:

           testReleaseSet =  list( set(testReleaseSet) & set(releases) )



    return testReleaseSet


def to_pretty_label(selRule):
    return selRule


def collect_inputs_for_measures(metric, projectStartDateMap):

    for table in ['table4']:

        if table == 'table4':
            samplingPolicies = []
            samplingPolicies.append('B_scikit-learn')
            samplingPolicies.append('E_B_scikit-learn_2')

            samplingPolicies.append('T_scikit-learn')
            samplingPolicies.append('E_F2_T2_scikit-learn')

            samplingPolicies.append('E')
            samplingPolicies.append('E_2')

        elif table == 'table5':

            samplingPolicies = []

            for bellwether_project in   ['django-payments', 'restheart', 'apollo', 'sauce-java', 'portia', 'opendht', 'dogapi', 'midpoint',
             'active_merchant', 'zulip', 'woboq_codebrowser', 'pry']:

                samplingPolicies.append(bellwether_project+'_ALL_CHANGES')
                samplingPolicies.append(bellwether_project + '_E_CHANGES')

        f = open('./output/'+table+'/z' + metric  + '.txt', "a+")

        print("Generating ",'./output/'+table+'/z' + metric  + '.txt')

        for classifier in getSimpleNames():

            for selRule in  samplingPolicies:

                metricValues = []

                for p in getProjectNames():

                    df = pd.read_csv('./output/project_' + p + '_results.csv')
                    df = df[(df['test_changes'] > 5) & (df['d2h'] != 'CLF_ERROR')]

                    # sixmonths = projectStartDateMap[p] + (6 * one_month )
                    # df = df[ df['testReleaseDate'] > sixmonths ]

                    df = df[ df['classifier'] == classifier ]

                    commonReleases = get_common_releases(df, samplingPolicies)

                    if len(df) > 0:
                        sDF = df[ (df['testReleaseDate'].isin(commonReleases) ) & ( df['trainApproach'] == selRule ) ]
                    else:
                        continue

                    v = sDF[metric].values.tolist()

                    v = convert(v, metric)
                    metricValues += v


                f.write(to_pretty_label(selRule) + "_" + classifier + "\n")
                line = ''
                for c in metricValues:

                    line += str(c) + " "

                f.write(line.strip() + "\n\n")

def generate_scottknott_inputs():

    removeExistingFiles()

    projectStartDateMap = {}

    for p in getProjectNames():
        rrs = getProject(p).getReleases()
        projectStartDateMap[p] = min([r.getStartDate() for r in rrs])

    procs = []

    for metric in ['recall', 'pf', 'g-score', 'roc_auc', 'd2h', 'ifa', 'brier']:
        proc = Process(target=collect_inputs_for_measures(metric, projectStartDateMap), args=(metric,))
        procs.append(proc)
        proc.start()

    # Complete the processes
    for proc in procs:
        proc.join()


def generate_project_results():

    procs = []
    for name in getProjectNames():
        proc = Process(target=test_all_releases, args=(name,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()









