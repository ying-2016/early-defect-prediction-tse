import math



"""
@author : Shrikanth N C (nc.shrikanth@gmail.com)
"""

"""
period constants
"""
one_year = 31556926
one_month = 2629743
one_day = 86400

CONSIDER_FIRST_X_RELEASES = math.inf

E_TRAIN_LIMIT = 150

METRICS_LIST = ['f1','precision' , 'recall', 'pf', 'g-score', 'd2h', 'pci_20',  'ifa', 'ifa_roc', 'roc_auc', 'pd', 'tp', 'tn', 'fp', 'fn', 'negopos', 'balance', 'brier', 'popt20']

NUM_RELEASES_TO_CONSIDER = math.inf
MIN_NUM_FILES_PER_RELEASE = 2
MAX_NUM_FILES_PER_RELEASE =  math.inf

SAMPLE_LIMIT = math.inf
DEBUG = False
UNIT_TEST = False

Dummy_Flag = False
RESULTS_FOLDER = 'output'

BELLWETHER_PROJECT = 'scikit-learn'

BELLWETHER_PROJECTS = ['django-payments', 'restheart', 'apollo', 'sauce-java', 'portia', 'opendht', 'dogapi', 'midpoint',
         'active_merchant', 'zulip', 'woboq_codebrowser', 'pry']


