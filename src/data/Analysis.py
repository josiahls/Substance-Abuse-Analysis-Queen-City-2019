import pandas as pd
import numpy as np

"""
We need to label each column based on these 3:

Fixed Variables    Variables that cannot be changed.
Dynamic Variables  Variables that can be changed.
Decision Variables Variables whose values we would like to minimize.

"""

class ParkerAnalysis:
    IGNORED_VARIABLES = ['YEAR', 'CASEID']
    FIXED_VARIABLES = []
    DYNAMIC_VARIABLES = []
    DECISION_VARIABLES = []
    """ Cleaning VARIABLES """
    CATEGORICAL_VARIABLES = []
    BOOLEAN_VARIABLES = []
    HARD_NOT_NAN_VARIABLES = [] + DECISION_VARIABLES
    MILT_TIME_VARIABLES = []

class JosiahAnalysis:
    IGNORED_VARIABLES = ['YEAR', 'CASEID', 'CBSA']
    FIXED_VARIABLES = ['SERVSETA', 'DISYR', 'AGE', 'GENDER', 'RACE', 'ETHNIC', 'MARSTAT', 'EDUC', 'EMPLOY', 'DETNLF', 'PREG', 'VET', 'LIVARAG', 'PRIMINC', 'ARRESTS', 'STFIPS', 'REGION', 'DIVISION', 'SERVSETD', 'METHUSE', 'DAYWAIT', 'PSOURCE', 'DETCRIM', 'NOPRIOR', 'SUB1', 'ROUTE1', 'FREQ1', 'FRSTUSE1', 'SUB2', 'ROUTE2', 'FREQ2', 'FRSTUSE2', 'SUB3', 'ROUTE3', 'FREQ3', 'FRSTUSE3', 'NUMSUBS', 'IDU', 'ALCFLG', 'COKEFLG', 'MARFLG', 'HERFLG', 'METHFLG', 'OPSYNFLG', 'PCPFLG', 'HALLFLG', 'MTHAMFLG', 'AMPHFLG', 'STIMFLG', 'BENZFLG', 'TRNQFLG', 'BARBFLG', 'SEDHPFLG', 'INHFLG', 'OTCFLG', 'OTHERFLG', 'ALCDRUG', 'DSMCRIT', 'PSYPROB', 'HLTHINS', 'PRIMPAY']
    DYNAMIC_VARIABLES = []
    DECISION_VARIABLES = ['REASON', 'LOS']
    """ Cleaning VARIABLES """
    CATEGORICAL_VARIABLES = ['SERVSETA', 'DISYR', 'AGE', 'GENDER', 'RACE', 'ETHNIC', 'MARSTAT', 'EDUC', 'EMPLOY', 'DETNLF', 'PREG', 'VET', 'LIVARAG', 'PRIMINC', 'ARRESTS', 'STFIPS', 'REGION', 'DIVISION', 'SERVSETD', 'METHUSE', 'DAYWAIT', 'PSOURCE', 'DETCRIM', 'NOPRIOR', 'SUB1', 'ROUTE1', 'FREQ1', 'FRSTUSE1', 'SUB2', 'ROUTE2', 'FREQ2', 'FRSTUSE2', 'SUB3', 'ROUTE3', 'FREQ3', 'FRSTUSE3', 'NUMSUBS', 'IDU', 'ALCFLG', 'COKEFLG', 'MARFLG', 'HERFLG', 'METHFLG', 'OPSYNFLG', 'PCPFLG', 'HALLFLG', 'MTHAMFLG', 'AMPHFLG', 'STIMFLG', 'BENZFLG', 'TRNQFLG', 'BARBFLG', 'SEDHPFLG', 'INHFLG', 'OTCFLG', 'OTHERFLG', 'ALCDRUG', 'DSMCRIT', 'PSYPROB', 'HLTHINS', 'PRIMPAY']
    BOOLEAN_VARIABLES = []
    HARD_NOT_NAN_VARIABLES = [] + DECISION_VARIABLES

if __name__ == '__main__':
    data = pd.read_csv('./HackTrain.csv', nrows=200)
    columns = data.columns

    columns_and_values = np.hstack(
        (np.array(columns).reshape((-1, 1)), np.array([data[c].unique()[:] for c in columns]).reshape(-1, 1)))

    print(columns_and_values)






