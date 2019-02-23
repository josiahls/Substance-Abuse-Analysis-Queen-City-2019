import pandas as pd
import numpy as np

"""
We need to label each column based on these 3:

Fixed Variables    Variables that cannot be changed.
Dynamic Variables  Variables that can be changed.
Decision Variables Variables whose values we would like to minimize.

"""

class ParkerAnalysis:
    IGNORED_VARIABLES = ['YEAR', 'CASEID', 'CBSA', 'STFIPS']
    INDEX_VARIABLES = ['CASEID']
    DECISION_VARIABLES = ['REASON', 'LOS']

    """ Cleaning VARIABLES """
    CATEGORICAL_VARIABLES = ['RACE', 'MARSTAT', 'LIVARAG', 'PRIMINC', 'REGION', 'DIVISION', 'SERVSETD', 'DETCRIM', 'SUB1', 'ROUTE1', 'SUB2', 'ROUTE2', 'SUB3', 'ROUTE3', 'IDU', 'ALCDRUG', 'DSMCRIT', 'HLTHINS', 'PRIMPAY']
    CONTINUOUS_AND_1HOT_VARIABLES = {'AGE': ['12'], 'ARRESTS': ['2'], 'DAYWAIT': ['X'], 'NOPRIOR': ['5'] } # the value of 12 in age needs an additional 1hot column
    CONTINUOUS_VARIABLES = ['EDUC', 'EMPLOY', 'DETNLF', 'FREQ1', 'FRSTUSE1', 'FREQ2', 'FRSTUSE2', 'FREQ3', 'FRSTUSE3', 'NUMSUBS']
    CONTINUOUS_VARIABLES += [e for e in CONTINUOUS_AND_1HOT_VARIABLES]
    BOOLEAN_VARIABLES = ['SEX', 'PREG', 'VET', 'METHUSE', 'ALCFLG', 'COKEFLG', 'MARFLG', 'HERFLG', 'METHFLG', 'OPSYNFLG', 'PCPFLG', 'HALLFLG', 'MTHAMFLG', 'AMPHFLG', 'STIMFLG', 'BENZFLG', 'TRNQFLG', 'BARBFLG', 'SEDHPFLG', 'INHFLG', 'OTCFLG', 'OTHERFLG', 'PSYPROB'] # exactly 2 values that need to be condensed to one
    CONVERT_TO_BOOLEAN_VARIABLES = {'ETHNIC': 5}
    CATEGORICAL_VARIABLES += [e for e in CONVERT_TO_BOOLEAN_VARIABLES]
    #CUSTOM_VARIABLES = ['DSMCRIT'] # these will be passed to a function with the name of the variable for cleaning
    # geo variable: STFIPS, CBSA
    # look at EMPLOY/DETNLF

    # ignored STFIPS, CBSA
    # further break down: SERVSETD
    # LOS needs to be changed to the correct day range
    # more frequency: FREQ1, FREQ2, FREQ3
    # FRSTUSE1 is a continuous and 1 hot, but ignore the 0.2%
    # ALCDRUG should be split
    # DSMCRIT could be split

class JosiahAnalysis:
    IGNORED_VARIABLES = ['YEAR', 'CASEID', 'CBSA']
    INDEX_VARIABLES = ['CASEID']
    CATEGORICAL_VARIABLES = ['SERVSETA', 'DISYR', 'AGE', 'GENDER', 'RACE', 'ETHNIC', 'MARSTAT', 'EDUC', 'EMPLOY', 'DETNLF', 'PREG', 'VET', 'LIVARAG', 'PRIMINC', 'ARRESTS', 'STFIPS', 'REGION', 'DIVISION', 'SERVSETD', 'METHUSE', 'DAYWAIT', 'PSOURCE', 'DETCRIM', 'NOPRIOR', 'SUB1', 'ROUTE1', 'FREQ1', 'FRSTUSE1', 'SUB2', 'ROUTE2', 'FREQ2', 'FRSTUSE2', 'SUB3', 'ROUTE3', 'FREQ3', 'FRSTUSE3', 'NUMSUBS', 'IDU', 'ALCFLG', 'COKEFLG', 'MARFLG', 'HERFLG', 'METHFLG', 'OPSYNFLG', 'PCPFLG', 'HALLFLG', 'MTHAMFLG', 'AMPHFLG', 'STIMFLG', 'BENZFLG', 'TRNQFLG', 'BARBFLG', 'SEDHPFLG', 'INHFLG', 'OTCFLG', 'OTHERFLG', 'ALCDRUG', 'DSMCRIT', 'PSYPROB', 'HLTHINS', 'PRIMPAY']
    CONTINUOUS_VARIABLES = []
    DECISION_VARIABLES = ['REASON', 'LOS']

if __name__ == '__main__':
    data = pd.read_csv('./HackTrain.csv', nrows=200)
    columns = data.columns

    columns_and_values = np.hstack(
        (np.array(columns).reshape((-1, 1)), np.array([data[c].unique()[:] for c in columns]).reshape(-1, 1)))

    print(columns_and_values)






