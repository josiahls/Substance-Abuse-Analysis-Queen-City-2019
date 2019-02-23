import sys

from sklearn.impute import SimpleImputer

sys.path.append('../')

from torch.distributions.transforms import Transform
from torch.utils.data import Dataset
import pandas as pd
import os
from src.data.Analysis import JosiahAnalysis
from src.data.Transforms import *
from torchvision.transforms import Compose
import numpy as np
from pathlib import Path

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# from utils import *


class SubstanceAbuseDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file='HackTrain.csv', root_dir='./', transform: Transform = None, n_rows=None,
                 master_columns=None, dataframe=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all images / other datasets.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        if dataframe is None:
            self.traffic_frame = pd.read_csv(os.path.join(str(Path(__file__).resolve().parents[1]), 'data', csv_file),
                                             nrows=n_rows)
            # TODO Parker Randomize here ^^^^
        else:
            self.traffic_frame = dataframe
        self.raw_frame = pd.DataFrame.copy(self.traffic_frame)

        # Columns to keep
        keep_columns = JosiahAnalysis.CATEGORICAL_VARIABLES + JosiahAnalysis.CONTINUOUS_VARIABLES + \
                       JosiahAnalysis.DECISION_VARIABLES + JosiahAnalysis.INDEX_VARIABLES
        keep_columns = [c for c in keep_columns if c in self.traffic_frame.columns]
        # self.index_frame = self.traffic_frame[keep_columns + JosiahAnalysis.INDEX_VARIABLES]
        self.traffic_frame = self.traffic_frame[keep_columns]

        # # Remove rows that have nans in specific columns
        # if all(c in self.traffic_frame.columns for c in JosiahAnalysis.HARD_NOT_NAN_VARIABLES):
        #     self.traffic_frame[JosiahAnalysis.HARD_NOT_NAN_VARIABLES].dropna(inplace=True)

        # # Impute the -9 Values
        # normalize_columns = [c for c in self.traffic_frame.columns if c not in JosiahAnalysis.INDEX_VARIABLES]
        # imp_mean = SimpleImputer(missing_values=-9, strategy='mean')
        # imp_mean.fit(self.traffic_frame[normalize_columns])
        # self.traffic_frame[normalize_columns] = imp_mean.transform(self.traffic_frame[normalize_columns])

        # One Hot Categorical Columns
        accum_categoricals = []
        for column in [_ for _ in JosiahAnalysis.CATEGORICAL_VARIABLES if _ in self.traffic_frame]:
            one_hot_slice = pd.get_dummies(self.traffic_frame[column])
            one_hot_slice.columns = [column + '_' + str(c) for c in one_hot_slice.columns]
            self.traffic_frame = self.traffic_frame.join(one_hot_slice, how='left')
            self.traffic_frame.drop(labels=column, axis=1, inplace=True)
            # Update the decision, dynamic, and fixed variables to take all this into account
            if column not in JosiahAnalysis.DECISION_VARIABLES:
                JosiahAnalysis.DECISION_VARIABLES += list(one_hot_slice.columns)
            if column not in JosiahAnalysis.CATEGORICAL_VARIABLES:
                accum_categoricals += list(one_hot_slice.columns)
            if column not in JosiahAnalysis.CONTINUOUS_VARIABLES:
                JosiahAnalysis.CONTINUOUS_VARIABLES += list(one_hot_slice.columns)

        # If there are still rows that are nan, drop them
        self.traffic_frame.dropna(inplace=True)

        # Same the max value for all the columns for easy re-translation
        self.max_value_key = self.traffic_frame.max(axis=0)

        normalize_columns = [c for c in self.traffic_frame.columns if c not in JosiahAnalysis.INDEX_VARIABLES]
        # Lastly, verify each row, and normalize it if needed
        self.traffic_frame[normalize_columns] = self.traffic_frame[normalize_columns] \
            .div(self.traffic_frame[normalize_columns].max(axis=0), axis=1)

        if master_columns is not None:
            # Add missing columns as zeros
            for c in [c for c in master_columns if c not in self.traffic_frame.columns]:
                self.traffic_frame.assign(Name=c)
                self.traffic_frame[c] = 0

        # Sort the columns so that their values match other frame loads
        self.traffic_frame = self.traffic_frame.reindex(sorted(self.traffic_frame.columns), axis=1)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.traffic_frame)

    def __getitem__(self, idx: int):
        sample = self.traffic_frame.iloc[int(idx)].to_dict()

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    traffic_dataset = SubstanceAbuseDataset('./HackTrain.csv', './', Compose([]), n_rows=100)

    for i in range(len(traffic_dataset)):
        transformed_sample = traffic_dataset[i]
