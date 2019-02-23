import sys
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
# from utils import *


class SubstanceAbuseDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file='HackTrain.csv', root_dir='./', transform: Transform = None, n_rows=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all images / other datasets.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # We want to load the data frame, but only keep relevant columns
        self.traffic_frame = pd.read_csv(os.path.join(str(Path(__file__).resolve().parents[1]), 'data', csv_file),
                                         nrows=n_rows)[JosiahAnalysis.FIXED_VARIABLES + JosiahAnalysis.DYNAMIC_VARIABLES
                                                       + JosiahAnalysis.DECISION_VARIABLES]

        # Remove rows that have nans in specific columns
        if all(c for c in JosiahAnalysis.HARD_NOT_NAN_VARIABLES if c in self.traffic_frame.columns):
            self.traffic_frame[JosiahAnalysis.HARD_NOT_NAN_VARIABLES].dropna(inplace=True)

        # One Hot Categorical Columns
        for column in JosiahAnalysis.CATEGORICAL_VARIABLES:
            one_hot_slice = pd.get_dummies(self.traffic_frame[column])
            one_hot_slice.columns = [column + '_' + str(c) for c in one_hot_slice.columns]
            self.traffic_frame = self.traffic_frame.join(one_hot_slice, how='left')
            self.traffic_frame.drop(labels=column, axis=1, inplace=True)
            # Update the decision, dynamic, and fixed variables to take all this into account
            if column in JosiahAnalysis.DECISION_VARIABLES:
                JosiahAnalysis.DECISION_VARIABLES += list(one_hot_slice.columns)
            if column in JosiahAnalysis.FIXED_VARIABLES:
                JosiahAnalysis.FIXED_VARIABLES += list(one_hot_slice.columns)
            if column in JosiahAnalysis.DYNAMIC_VARIABLES:
                JosiahAnalysis.DYNAMIC_VARIABLES += list(one_hot_slice.columns)

        # If there are still rows that are nan, drop them
        self.traffic_frame.dropna(inplace=True)

        # Lastly, verify each row, and normalize it if needed
        self.traffic_frame = self.traffic_frame.div(self.traffic_frame.max(axis=0), axis=1)

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
    # traffic_dataset = TrafficDataset('./traffic.csv', './', None, n_rows=100)
    #
    # for i in range(len(traffic_dataset)):
    #     sample = traffic_dataset[i]
    #
    #     if i > 5:
    #         break

    traffic_dataset = SubstanceAbuseDataset('./HackTrain.csv', './', Compose([]), n_rows=100)

    for i in range(len(traffic_dataset)):
        transformed_sample = traffic_dataset[i]
