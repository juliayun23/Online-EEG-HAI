#Data class for EEG headset (Trials)

from .dataset import Dataset, DatasetError
import os
import numpy as np
import scipy.io
import pandas as pd
from sklearn.model_selection import train_test_split


class EEGData(Dataset):

    def __init__(self, base_dir, identifier, **kwargs):

        super(EEGData, self).__init__(**kwargs)

        self.base_dir = base_dir
        self.data_id = identifier
        self.data_dir = base_dir
        self.data_type = 'EEG'
        self.data_name = 'EEGHeadset'

        # length of a trial (in seconds)
        self.trial_len = 1
        # motor imagery appears in interval (in seconds)
        self.mi_interval = [4, 7]
        # idle perior prior to start of signal (in seconds)
        self.trial_offset = 0
        # total length of a trial (in seconds)
        self.trial_total = self.trial_len
        # sampling frequency (in Hz)
        self.expected_freq_s = 256

        #training
        self.fT = os.path.join(self.data_dir, "ACTION_T.csv")
        #evaluate
        self.fE = os.path.join(self.data_dir, "ACTION_E.csv")

        for f in [self.fT, self.fE]:
            if not os.path.isfile(f):
                raise DatasetError("EEGData Dataset file '{f}' unavailable".format(f=f))

        # variables to store data
        self.raw_data = None
        self.labels = None
        self.trials = None
        self.sampling_freq = None


    def load(self, **kwargs):
        data_bt = []
        trials_bt = []
        
        df = pd.read_csv(self.fT)
        # training_data, testing_data = train_test_split(df, test_size = 0.2, shuffle = False)

        # data_bt = training_data.drop(columns=['ACTION'])
        #actions = df['ACTION'].values

        self.raw_data = df.drop(columns=['ACTION']).to_numpy()
        self.trials = df.index.values[::256]
        self.labels = df['ACTION'].values[::256]-1
        self.sampling_freq = self.expected_freq_s

        return self

