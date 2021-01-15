# import for convenience. Individual dataset implementatios are kept separate as
# they may require several subroutines that otherwise clutter the namespace
from .dataset import Dataset
from .nst import NST
from .graz import GrazB
from .khushaba import Khushaba
from .nst_emg import NST_EMG
from .eeg_data import EEGData
from .graz_2a import GrazA

