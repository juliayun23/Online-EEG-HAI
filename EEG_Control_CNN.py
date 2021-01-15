import os
import sys

sys.path.insert(0, './/EEG//cyUSB//')
sys.path.insert(0, './/EEG')
sys.path.append('..\..')

import cyPyWinUSB as hid
import queue
from cyCrypto.Cipher import AES
from cyCrypto import Random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gumpy
from pynput.keyboard import Key, Controller
from random import randrange
import csv
import time 
import numpy as np
from numpy import linspace
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
import statistics 
from statistics import mode 
import pandas as pd
from scipy.interpolate import griddata
import time
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
import multiprocessing
from multiprocessing import Process
import kapre
from kapre.time_frequency import Spectrogram
from kapre.utils import Normalization2D
import keras
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout, Conv2D, MaxPooling2D #,LSTM
import keras.utils as ku
from keras.callbacks import ModelCheckpoint, CSVLogger
from datetime import datetime


DEBUG = 1
tasks = queue.Queue()
shared_queue = multiprocessing.Queue()
left_train_data = []
right_train_data = []
predictor_list = []
RESOLUTION = 256
CHANNEL = 14

ch_pos = [[1,4],[3,4],[0.2,3],[1.5,2.8],[2.5,2.8],[3.8,3],[0.5,2.5],[3.5,2.5],[-0.1,2],[4.1,2],[0.4,0.4],[3.6,0.4],[1.5,0],[2.5,0]]
pos_annotation = ['AF3','AF4','F7','F3','F4','F8','FC5','FC6','T7','T8','P7','P8','O1','O2']
n_samples, n_rows = 512, 14
chunk_size = 250
topomap_sample_size = 256
row_annotation = ['F3', 'FC5', 'AF3', 'F7','T7','P7','O1','O2','P8','T8','F8','AF4','FC6','F4']
flt_a = []
flt_b = []
flt_t = []
norm_data = []
fig = plt.figure(figsize=(6.5,5))
delimiter = ", "
line_start_idx = 0
cbar_updated = False
live_eeg_chunk = []
flt_lo, flt_hi = 1, 35
lo_t, lo_a, lo_b = 4, 8, 12
hi_t, hi_a, hi_b = 8, 12, 35

class EEG(object):
    
    def __init__(self):
        self.hid = None
        
        devicesUsed = 0
    
        for device in hid.find_all_hid_devices():
                if device.product_name == 'EEG Signals':
                    devicesUsed += 1
                    self.hid = device
                    self.hid.open()
                    self.serial_number = device.serial_number
                    device.set_raw_data_handler(self.dataHandler)                   
        if devicesUsed == 0:
            os._exit(0)
        sn = self.serial_number
        
        # EPOC+ in 16-bit Mode.
        k = ['\0'] * 16
        k = [sn[-1],sn[-2],sn[-2],sn[-3],sn[-3],sn[-3],sn[-2],sn[-4],sn[-1],sn[-4],sn[-2],sn[-2],sn[-4],sn[-4],sn[-2],sn[-1]]
        
        # EPOC+ in 14-bit Mode.
        #k = [sn[-1],00,sn[-2],21,sn[-3],00,sn[-4],12,sn[-3],00,sn[-2],68,sn[-1],00,sn[-2],88]
        
        self.key = str(''.join(k))
        self.cipher = AES.new(self.key.encode("utf8"), AES.MODE_ECB)

    def dataHandler(self, data):
        join_data = ''.join(map(chr, data[1:]))
        data = self.cipher.decrypt(bytes(join_data,'latin-1')[0:32])
        if str(data[1]) == "32": # No Gyro Data.
            return
        tasks.put(data)
        shared_queue.put(data)

    def convertEPOC_PLUS(self, value_1, value_2):
        edk_value = "%.8f" % (((int(value_1) * .128205128205129) + 4201.02564096001) + ((int(value_2) -128) * 32.82051289))
        return edk_value

    def get_data(self):
       
        data = tasks.get()
        #print(str(data[0])) COUNTER

        try:
            packet_data = ""
            for i in range(2,16,2):
                packet_data = packet_data + str(self.convertEPOC_PLUS(str(data[i]), str(data[i+1]))) + delimiter

            for i in range(18,len(data),2):
                packet_data = packet_data + str(self.convertEPOC_PLUS(str(data[i]), str(data[i+1]))) + delimiter

            packet_data = packet_data[:-len(delimiter)]
            return str(packet_data)

        except Exception as exception2:
            print(str(exception2))

            
class liveEEG_CNN():
    def __init__(self, x, y, n_classes = 2):
        self.print_version_info()

        # self.data_dir = data_dir
        self.cwd = os.getcwd()
        self.n_classes = n_classes
        kwargs = {'n_classes': self.n_classes}


        self.MODELNAME = "CNN_STFT"

        self.fs = 256
        # self.fs = self.data_notlive.sampling_freq
        self.lowcut = 2
        self.highcut = 60
        self.anti_drift = 0.5
        # self.f0 = 50.0  # freq to be removed from signal (Hz) for notch filter
        # self.Q = 30.0  # quality factor for notch filter
        # # w0 = f0 / (fs / 2)
        self.AXIS = 0
        self.CUTOFF = 50.0
        # self.w0 = self.CUTOFF / (self.fs / 2)
        self.dropout = 0.5

        self.x_train = x
        self.y_train = y
        if DEBUG:
            print("Shape of x_train: ", self.x_train.shape)
            print("Shape of y_train: ", self.y_train.shape)

        print("EEG Data loaded and processed successfully!")

        ### roll shape to match to the CNN
        self.x_rolled = np.rollaxis(self.x_train, 2, 1)
        # self.x_rolled = self.x_train
        # self.x_rolled = np.rollaxis(self.x_train, 1, 0)




        if DEBUG:
            print('X shape: ', self.x_train.shape)
            print('X rolled shape: ', self.x_rolled.shape)


        self.x_augmented_rolled = self.x_rolled
        self.y_augmented = self.y_train

        ### try to load the .json model file, otherwise build a new model
        self.loaded = 0
        if os.path.isfile(os.path.join(self.cwd,self.MODELNAME+".json")):
            self.load_CNN_model()
            if self.model:
                self.loaded = 1

        if self.loaded == 0:
            print("Could not load model, will build model.")
            self.build_CNN_model()
            if self.model:
                self.loaded = 1

        ### Create callbacks for saving
        saved_model_name = self.MODELNAME
        TMP_NAME = self.MODELNAME + "_" + "_C" + str(self.n_classes)
        for i in range(99):
            if os.path.isfile(saved_model_name + ".csv"):
                saved_model_name = TMP_NAME + "_run{0}".format(i)

        ### Save model -> json file
        json_string = self.model.to_json()
        model_file = saved_model_name + ".json"
        open(model_file, 'w').write(json_string)

        ### define where to save the parameters to
        model_file = saved_model_name + 'monitoring' + '.h5'
        checkpoint = ModelCheckpoint(model_file, monitor='val_loss',
                                     verbose=1, save_best_only=True, mode='min')
        log_file = saved_model_name + '.csv'
        csv_logger = CSVLogger(log_file, append=True, separator=';')
        self.callbacks_list = [csv_logger, checkpoint]  # callback list


###############################################################################
    def load_CNN_model(self):
        print('Load model', self.MODELNAME)
        model_path = self.MODELNAME + ".json"
        if not os.path.isfile(model_path):
            raise IOError('file "%s" does not exist' % (model_path))
        self.model = model_from_json(open(model_path).read(),custom_objects={'Spectrogram': kapre.time_frequency.Spectrogram,
                                     'Normalization2D': kapre.utils.Normalization2D})
        #self.model = load_model(self.cwd,self.MODELNAME,self.MODELNAME+'monitoring')
        #TODO: get it to work, but not urgently required
        #self.model = []



###############################################################################
    def build_CNN_model(self):
        ### define CNN architecture
        print('Build model...')
        print('input shape:', self.x_augmented_rolled.shape)

        self.model = Sequential()
        self.model.add(Spectrogram(n_dft=128, n_hop=16, input_shape=(self.x_augmented_rolled.shape[1:]),
                              return_decibel_spectrogram=False, power_spectrogram=2.0,
                              trainable_kernel=False, name='static_stft'))
        self.model.add(Normalization2D(str_axis = 'freq'))
        
        # Conv Block 1
        self.model.add(Conv2D(filters = 24, kernel_size = (12, 12),
                         strides = (1, 1), name = 'conv1',
                         padding = 'same'))
        self.model.add(BatchNormalization(axis = 1))
        self.model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2), padding = 'valid',
                               data_format = 'channels_last'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.dropout))

        # Conv Block 2
        self.model.add(Conv2D(filters = 48, kernel_size = (8, 8),
                         name = 'conv2', padding = 'same'))
        self.model.add(BatchNormalization(axis = 1))
        self.model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid',
                               data_format = 'channels_last'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.dropout))

        # Conv Block 3
        self.model.add(Conv2D(filters = 96, kernel_size = (4, 4),
                         name = 'conv3', padding = 'same'))
        self.model.add(BatchNormalization(axis = 1))
        self.model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2),
                               padding = 'valid',
                               data_format = 'channels_last'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.dropout))

        # classificator
        self.model.add(Flatten())
        self.model.add(Dense(self.n_classes))  # two classes only
        self.model.add(Activation('softmax'))

        print(self.model.summary())
        self.saved_model_name = self.MODELNAME



###############################################################################
    def print_version_info(self):
        now = datetime.now()

        print('%s/%s/%s' % (now.year, now.month, now.day))
        print('Keras version: {}'.format(keras.__version__))
        if keras.backend.backend() == 'tensorflow':
            import tensorflow
            print('Keras backend: {}: {}'.format(keras.backend.backend(), tensorflow.__version__))
        else:
            import theano
            print('Keras backend: {}: {}'.format(keras.backend.backend(), theano.__version__))
        print('Keras image dim ordering: {}'.format(keras.backend.image_data_format()))
        print('Kapre version: {}'.format(kapre.__version__))

###############################################################################
    ### train the model with the notlive data or sinmply load a pretrained model
    def fit(self, load=False):
        #TODO: use method train_on_batch() to update model
        self.batch_size = 32
        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

        if not load:
            print('Train...')
            self.model.fit(self.x_augmented_rolled, self.y_augmented,
                 batch_size=self.batch_size,
                  # epochs=100,
                 epochs=80,
                 shuffle=True,
                 validation_split=0.2,
                 callbacks=self.callbacks_list)
        else:
            print('Load...')
            self.model = keras.models.load_model('CNN_STFTmonitoring.h5',
                                     custom_objects={'Spectrogram': kapre.time_frequency.Spectrogram,
                                     'Normalization2D': kapre.utils.Normalization2D})

        #CNN_STFT__C2_run4monitoring.h5

###############################################################################
 ### do the live classification
    def classify_live(self, x):                    
        self.x_live = x
        self.x_live = np.rollaxis(self.x_live, 2, 1)
        
        print(datetime.now(),"=================predict==================")
        # print("Shape of x_live: ", self.x_live.shape)

        ### do the prediction
        y_pred = np.zeros(self.x_live.shape[0])
        if self.loaded and self.x_live.any():
            pred = self.model.predict(self.x_live,batch_size=64)
            print('prediction details: ', pred)

            #classes = self.model.predict(self.x_live_augmented,batch_size=64)
            #pref0 = sum(classes[:,0])
            #pref1 = sum(classes[:,1])
            #if pref1 > pref0:
            #    y_pred = 1
            #else:
            #    y_pred = 0

            ### argmax because output is crossentropy
            for i in range(len(pred)):
                y_pred[i] = np.argmax(pred[i])
            #y_pred = np.argmax(y_pred)
            # print('y_pred: ', y_pred)
        
        return  y_pred
    


def keyboard_control(decision):
    # print("=================keyboard_control {}=================".format(decision))
    keyboard = Controller()
    SLEEP_TIME = 0.1 
    if decision == 0 :
        keyboard.press(Key.up)
        keyboard.press(Key.left)
        time.sleep(SLEEP_TIME)
        keyboard.release(Key.left)
        keyboard.release(Key.up)
    elif decision == 1:
        keyboard.press(Key.up)
        keyboard.press(Key.right)
        time.sleep(SLEEP_TIME)
        keyboard.release(Key.right)
        keyboard.release(Key.up)
    # elif decision == 2:
    #     keyboard.press(Key.up)
    #     time.sleep(0.5)
    #     keyboard.release(Key.up)
    keyboard.press(Key.up)
    time.sleep(0.5)
    keyboard.release(Key.up)



def train_cnn():
    print("=================start training=================")

    # ======load dataset========
    data_base_dir = './data'
    graza_base_dir = os.path.join(data_base_dir, 'graz-2a')
    subject = 'A01'
    graza_data = gumpy.data.GrazA(graza_base_dir, subject)
    graza_data.load()
    graza_data.print_stats() 
    
    # ======extract trails, only extract data for class1 and class2========    
    class1_mat, class2_mat = gumpy.utils.extract_trials2(graza_data.raw_data, graza_data.trials, graza_data.labels, graza_data.trial_total, graza_data.sampling_freq, nbClasses = 2)
    X_ori = np.concatenate((class1_mat, class2_mat))
    X = np.reshape(X_ori, (X_ori.shape[0]*X_ori.shape[1], X_ori.shape[2]))
    labels_c1 = np.zeros((class1_mat.shape[0],))
    labels_c2 = np.ones((class2_mat.shape[0],))
    y = np.concatenate((labels_c1, labels_c2))
    
    # ======preprocess: filter and normalize========
    X = gumpy.signal.butter_bandpass(X, lo=flt_lo, hi=flt_hi)
    print("filtered X: ", X)
    X = gumpy.signal.normalize(X, 'mean_std')
    print("normalized X: ", X)
    print("""Normalized Data:
      Mean    = {:.3f}
      Min     = {:.3f}
      Max     = {:.3f}
      Std.Dev = {:.3f}""".format(
      np.nanmean(X),np.nanmin(X),np.nanmax(X),np.nanstd(X)
    ))
    
    # ======PCA reduct dimension from 22 to 14========
    pca = PCA(n_components=14)
    pca.fit(X)
    X_pca = pca.transform(X)

    # ======train CNN model========
    X = np.reshape(X_pca, (X_ori.shape[0],X_ori.shape[1],14))
    y = ku.to_categorical(y)
    eeg_cnn = liveEEG_CNN(X,y)
    eeg_cnn.print_version_info()
    eeg_cnn.fit(load = True)
    return eeg_cnn
    
    
def real_time_predict(predictor):
    print("=================start prediction=================")
    # cyHeadset = EEG()
    ACTION = ['LEFT','RIGHT','UP']
    while 1:
        time.sleep(0.5)
        
        while not tasks.empty() and len(live_eeg_chunk) < chunk_size:
            signal_str = cyHeadset.get_data().split(",")
            signal = [float(i) for i in signal_str]
            live_eeg_chunk.append(signal)
            # print("live chunk size:", len(live_eeg_chunk))

    
        # when enough data are collected, preprocess the data and make prediction
        if len(live_eeg_chunk)== chunk_size:
            X = np.array(live_eeg_chunk)
            # =====preprocess: filter and normalize========
            X = gumpy.signal.butter_bandpass(X, lo=flt_lo, hi=flt_hi)
            # print("filtered X: ", X)
            X = gumpy.signal.normalize(X, 'mean_std')
            # print("normalized X: ", X)
            # =====expand dimnesion from 2000*14 to 1*2000*14=====
            X = np.expand_dims(X, axis=0)
            # =====real-time predict========
            # print("real_time_predict X shape: ", X.shape)
            result = predictor.classify_live(X)[0]
            print("{: <5} - {}".format(ACTION[int(result)],X))
            live_eeg_chunk.clear()
            # =====send control signal========
            keyboard_control(int(result))
        else:
            keyboard_control(2)

        # keyboard_control(int(result))
        #real_time_visualize(signal)
  

    
def real_time_visualize(queue):
    
    def plot_topomap(data, ax, fig):
        '''
        Plot topographic plot of EEG data. This specialy design for Emotiv 14 electrode data. 
        This can be change for any other arrangement by changing ch_pos (channel position array)
        Input: data- 1D array 14 data values
               ax- Matplotlib subplot object to be plotted every thing
               fig- Matplot lib figure object to draw colormap
        '''
        global ch_pos, pos_annotation
        N = 300            
        xy_center = [2,2]  
        radius = 2 
    
        x,y = [],[]
        for i in ch_pos:
            x.append(i[0])
            y.append(i[1])
    
        xi = linspace(-2, 6, N)
        yi = linspace(-2, 6, N)
        zi = griddata((x, y), data, (xi[None,:], yi[:,None]), method='cubic')
    
        dr = xi[1] - xi[0]
        for i in range(N):
            for j in range(N):
                r = np.sqrt((xi[i] - xy_center[0])**2 + (yi[j] - xy_center[1])**2)
                if (r - dr/2) > radius:
                    zi[j,i] = "nan"
        
        dist = ax.contourf(xi, yi, zi, 60, cmap = plt.cm.jet, zorder = 1)
        ax.contour(xi, yi, zi, 15, linewidths = 0.5,colors = "grey", zorder = 2)
    
        ax.scatter(x, y, marker = 'o', c = 'b', s = 15, zorder = 3)
        for i in range(len(ch_pos)):
            ax.annotate(pos_annotation[i], (x[i]+0.1, y[i]))
        circle = matplotlib.patches.Circle(xy = xy_center, radius = radius, edgecolor = "k", facecolor = "none", zorder=4)
        ax.add_patch(circle)
    
        for loc, spine in ax.spines.items():
            spine.set_linewidth(0)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
        circle = matplotlib.patches.Ellipse(xy = [0,2], width = 0.4, height = 1.0, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
        ax.add_patch(circle)
        circle = matplotlib.patches.Ellipse(xy = [4,2], width = 0.4, height = 1.0, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
        ax.add_patch(circle)
        
        xy = [[1.6,3.6], [2,4.3],[2.4,3.6]]
        polygon = matplotlib.patches.Polygon(xy = xy, edgecolor = "k", facecolor = "w", zorder = 0)
        ax.add_patch(polygon) 
        
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 4.5)
    
        return ax, dist
    
    def plot_signals(data, ax, fig):
        # print("in plot_signals, data.shape:", data.shape)
        t = 10 * np.arange(n_samples) / n_samples
        ticklocs = []
        # ax.set_xlim(0, 10)
        ax.set_xticks(np.arange(3))
        dmin = data.min()
        dmax = data.max()
        dr = (dmax - dmin) * 0.7  # Crowd them a bit.
        y0 = dmin
        y1 = (n_rows - 1) * dr + dmax
        ax.set_ylim(y0, y1)
        
        segs = []
        for i in range(n_rows):
            segs.append(np.column_stack((t, data[:, i])))
            ticklocs.append(i * dr)
        
        offsets = np.zeros((n_rows, 2), dtype=float)
        offsets[:, 1] = ticklocs
        
        lines = LineCollection(segs, offsets=offsets, transOffset=None)
        ax.add_collection(lines)
        
        ax.set_yticks(ticklocs)
        ax.set_yticklabels(row_annotation)
    
        ax.set_xlabel('Timelapse (s)')
        
        return ax
    
    def convertEPOC_PLUS(value_1, value_2):
        edk_value = "%.8f" % (((int(value_1) * .128205128205129) + 4201.02564096001) + ((int(value_2) -128) * 32.82051289))
        return edk_value
    
    def get_data():
       
        data = queue.get()

        try:
            packet_data = ""
            for i in range(2,16,2):
                packet_data = packet_data + str(convertEPOC_PLUS(str(data[i]), str(data[i+1]))) + delimiter

            for i in range(18,len(data),2):
                packet_data = packet_data + str(convertEPOC_PLUS(str(data[i]), str(data[i+1]))) + delimiter

            packet_data = packet_data[:-len(delimiter)]
            return str(packet_data)

        except Exception as exception2:
            print(str(exception2))
    
    def init_ani():
        print("in init:")

    def animate(i):
        #print(datetime.now(), "---in animate:", i)
        global line_start_idx, cbar_updated
        
        # print("shared queue size: ", queue.qsize())
        # print("eeg_chunk size: ", len(eeg_chunk))
        
        while not queue.empty():
            signal_str = get_data().split(",")
            signal = [float(i) for i in signal_str]
            eeg_chunk.append(signal)
            signal = gumpy.signal.normalize(signal, 'mean_std')
            norm_data.append(signal)
    
        # filter the data when enough data are collected
        if len(eeg_chunk)>= topomap_sample_size:
            for item in gumpy.signal.butter_bandpass(eeg_chunk, lo=lo_a, hi=hi_a): 
                flt_a.append(item)
            for item in gumpy.signal.butter_bandpass(eeg_chunk, lo=lo_b, hi=hi_b): 
                flt_b.append(item)
            for item in gumpy.signal.butter_bandpass(eeg_chunk, lo=lo_t, hi=hi_t): 
                flt_t.append(item)
            eeg_chunk.clear()
            # print("flt_a size:", len(flt_a))
        
        if len(flt_a)>i+1:
        
            ax1.clear()
            ax2.clear()
            ax3.clear()
            
            ax1.set_title('Theta Band')
            ax2.set_title('Alpha Band')
            ax3.set_title('Beta Band')
            _,dist1 = plot_topomap(flt_t[i], ax1, fig)
            _,dist2 = plot_topomap(flt_a[i], ax2, fig)
            _,dist3 = plot_topomap(flt_b[i], ax3, fig)
            if not cbar_updated:
                cbar = fig.colorbar(dist3, ax=[ax1,ax2,ax3], format='%.1e')
                cbar.ax.tick_params(labelsize=8)
                cbar_updated = True
            
        if len(norm_data) >= line_start_idx+n_samples:
            ax4.clear()
            norm_arr = np.array(norm_data)
            # print('norm_arr shape: ',norm_arr.shape)
            plot_signals(norm_arr[line_start_idx:line_start_idx+n_samples,:], ax4, fig)
            ax4.set_xticklabels(np.arange(line_start_idx, line_start_idx+1.5*n_samples, n_samples/2)/RESOLUTION)

            line_start_idx += int((0.5)*n_samples)

        
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    
    

    eeg_chunk = []
    
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    
    for ax in (ax1, ax2, ax3):
        ax.set_aspect('equal')
    plot_topomap(np.zeros(n_rows), ax1, fig)
    plot_topomap(np.zeros(n_rows), ax2, fig)
    plot_topomap(np.zeros(n_rows), ax3, fig)
   
    ax1.set_title('Theta Band')
    ax2.set_title('Alpha Band')
    ax3.set_title('Beta Band')
    
    # Plot the EEG signals
    ax4 = fig.add_subplot(212)
    data = np.zeros((n_samples, n_rows))
    plot_signals(data, ax4, fig)
    
    ani = FuncAnimation(fig, animate, init_func=init_ani, interval=200)
    
    # record the animation
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save('eeg_visualization1.mp4', writer=writer)
    
    
    # plt.tight_layout()
    plt.show()


    
cyHeadset = EEG()

def MainProgram():
    # cyHeadset = EEG()
    model = train_cnn()
    print('Main program')
    real_time_predict(model)

if __name__ == '__main__':
    p = Process(target=real_time_visualize, args=((shared_queue),))
    p.start()
    MainProgram()
    p.join()