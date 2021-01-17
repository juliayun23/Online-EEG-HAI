# Online-EEG-HAI
This is the source code for an online Human-Agent Interaction (HAI) system, controlling directions (left or right) of Unity games using online EEG signals from the Emotiv EPOC+ headset


## Setup
Hardware equipments required:
1. A Windows computer/laptop installed with [Unity](https://unity.com/) (The reading of EEG raw data from the headset is only supported by the Windows OS)
2. An [Emotiv Epoc+](https://www.emotiv.com/epoc/) Headset

Software environment:

The project supports Python3. The recommended version is Python 3.6.11.

Anaconda is recommended for managing the environment.


## Installation
1. Download the project folder.

    The EEG folder contains some libraries needed for this project.

    The gumpy folder is another library we use for the EEG signal pre-processing (modified from [gumpy](https://github.com/gumpy-bci/gumpy)). 

    The data folder contains the training IM dataset from [BCI Competition IV-2a](http://www.bbci.de/competition/iv/#dataset2a).

    The three files starting with "CNN_" are the files for the pretrained classification model.

2. Navigate to the gumpy folder. Execute ```pip install . ``` to install the gumpy library.

3. The project also requires the [kapre](https://pypi.org/project/kapre/) library. Install by executing ```pip install kapre==0.1.3 ```

4. Put on the Emotiv Epoc+ headset, make sure it's turned on and connected to the Windows computer/laptop. Make sure the contact quality is good. Open a Unity game on Windows, such as the Karting game.

5. Execute the EEG_Control_CNN.py ```python EEG_Control_CNN.py``` to control the Unity game using real-time EEG signals from the headset. By default, it's using the pretrained classification model. 

    If you want to train your own model using the training dataset, you can modify [here](https://github.com/nomatterhoe/Online-EEG-HAI/blob/main/EEG_Control_CNN.py#L437) by changing the "load" parameter to ```load = False``` 

