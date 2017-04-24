
# coding: utf-8

# In[33]:

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.time_frequency import tfr_morlet, psd_multitaper
from mne.datasets import somato


# In[38]:

raw_fname='./a_1.edf'

# Setup for reading the raw data
raw = mne.io.read_raw_edf(raw_fname)
events = mne.find_events(raw, stim_channel='STI 014')

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=True)

# Construct Epochs
event_id, tmin, tmax = 1, -1., 3.
baseline = (None, 0)
epochs = mne.Epochs(raw, events, event_id, picks=picks,preload=True)

epochs.resample(150., npad='auto')  # resample to reduce computation time


# In[ ]:

epochs.plot_psd(fmin=2., fmax=40.)


# In[24]:

from collections import defaultdict
ch_types = defaultdict(str)
channel_names = []
channel_types = []

for ch in eeg_dat.ch_names:
    if 'EEG' in ch:
        ch_types[ch] = 'eeg'
    elif 'EMG' in ch:
        ch_types[ch] = 'emg'
    elif 'STI' in ch: 
        ch_types[ch] = 'stim'
    else:
        ch_types[ch] = 'misc'
        
eeg_dat.set_channel_types(ch_types)


# Example of one way to start fixing up the channels
#  Priority again is to keep everything consistent with MNE, so first check for MNE functions that can do 
#   things like this, and that can update this information insode the data.info object. 

from collections import defaultdict
ch_types = defaultdict(str)

channel_names = []
channel_types =[]

# lists to store the array names

for ch in eeg_dat.ch_names:    
    if 'EEG' in ch:
        ch_types[ch] = 'eeg'
        
    elif 'EMG' in ch:
        ch_types[ch] = 'emg'
        # Keeps track that this is an EMG channels
        
    elif 'STI' in ch:
        ch_types[ch] = 'stim'
        # Keeps track that this is an STI channel
    
    else:
        ch_types[ch] = 'misc'
        # Keeps track of MISC channels
        
eeg_dat.set_channel_types(ch_types)
        
for ch in eeg_dat.ch_names:    
    if 'EEG' in ch:
        ch = ch[4:7]
        ch =''.join(e for e in ch if e.isalnum())
        channel_names.append(ch)
        channel_types.append('eeg')
        ch_types[ch] = 'eeg'
        
    elif 'EMG' in ch:
        ch = ch[4:7]
        ch =''.join(e for e in ch if e.isalnum())
        channel_names.append(ch)
        channel_types.append('emg')
        ch_types[ch] = 'emg'
        # Keeps track that this is an EMG channels
        
    elif 'STI' in ch:
        channel_names.append(ch)
        channel_types.append('stim')
        ch_types[ch] = 'stim'
        # Keeps track that this is an STI channel
    
    else:
        channel_names.append(ch)
        channel_types.append('misc')
        ch_types[ch] = 'misc'
        # Keeps track of MISC channels
        
# The EEG channels use the standard naming strategy.
# By supplying the 'montage' parameter, approximate locations
montage = 'standard_1020'

info = mne.create_info(channel_names, eeg_dat.info['sfreq'], channel_types, montage)
print(info)


# In[25]:

eeg_dat.info


# In[26]:

eeg_dat.plot()


# In[27]:

eeg_dat.plot_psd()


# In[31]:

eeg_dat.plot_psd( average=True)


# In[ ]:



