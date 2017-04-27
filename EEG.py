
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib qt')
import mne
import os
import numpy as np


# In[2]:

eeg_dat = mne.io.read_raw_edf('./session1/a_1.edf', preload=True)


# In[3]:

eeg_dat.ch_names


# In[4]:

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


# In[5]:

eeg_dat.info


# In[6]:

eeg_dat.plot()


# In[7]:

eeg_dat.plot_psd()


# In[8]:

notches = np.arange(60, 61, 60)
eeg_dat.notch_filter(notches)
print(eeg_dat.get_data(start=0).shape)
filtered = eeg_dat.copy().filter(1, 70, h_trans_bandwidth=10)
print(filtered.get_data(start=0).shape)


# In[9]:

filtered.plot_psd()
ch_names = filtered.info['ch_names']
picks = filtered.pick_channels(ch_names=ch_names)


# In[10]:

filtered.ch_names
data = filtered.get_data(start=0, stop=250)

print(data.shape)


# In[11]:

data2 = mne.time_frequency.stft(data, wsize=140, tstep=2)

print(data2.shape)


# In[12]:

filtered.ch_names


# In[13]:

print(np.real(data2))


# In[14]:

print(np.imag(data2))


# In[15]:

print(np.absolute(data2))


# In[ ]:




# In[ ]:



