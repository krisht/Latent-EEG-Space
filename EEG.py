import mne
import os
import numpy as np
from collections import defaultdict

eeg_dat = mne.io.read_raw_edf('./a_1.edf', preload=True)





# Example of one way to start fixing up the channels
#  Priority again is to keep everything consistent with MNE, so first check for MNE functions that can do 
#   things like this, and that can update this information insode the data.info object. 

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
        
eeg_dat.set_channel_types(ch_types)
# The EEG channels use the standard naming strategy.
# By supplying the 'montage' parameter, approximate locations
montage = 'standard_1020'

info = mne.create_info(channel_names, eeg_dat.info['sfreq'], channel_types, montage)
print(info)
notches = np.arange(60, 61, 60)
eeg_dat.notch_filter(notches)
filtered = eeg_dat.copy().filter(1, 70, h_trans_bandwidth=10)


ch_names = filtered.info['ch_names']
picks = filtered.pick_channels(ch_names=ch_names)


data = filtered.get_data(start=0)

print(data.shape)


data2 = mne.time_frequency.stft(data, wsize=140, tstep=2)

print(data2.shape)






