#!/bin/bash

# get training data (preprocessed)
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1FqWzk6qWEp80RWbFV4MCAhmSXX07uUEy" -O data/TONAS_note.zip
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=13MOh5tEQs0ftmVJ9ueczIdWNcBEXmAMX" -O data/TONAS_onset2.zip
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1hCCLBhV4kK2kvLQFzwLFYgF-4ieYsmWa" -O data/TONAS_offset2.zip
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1Fo1yL3S0KnJ3KcOcK7x5T-KTmjDc0jO1" -O data/ISMIR2014_note.zip
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1beAE2u31bIQJDv-btuoBeA3J5146Ojt0" -O data/ISMIR2014_onset2.zip
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1-4HEMdkDcaSTSkhhl_m7CTpumkehbztM" -O data/ISMIR2014_offset2.zip
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1ITHDTxLYLie02z-sD_823phvNcg8-Lc1" -O ans/TONAS_ans.zip
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1_fxgSF1-g5c9G_lG7Cq_3ZQ_sueZTf02" -O ans/ISMIR2014_ans.zip
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=16HwBuiqth_MFTusUw3wrms_yWdu2pjnF" -O pitch/ISMIR2014.zip

# unzip all files
unzip data/TONAS_note.zip -d data
unzip data/TONAS_onset2.zip -d data
unzip data/TONAS_offset2.zip -d data
unzip data/ISMIR2014_note.zip -d data
unzip data/ISMIR2014_onset2.zip -d data
unzip data/ISMIR2014_offset2.zip -d data
unzip ans/TONAS_ans.zip -d ans
unzip ans/ISMIR2014_ans.zip -d ans
unzip pitch/ISMIR2014.zip -d pitch