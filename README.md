# NSCNet - A Nightingale Songs Clustering Network

This repository contains the source code for a research on estimating the number of songs in the repertoire of a nightingale by means of **Deep Clustering**.
The dataset contains around 11k songs, that have been encoded, using the *encoding.py* script, based on *librosa*, into **Mel-spectrograms** (or, optionally, also into *raw waveforms* or *chromagrams*).
Three models have been implemented:
1. **BaseNet**: a baseline based on *PCA* dimensionality reduction followed by a clustering algorithm (*k-means* or *OPTICS*);
2. **VAENet**: same approach as above but using a Variational Autoencoder to compress the data;
3. **NSCNet**: the core of the project, an architecture based on a compression-clustering loop, with CNNs (*EfficientNet-B0*) followed by a *pseudo-label* generator (k-means) and a classification head, as depicted below.

![NSCNet representation](https://github.com/LIA-UniBo/NSCNet/blob/main/docs/NSCNet%20representation.jpg "NSCNet representation")

For more details, there is a report in the *docs* folder, containing all the information.

## Usage
To start a training and save results, it is enough to call one of the following methods from the *main.py* file:
- To launch the BaseNet model
```
basenet()
```
- To launch the VAENet model
```
vaenet()
```
- To launch the NSCNet model
```
nscnet()
```

Parameters can be optionally changed by accessing the *config.py* file in the main directory, or the specific files of configuration from each architecture's folder.

Results are stored in the *train* folder and they can be compared with Zipf's distribution by launching *utils/analysis/zipf_law_test.py*.

## Weights
The best trained weights for the NSCNet can be found here:
https://drive.google.com/drive/folders/1Iq2Y7q4eUzBbw2xZtSV3GFtIoJT4PpDC?usp=sharing
