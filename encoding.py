import librosa
import numpy as np

class soundEffects:

    def smooth(data, axis=-1):

        return librosa.feature.delta(data=data, axis=axis)

    def extract_harmony(signal):

        return librosa.effects.harmonic(signal)

    def apply_mu_compression(signal, quantize):

        return librosa.mu_compress(signal, quantize)

class soundEncoding:

    def __init__(path, sampling_rate=None, mu_compression=False, harmonic=False, window_length=512, transform_method="Fourier"):

        if transform_method!="Fourier" and transform_method!="Constant-Q":
            raise Exception("Transform method must be one betwee 'Fourier' and 'Constant-Q'")

        else:
            self.window_length = window_length
            self.transform_method = transform_method
            _initialize_encodings()

            _open(path, sampling_rate)

            if harmonic:
                self.raw_signal = soundEffects.extract_harmony(self.raw_signal)

            if mu_compression:
                self.raw_signal = soundEffects.apply_mu_compression(self.raw_signal, quantize=False)

    def _open(path, sampling_rate):

        raw_input, sr = librosa.load(path, sr=sampling_rate)

        self.path = path
        self.raw_signal = raw_input
        self.sampling_rate = sr

    def _initialize_encodings():

        self.spectrogram = None
        self.chromagram = None
        self.cens_chromagram = None
        self.mel_spectrogram = None
        self.mfcc = None

    def get_spectrogram():

        if self.spectrogram==None:
            if self.transform_method=="Fourier":
                self.spectrogram = np.abs(librosa.stft(y=self.raw_signal, hop_length=self.window_length))
            elif self.transform_method=="Constant-Q":
                self.spectrogram = np.abs(librosa.cqt(y=self.raw_signal, hop_length=self.window_length))

        return self.spectrogram

    def get_chromagram():

        if self.chromagram==None:
            if self.transform_method=="Fourier":
                self.chromagram = librosa.feature.chroma_stft(y=self.raw_signal, sr=self.sampling_rate, hop_length=self.window_length)
            elif self.transform_method=="Constant-Q":
                self.chromagram = librosa.feature.chroma_cqt(y=self.raw_signal, sr=self.sampling_rate, hop_length=self.window_length)

        return self.chromagram

    def get_normalized_chromagram():

        if self.cens_chromagram==None:
            self.cens_chromagram = librosa.feature.chroma_cens(y=self.raw_signal, sr=self.sampling_rate, hop_length=self.window_length)

        return self.cens_chromagram

    def get_mel_spectrogram():

        if self.mel_spectrogram==None:
            self.mel_spectrogram = librosa.feature.melspectrogram(y=self.raw_signal, sr=self.sampling_rate, hop_length=self.window_length)

        return self.mel_spectrogram

    def get_mfcc():

        if self.mfcc==None:
            self.mfcc = librosa.feature.mfcc(y=self.raw_signal, sr=self.sampling_rate)

        return self.mfcc
