import librosa
import numpy as np


class SoundEffects:

    def smooth(data, axis=-1):
        return librosa.feature.delta(data=data, axis=axis)

    def extract_harmony(signal):
        return librosa.effects.harmonic(signal)

    def apply_mu_compression(signal, mu, quantize):
        return librosa.mu_compress(signal, mu=mu, quantize=quantize)

    def trim(signal):
        trimmed_signal, index = librosa.effects.trim(signal, top_db=50)
        return trimmed_signal


class SoundEncoding:

    def __init__(self, path, sampling_rate=None, trim=True, mu_compression=False, harmonic=False, window_length=512,
                 transform_method="Fourier"):

        if transform_method != "Fourier" and transform_method != "Constant-Q":
            raise Exception("Transform method must be one between 'Fourier' and 'Constant-Q'")

        else:
            self.window_length = window_length
            self.transform_method = transform_method
            self._initialize_encodings()

            self._open(path, sampling_rate)

            if trim:
                self.raw_signal = SoundEffects.trim(self.raw_signal)

            if harmonic:
                self.raw_signal = SoundEffects.extract_harmony(self.raw_signal)

            if mu_compression:
                self.raw_signal = SoundEffects.apply_mu_compression(self.raw_signal, mu=256, quantize=False)

    def _open(self, path, sampling_rate):

        raw_input, sr = librosa.load(path, sr=sampling_rate)

        self.path = path
        self.raw_signal = raw_input
        self.sampling_rate = sr

    def _initialize_encodings(self):

        self.spectrogram = None
        self.chromagram = None
        self.cens_chromagram = None
        self.mel_spectrogram = None
        self.mfcc = None

    def get_raw_waveform(self):

        return self.raw_signal

    def get_spectrogram(self):

        if self.spectrogram is None:
            if self.transform_method == "Fourier":
                self.spectrogram = np.abs(librosa.stft(y=self.raw_signal, hop_length=self.window_length))
            elif self.transform_method == "Constant-Q":
                self.spectrogram = np.abs(librosa.cqt(y=self.raw_signal, hop_length=self.window_length))

        return self.spectrogram

    def get_chromagram(self):

        if self.chromagram is None:
            if self.transform_method == "Fourier":
                self.chromagram = librosa.feature.chroma_stft(y=self.raw_signal, sr=self.sampling_rate,
                                                              hop_length=self.window_length)
            elif self.transform_method == "Constant-Q":
                self.chromagram = librosa.feature.chroma_cqt(y=self.raw_signal, sr=self.sampling_rate,
                                                             hop_length=self.window_length)

        return self.chromagram

    def get_normalized_chromagram(self):

        if self.cens_chromagram is None:
            self.cens_chromagram = librosa.feature.chroma_cens(y=self.raw_signal, sr=self.sampling_rate,
                                                               hop_length=self.window_length)

        return self.cens_chromagram

    def get_mel_spectrogram(self):

        if self.mel_spectrogram is None:
            self.mel_spectrogram = librosa.feature.melspectrogram(y=self.raw_signal, sr=self.sampling_rate,
                                                                  hop_length=self.window_length)

        return self.mel_spectrogram

    def get_mfcc(self):

        if self.mfcc is None:
            self.mfcc = librosa.feature.mfcc(y=self.raw_signal, sr=self.sampling_rate)

        return self.mfcc
