import os
import os.path as osp
from argparse import ArgumentParser
from pathlib import Path
import warnings

from utils.extractor.song_extractor import SongExtractor

warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("-i", "--input", dest="data_dir",
                    help="raw data directory of audio files", metavar="FILE", type=Path)
parser.add_argument("-o", "--output", dest="out_dir",
                    help="directory where to save processed files", type=Path)
args = parser.parse_args()

DATA_DIR = args.data_dir
OUT_DIR = args.out_dir

BIRDS = {'andrei', 'biggie', 'bobert', 'chimichanga', 'clipper'}
BIRDS_FOLDERS = [b + '_bos' for b in BIRDS]
CHANNELS = [1, 2]

# build lists of all audio and offset files
# N.B. Make sure ONLY channels audio files are present in data folder
AUDIO_FILES = [osp.join(DATA_DIR, folder, f) for folder in BIRDS_FOLDERS for f in os.listdir(osp.join(DATA_DIR, folder)) if f.endswith(".wav")]
OFFSET_FILES = [osp.join(DATA_DIR, folder, f) for folder in BIRDS_FOLDERS for f in os.listdir(osp.join(DATA_DIR, folder)) if f.endswith(".txt")]
# build instances of SongExtractor for each couple of audio and offset files
extractors = [SongExtractor(audio, offset) for audio, offset in zip(AUDIO_FILES, OFFSET_FILES)]

i = 0
for bird in BIRDS:
    for ch in CHANNELS:
        save_dir = osp.join(OUT_DIR, bird, "CH{}".format(ch))
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        extractors[i](save_dir)
        i += 1