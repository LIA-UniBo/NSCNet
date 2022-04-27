import csv
import os.path as osp
import os
from pydub import AudioSegment
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("-i", "--input", dest="data_dir",
                    help="raw data directory of audio files", metavar="FILE", type=Path)
parser.add_argument("-o", "--output", dest="out_dir",
                    help="directory where to save processed files", type=Path)
args = parser.parse_args()

class SongExtractor:
  """
  Extract and save the songs present in an audio file, given the
  file with onset and offset of each song
  """
  def __init__(self, audio_file, offset_file):
    """
    Initialize audio and offset file paths

    Parameters
    ----------
    audio_file : string
        path of audio file
    offset_file : string
        path of offset file
    """
    self.audio_file = audio_file
    self.offset_file = offset_file

  def read_offsets(self, separator = "\t"):
    """
    Read offset file and return a list with its lines

    Parameters
    ----------
    separator : string
        separator used by the library (default: \t)

    Returns
    -------
    list
        a list of lists of onset, offset and progressive id
    """
    with open(self.offset_file, newline = '') as f:
      file_reader = csv.reader(f, delimiter=separator)
      return [l for l in file_reader]

  def process_audio(self, out_dir, format = "wav"):
    """
    Process the audio and save splitted songs

    Parameters
    ----------
    out_dir : string
        output path where to save processed songs
    format : string
        file format of cutted songs (default: wav)
    """
    count = 0
    offsets = self.read_offsets()
    audio = AudioSegment.from_file(self.audio_file)
    for o in tqdm(offsets):
      audio_span = audio[float(o[0])*1000:float(o[1])*1000]
      filename = "song_{}.{}".format(o[2], format)
      audio_span.export(osp.join(out_dir, filename), format=format)
      count += 1
    print("\n{} songs successfully saved".format(count))

  def __call__(self, out_dir):
    self.process_audio(out_dir)

DATA_DIR = args.data_dir
OUT_DIR = args.out_dir

BIRDS_FOLDERS = os.listdir(DATA_DIR)
CHANNELS = [1, 2]

# build lists of all audio and offset files
# N.B. Make sure ONLY channels audio files are present in data folder
AUDIO_FILES = [osp.join(DATA_DIR, folder, f) for folder in BIRDS_FOLDERS for f in os.listdir(osp.join(DATA_DIR, folder)) if f.endswith(".wav")]
OFFSET_FILES = [osp.join(DATA_DIR, folder, f) for folder in BIRDS_FOLDERS for f in os.listdir(osp.join(DATA_DIR, folder)) if f.endswith(".txt")]
# build instances of SongExtractor for each couple of audio and offset files
extractors = [SongExtractor(audio, offset) for audio, offset in zip(AUDIO_FILES, OFFSET_FILES)]

i = 0
for bird in BIRDS_FOLDERS:
    for ch in CHANNELS:
        save_dir = osp.join(OUT_DIR, bird, "CH{}".format(ch))
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        extractors[i](save_dir)
        i += 1
