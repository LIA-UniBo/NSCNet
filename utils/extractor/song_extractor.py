import csv
import os.path as osp
from pydub import AudioSegment
from tqdm import tqdm

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
