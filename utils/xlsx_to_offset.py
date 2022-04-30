from openpyxl import load_workbook
import os.path as osp
import os
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", "--input", dest="data_dir",
                    help="raw data directory of audio files", metavar="FILE", type=Path)
parser.add_argument("-o", "--output", dest="out_dir",
                    help="directory where to save processed files", type=Path)
args = parser.parse_args()

DATA_DIR = args.data_dir
CHANNELS = [1, 2]
BIRDS_FOLDERS = os.listdir(DATA_DIR)

AUDIO_FILES = [osp.join(DATA_DIR, folder, f) for folder in BIRDS_FOLDERS for f in os.listdir(osp.join(DATA_DIR, folder)) if f.endswith(".wav")]
OFFSET_FILENAMES = [osp.splitext(f)[0] for f in AUDIO_FILES]

for folder in BIRDS_FOLDERS:
    XLSX_FILE = [f for f in os.listdir(osp.join(DATA_DIR, folder)) if f.endswith(".xlsx")][0]
    for ch in CHANNELS:
        wb = load_workbook(osp.join(DATA_DIR, folder, XLSX_FILE))
        ws = wb.worksheets[ch]
        with open(osp.join(DATA_DIR, folder, OFFSET_FILENAMES[ch-1] + '_label.txt'), 'w+') as outfile:
            print(osp.join(DATA_DIR, folder, OFFSET_FILENAMES[ch-1] + '_label.txt'))

            for idx, row in enumerate(ws.rows):
                if (idx > 0):
                    outfile.write(str(float(row[0].value)) + "\t" + str(float(row[1].value)) + "\t" + str(row[2].value) + "\n")
