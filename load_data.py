from pathlib import Path

import textgrids
import librosa

ROOT_DIR = Path("data/Data")
AUDIOS_DIR = ROOT_DIR / "Audio"
ANNOTATIONS_DIR = ROOT_DIR / "Annotation"

def get_file_paths(name: str):
    audio_path = AUDIOS_DIR / (name + ".wav")
    annotation_path = ANNOTATIONS_DIR / (name + ".TextGrid")

    return audio_path, annotation_path

def read_audio(audio_path: str):
    signal, freq = librosa.load(audio_path)
    return signal, freq

def read_annotation(ann_path: str):
    grid = textgrids.TextGrid(ann_path)
    annotations = []
    
    for interval in grid['silences']:
        label = int(interval.text) # 1 = speech, 0 = no speech
        annotations.append((interval.dur, label))

    return annotations
