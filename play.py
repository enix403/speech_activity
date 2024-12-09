from pathlib import Path

ROOT_DIR = Path("data/Data")
AUDIOS_DIR = ROOT_DIR / "Audio"
ANNOTATIONS_DIR = ROOT_DIR / "Annotation"

# annotation_path = "data/Data/Annotation/Female/TMIT/SI2220.TextGrid"

"""
labeled_list  = []
grid = textgrids.TextGrid(path)

for interval in grid['silences']:
    label = int(interval.text)

    dur = interval.dur
    dur_msec = dur * 1000 # sec -> msec
    num_frames = int(round(dur_msec /30)) # the audio is divided into 30 msec frames
    print(dur_msec)
    for i in range(num_frames):
        
        labeled_list.append(label)

return labeled_list
"""

"""
MS_PER_FRAME = 30

grid = textgrids.TextGrid(ann_path)
labels_per_frame = []

for interval in grid['silences']:
    label = int(interval.text) # 1 = speech, 0 = no speech
    ms = interval.dur * 1000
    num_frames = int(round(ms / MS_PER_FRAME))
    for _ in range(num_frames):
        labels_per_frame.append(label)
"""


def read_audio(audio_path: str):
    signal, freq = librosa.load(audio_path)
    return signal, freq


# how many sample points make up a frame (30ms)

30 / time_period

figure = plt.Figure(figsize=(10, 7), dpi=85)
plt.plot(t, signal)

for i, frame_labeled in enumerate(labels_per_frame):
    start = i * num_frame_samples
    end = start + num_frame_samples - 1

    if frame_labeled == 1:
        plt.axvspan(
            xmin= t[start], xmax=t[end],
            ymin=-1000, ymax=1000,
            alpha=0.4, zorder=-100,
            facecolor='g', label='Speech'
        )

