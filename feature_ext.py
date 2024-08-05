import scipy
import numpy as np
import python_speech_features as psf

from load_data import *

"""
Input: waveform (signal, frequeny)

Required:

(from paper)

sequence of 32 × 32 spectrogram images
constructed by computing 32-dimensional
log-mel-filterbank energies
using a frame step of 10 ms, and
stack-ing them together over 320 ms
to form one input image.

"""

winlen = 0.025
winstep = 0.010
num_mel_filters = 32

def create_frame_labels(
    signal_labels,
    frequency
):
    # (num_frames, frame_len)
    frame_labels = psf.sigproc.framesig(
        sig=signal_labels,
        frame_len=winlen * frequency,
        frame_step=winstep * frequency,
        winfunc=np.ones
    )

    # (num_frames, 1)
    frame_labels, _ = scipy.stats.mode(frame_labels, axis=1, keepdims=True)

    return frame_labels

def calc_nfft(frequency):
    est_frame_len = int(np.round(winlen * frequency))
    return 2**int(np.ceil(np.log2(est_frame_len)))

def create_spectograms(
    signal, # list of samples: (N,)
    frequency: float, # sampling frequency (in seconds)
    signal_labels, # binary labels of each sample (N,)
):
    # Get (log) mel-filterbank energies
    # (num_frames, num_mel_filters)
    frame_emb = psf.base.logfbank(
        signal=signal,
        samplerate=frequency,
        winlen=winlen,
        winstep=winstep,
        nfilt=num_mel_filters,
        nfft=calc_nfft(frequency),
        lowfreq=0,
        highfreq=None,
    )

    # Get corresponding frame labels
    # (num_frames, 1)
    frame_labels = create_frame_labels(
        signal_labels,
        frequency
    )


def load_and_extract(name: str):
    audio_path, ann_path = get_file_paths(name)

    signal, frequency = read_audio(audio_path)

    # list[(duration seconds, int label)]
    annotations = read_annotation(ann_path)

    # TODO: how to deal with rounding errors ?
    signal_labels = []
    for (duration, label) in annotations:
        num_dur_samples = int(np.round(duration * frequency))
        for _ in range(num_dur_samples):
            signal_labels.append(label)
