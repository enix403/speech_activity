"""
Explain this code

```
# Function for reading labels from .TextGrig file:
def readLabels(path, sample_rate):
    '''
    Read the file and return the list of SPEECH/NONSPEECH labels for each frame
    '''
        
    labeled_list  = []
    grid = textgrids.TextGrid(path)

    for interval in grid['silences']:
        label = int(interval.text)

        dur = interval.dur
        dur_samples = int(np.round(dur * sample_rate)) # sec -> num of samples
        
        for i in range(dur_samples):
            labeled_list.append(label)

    return labeled_list



audio_files = [ .... ]
annotation_files = [ .... ]

# Algorithm based on a article "A HYBRID CNN-BILSTM VOICE ACTIVITY DETECTOR"

# In preprocessing part we need extract Mel filter bank energies from signal as a features. Features are extracted every 10 ms using a 25 ms window. We will use 32 Mel log energies and the log energy of frame. And after extracting, we need form sequences of 32 × 32 spectrogram imagesas input features.

#  Set params for model:
preemphasis_coef = 0.97 # Coefficient for pre-processing filter
frame_length = 0.025 # Window length in sec
frame_step = 0.01 # Length of step in sec
num_nfft = 512 # Point for FFT
num_features = 32 # Number of Mel filters
n_frames = 32 # Number of frames for uniting in image

# Extraction features for each file:
dataset = list()

for i in tqdm(range(len(audio_files))):
    sig, sample_rate = librosa.load(audio_files[i])
    markers = readLabels(path=annotation_files[i], sample_rate=sample_rate)

    # Extract logfbank features:
    features_logfbank = python_speech_features.base.logfbank(
        signal=sig,
        samplerate=sample_rate,
        winlen=frame_length,
        winstep=frame_step,
        nfilt=num_features,
        nfft=num_nfft,
        lowfreq=0,
        highfreq=None,
        preemph=preemphasis_coef,
    )

    # Reshape labels for each group of features:
    markers_of_frames = python_speech_features.sigproc.framesig(
        sig=markers,
        frame_len=frame_length * sample_rate,
        frame_step=frame_step * sample_rate,
        winfunc=np.ones,
    )

    # For every frame calc label:
    marker_per_frame = np.zeros(markers_of_frames.shape[0])
    marker_per_frame = np.array(
        [
            1
            if np.sum(markers_of_frames[j], axis=0) > markers_of_frames.shape[0] / 2
            else 0
            for j in range(markers_of_frames.shape[0])
        ]
    )

    spectrogram_image = np.zeros((n_frames, n_frames))
    for j in range(int(np.floor(features_logfbank.shape[0] / n_frames))):
        spectrogram_image = features_logfbank[j * n_frames : (j + 1) * n_frames]
        label_spectrogram_image = (
            1
            if np.sum(marker_per_frame[j * n_frames : (j + 1) * n_frames])
            > n_frames / 2
            else 0
        )
        dataset.append((label_spectrogram_image, spectrogram_image))

```
"""



"""
Spectogram
log mel-filterbank energies
"""


python_speech_features.base.logfbank(
    signal=signal,
    samplerate=freq,
    winlen=frame_length,
    winstep=frame_step,
    nfilt=num_features,
    nfft=num_nfft,
    lowfreq=0,
    highfreq=None,
    preemph=preemphasis_coef,
)



