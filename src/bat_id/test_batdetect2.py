from batdetect2 import api, plot
import matplotlib.pyplot as plt
import pandas as pd
import os

from bat_id import explore

import seaborn as sns
from sklearn.decomposition import PCA
import scipy.io.wavfile as wavfile

# Current Approach
# 1. Detect pulses with BatDetect2
# 2. Select for pulses of certain freq, get start and end times
# 3. Use powerflock to get sig_values
# 4. Use start and end times to get sig_values at those start and end times


AUDIO_DIR = "test_audio"
SIG_DIR = "test_audio_sig_values"
PROB_CUTOFF = 0.4


# Detect pulses with BatDetect2
def detect_pulses_from_audio(filename):
    audio_file = os.path.join(AUDIO_DIR, filename) 
    audio = api.load_audio(audio_file)
    detections, features, spec = api.process_audio(audio)

    return detections, features, spec


# Load saved sig_values for entire file
def load_and_update_sig_values(filename):
    sig_values = pd.read_csv(os.path.join(SIG_DIR, filename) )

    new_sig_values = pd.DataFrame({
                    "flatness": sig_values["flatness"],
                    "contiuity": sig_values["continuity"],
                    "pitch": sig_values["pitch"],
                    "freq_mod": sig_values["freq_mod"]
                    })

    new_sig_values.index = round(sig_values["Unnamed: 0"], 2)

    return new_sig_values


# Acquire sig_values at time points
def save_sig_value_slices(filename, detections, new_sig_values):
    file_ct = 0
    for det in detections:
        if det["det_prob"] > PROB_CUTOFF:
            start_time, end_time = round(det["start_time"], 2), round(det["end_time"], 2)

            if not start_time in new_sig_values.index:
                start_time -= 0.01

            start_index = int(new_sig_values.index.get_indexer([start_time]))
            sig_values_slice = new_sig_values.iloc[start_index:start_index + 3]

            sig_values_slice.to_csv("test_audio_slices/" +  filename[:-4] + "_" + str(file_ct) + ".csv", index_label=False)
            file_ct += 1


def filename_wav_to_csv(filename):
    return filename[:-3] + "csv"


def detect_pulses_dir():
    for filename in os.listdir(AUDIO_DIR):
        detections, features, spec = detect_pulses_from_audio(filename)
        new_sig_values = load_and_update_sig_values(filename_wav_to_csv(filename))
        save_sig_value_slices(filename, detections, new_sig_values)


# go throuh files
    # get one
    # get all files with that name
    # do an elementwise mean
    # save to new csv
def average_pulse_sigs(dir_pulse_sigs, dir_out):
    files = os.listdir(dir_pulse_sigs)

    unique_names = set([file[:31] for file in files])

    files_with_same_name = {name : [] for name in unique_names}
    for name in unique_names:
        for file in files:
            if file[:31] == name:
                files_with_same_name[name].append(file)

    print(files_with_same_name)



def vis_specs():
    config = api.get_config(
        detection_threshold=0.4,
        time_expansion_factor=1,
        max_duration=3,
    )
    detections, features, spec = detect_pulses_from_audio('../test_audio/barn1_D20220723T000059m428-Coto.wav')
    ax = plot.spectrogram_with_detections(spec, detections, config=config, figsize=(15, 4))
    
    # print(ax)

    # new_plot = ax.plot()
    # print(new_plot)
    # fig = new_plot[0].get_figure()
    # Fs, aud = wavfile.read('../test_audio/barn1_D20220723T000059m428-Coto.wav')


    ax.figure.savefig("output.png")




def main():
    # vis_specs()


   #  detect_pulses_dir()
    features, labels = explore.create_dataset_from_csv("test_audio_slices")
    explore.cluster_dataset(features, labels)

    X, Y = PCA(n_components=2).fit_transform(features).T
    sns.scatterplot(
        x=X,
        y=Y,
        style=[label for label in labels],
        # hue=[d["det_prob"] for d in detections],
    )


if __name__ == "__main__":
    main()












# for filename in os.listdir(DIRECTORY):
#     audio_file = os.path.join(DIRECTORY, filename) 
#     audio = api.load_audio(audio_file)
#     detections, features, spec = api.process_audio(audio)
#     print(len(detections))






# AUDIO_FILE = "test_audio/barn1_D20220723T000059m428-Coto.wav"

# # Process a whole file
# results = api.process_file(AUDIO_FILE)

# # Or, load audio and compute spectrograms
# audio = api.load_audio(AUDIO_FILE)
# # spec = api.generate_spectrogram(audio)

# # And process the audio or the spectrogram with the model
# detections, features, spec = api.process_audio(audio)
# # detections, features = api.process_spectrogram(spec)

# detections, features = api.process_spectrogram(spec, config=config)

# # Do something else ...