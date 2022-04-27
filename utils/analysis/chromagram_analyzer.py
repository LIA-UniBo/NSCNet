import librosa
import librosa.display
import encoding
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def get_chromagrams(path):
    chromagrams = []

    for i in tqdm(range(1, 132)):
        file_path = os.path.join(path, "song_" + str(i) + ".wav")
        file = encoding.SoundEncoding(file_path, harmonic=False)
        chromagrams.append(file.get_chromagram())

    return chromagrams


def plot_chromagram(chromagram, file_path=None):
    fig = plt.figure()
    #, y_axis='chroma', x_axis='time'
    img = librosa.display.specshow(chromagram)
    plt.colorbar()
    # plt.imshow(img)

    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig)


def normalize_chromagram(chromagram):
    time_steps = chromagram.shape[1]
    for time_step in range(time_steps):
        total = np.sum(chromagram[:, time_step])
        chromagram[:, time_step] /= total


def compute_vector_entropy(chromo_vector):
    probabilities = np.copy(chromo_vector)
    log_probabilities = np.log2(probabilities)
    entropy = -sum(probabilities * log_probabilities)

    return entropy


def compute_chromo_entropy(chromagram):
    entropies = []

    time_steps = chromagram.shape[1]
    for time_step in range(time_steps):
        vector_entropy = compute_vector_entropy(chromagram[:, time_step])
        entropies.append(vector_entropy)

    return np.mean(entropies)


def get_highest_pitches(chromagram):
    pitches = []

    time_steps = chromagram.shape[1]
    for time_step in range(time_steps):
        highest_pitch = np.argmax(chromagram[:, time_step]) + 1
        pitches.append(highest_pitch)

    return pitches


def pad_sequence(vector, length):
    vect_len = len(vector)
    diff = length - vect_len
    if diff > 0:
        for i in range(diff):
            vector.append(0)


def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_product = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    return dot_product / norm_product


# -----------------------------------------------------------
chromagrams = get_chromagrams(os.path.join("data_raw", "CH1"))

for idx, c in enumerate(chromagrams):
    plot_chromagram(c, os.path.join("data_raw", "processed", f'{idx}.png'))

for chromagram in chromagrams:
    normalize_chromagram(chromagram)

entropies = [compute_chromo_entropy(chromagram) for chromagram in chromagrams]
mean_entropy = np.mean(entropies)
std_entropy = np.std(entropies)
print("Max entropy allowed: ", np.log2(12))
print("Mean: ", mean_entropy)
print("Standard deviation: ", std_entropy)
print()

pitches_vectors = [get_highest_pitches(chromagram) for chromagram in chromagrams]
max_length = max(len(pitches_vector) for pitches_vector in pitches_vectors)
for pitches_vector in pitches_vectors:
    pad_sequence(pitches_vector, max_length)

similarity_matrix = np.zeros((len(pitches_vectors), len(pitches_vectors)))
for i in range(len(pitches_vectors)):
    for j in range(len(pitches_vectors)):
        similaririty = cosine_similarity(pitches_vectors[i], pitches_vectors[j])
        similarity_matrix[i][j] = similaririty
print(similarity_matrix)
