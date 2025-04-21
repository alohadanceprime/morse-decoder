import os
import librosa
import numpy as np


def create_spectrogram(audio_path, output_path, sr=8000):
    y, sr = librosa.load(audio_path, sr=sr)

    n_fft = 128
    hop_length = 64
    win_length = n_fft
    window = 'hann'

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window))

    # Определение основной частоты
    avg_spectrum = np.mean(S, axis=1)
    main_freq_idx = np.argmax(avg_spectrum)

    # Рассчитываем частоты для бинов
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    main_freq = freqs[main_freq_idx]

    # Определяем диапазон для обрезки
    freq_range = 1
    start_freq = max(0, main_freq - freq_range)
    end_freq = main_freq + freq_range

    start_idx = np.argmin(np.abs(freqs - start_freq))
    end_idx = np.argmin(np.abs(freqs - end_freq)) + 1

    S_cropped = S[start_idx:end_idx, :]

    S_db = librosa.amplitude_to_db(S_cropped, ref=np.max)

    # Сохраняем как numpy array
    np.save(output_path, S_db.astype(np.float32))


os.makedirs("data/spectrogramms", exist_ok=True)
audio_dir = "data/audio"
for filename in os.listdir(audio_dir):
    if filename.endswith(".opus"):
        audio_path = os.path.join(audio_dir, filename)
        output_path = os.path.join("data/spectrogramms",
                                   f"{os.path.splitext(filename)[0]}.npy")
        create_spectrogram(audio_path, output_path)
