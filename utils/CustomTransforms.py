import torchaudio

def LogMelTransform(x):
    
    # get mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate = 16000,
                                                           n_fft = 1024,
                                                           win_length = 1024,
                                                           hop_length = 502,
                                                           n_mels = 128)

    # get amplitude into db
    amplitude_transform = torchaudio.transforms.AmplitudeToDB(stype='amplitude', 
                                                              top_db=150)

    # apply transform
    output = amplitude_transform(mel_spectrogram(x))
    
    # normalize amplitude of spectrogram [0.0 to 1.0]
    output = output - output.min()
    output = output / output.max()
    
    return output