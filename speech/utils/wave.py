import soundfile


def array_from_wave(file_name):
    audio, samp_rate = soundfile.read(file_name, dtype="int16")
    return audio, samp_rate


def wav_duration(file_name):
    info = soundfile.info(file_name)
    return info.frames / info.samplerate
