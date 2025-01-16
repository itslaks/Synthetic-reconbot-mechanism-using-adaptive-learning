import pyaudio
import wave
import numpy as np
import os
from scipy.signal import lfilter


def speech_to_text():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100  # Increased sample rate for better quality
    MAX_RECORD_SECONDS = 30  # Maximum recording time
    SILENCE_THRESHOLD = 500  # Adjust based on your environment
    SILENCE_DURATION = 1.5  # Stop after 1.5 seconds of silence
    PRE_BUFFER_DURATION = 0.5  # Keep 0.5 seconds before speech starts

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* Listening...")

    frames = []
    silent_chunks = 0
    has_started = False
    pre_buffer = []

    # Pre-emphasis filter coefficients
    pre_emphasis = 0.97
    emphasis_filter = [1.0, -pre_emphasis]

    for i in range(0, int(RATE / CHUNK * MAX_RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Apply pre-emphasis filter
        audio_data = lfilter(emphasis_filter, [1], audio_data)

        rms = np.sqrt(np.mean(audio_data ** 2))

        if rms > SILENCE_THRESHOLD:
            silent_chunks = 0
            if not has_started:
                has_started = True
                frames.extend(pre_buffer)
            frames.append(audio_data)
        else:
            silent_chunks += 1
            if has_started:
                frames.append(audio_data)
            else:
                pre_buffer.append(audio_data)
                if len(pre_buffer) > int(PRE_BUFFER_DURATION * RATE / CHUNK):
                    pre_buffer.pop(0)

        if has_started and silent_chunks > int(SILENCE_DURATION * RATE / CHUNK):
            break

    print("* Done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    if not os.path.exists('audio'):
        os.makedirs('audio')

    # Normalize audio
    audio_data = np.concatenate(frames)
    audio_data = audio_data / np.max(np.abs(audio_data))
    audio_data = (audio_data * 32767).astype(np.int16)

    wf = wave.open('audio/recording.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(audio_data.tobytes())
    wf.close()


if __name__ == "__main__":
    speech_to_text()