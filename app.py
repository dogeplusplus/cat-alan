import mlflow
import torch
import torchaudio
import streamlit as st
import moviepy.editor as mp
import matplotlib.pyplot as plt

from pathlib import Path
from tempfile import TemporaryDirectory


model_path = "examples/model"
model = mlflow.pytorch.load_model(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

classes = {
    0: "Happy",
    1: "Resting",
    2: "Angry",
    3: "Paining",
    4: "MotherCall",
    5: "Warning",
    6: "HuntingMind",
    7: "Fighting",
    8: "Defence",
    9: "Mating",
}


def preprocess_audio(audio):
    mono = torch.mean(audio, axis=0, keepdim=True)
    reshaped = torch.unsqueeze(mono, 0)
    return reshaped.to(device)


def predict(model, audio):
    log_softmax = model(audio)
    probabilities = torch.squeeze(torch.exp(log_softmax))
    return probabilities.cpu().detach().numpy()


st.title("Meow Sentiment Analysis")
st.sidebar.subheader("Provide Audio or Video file")
uploaded_video = st.sidebar.file_uploader("Video Path", type=["mp4"])


raw_bytes = uploaded_video.read()

with TemporaryDirectory() as temp_dir:
    temp_video_path = Path(temp_dir, "temp_video.mp4")
    with open(temp_video_path, "wb") as f:
        f.write(raw_bytes)

    video_object = mp.VideoFileClip(str(temp_video_path))
    audio_data = video_object.audio
    temp_audio_path = Path(temp_dir, "temp_audio.wav")
    audio_data.write_audiofile(temp_audio_path)
    audio, sr = torchaudio.load(temp_audio_path)
    model_input = preprocess_audio(audio)

    probabilities = predict(model, model_input)


    fig, ax = plt.subplots()
    sorted_pairs = sorted(zip(classes.values(), probabilities), key=lambda x: x[1])
    tuples = zip(*sorted_pairs)
    class_axis, label_axis = [list(t) for t in tuples]
    ax.barh(class_axis, label_axis)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Sentiment")
    st.pyplot(fig)

video_widget = st.video(raw_bytes)


