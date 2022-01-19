import torch
import mlflow
import torchaudio
import numpy as np
import streamlit as st
import moviepy.editor as mp
import matplotlib.pyplot as plt

from pathlib import Path
from tempfile import TemporaryDirectory


@st.cache(suppress_st_warning=True)
def load_model():
    model_path = "examples/model"
    model = mlflow.pytorch.load_model(model_path, map_location="cpu")
    # Switch off dropout
    model.eval()
    return model

classes = {
    0: "Happy",
    1: "Resting",
    2: "Angry",
    3: "Paining",
    4: "Mother Call",
    5: "Warning",
    6: "Hunting",
    7: "Fighting",
    8: "Defence",
    9: "Mating",
}

emojis = {
    0: "🐱 : Cat is happy.",
    1: "💤 : Cat is tired.",
    2: "😾 : Cat is angry.",
    3: "😿 : Cat sounds like it's in pain.",
    4: "🙀 : Cat is calling for mum.",
    5: "⚠️ : Cat is giving you a warning." ,
    6: "😼 : Cat wants to hunt.",
    7: "⚔️ : Cat is about to throw hands.",
    8: "🛡️  : Cat is on the defence.",
    9: "😻 : Cat wants to mate.",
}


def preprocess_audio(audio):
    device = "cpu"
    mono = torch.mean(audio, axis=0, keepdim=True)
    reshaped = torch.unsqueeze(mono, 0)
    return reshaped.to(device)


def predict(model, audio):
    log_softmax = model(audio)
    probabilities = torch.squeeze(torch.exp(log_softmax))
    return probabilities.cpu().detach().numpy()


model = load_model()

st.title("Meow Sentiment Analysis")
st.sidebar.subheader("Provide Audio or Video file")
uploaded_file = st.sidebar.file_uploader("File Path", type=["mp4", "mp3", "wav", "m4a"])
st.sidebar.write("`cat-alan` is an audio-classification model based on the M5 architecture. Uploading audio information, the model attempts to predict the emotion of the cat when speaking.")
st.sidebar.write("**For optimal results try to keep the audio to just the meow.")

left, right = st.columns([2, 1])

if uploaded_file is not None:
    raw_bytes = uploaded_file.read()

    extension = Path(uploaded_file.name).suffix

    with TemporaryDirectory() as temp_dir:
        # Serialize to video with moviepy, extract audio and then save the audio file
        if extension == ".mp4":
            temp_video_path = Path(temp_dir, "temp_video.mp4")
            with open(temp_video_path, "wb") as f:
                f.write(raw_bytes)

            video_object = mp.VideoFileClip(str(temp_video_path))
            audio_data = video_object.audio
            temp_audio_path = Path(temp_dir, "temp_audio.wav")
            audio_data.write_audiofile(temp_audio_path)
            video_widget = right.video(raw_bytes)
        else:
            # Save the audio file so we can access the array data
            temp_audio_path = Path(temp_dir, f"temp_audio{extension}")
            with open(temp_audio_path, "wb") as f:
                f.write(raw_bytes)
            audio_widget = right.audio(raw_bytes)
        audio, _ = torchaudio.load(temp_audio_path)

    model_input = preprocess_audio(audio)
    probabilities = predict(model, model_input)

    fig, ax = plt.subplots()
    sorted_pairs = sorted(zip(classes.values(), probabilities), key=lambda x: x[1])
    tuples = zip(*sorted_pairs)
    class_axis, label_axis = [list(t) for t in tuples]
    left.subheader(emojis[np.argmax(probabilities)])
    ax.barh(class_axis, label_axis)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Sentiment")
    left.pyplot(fig)

