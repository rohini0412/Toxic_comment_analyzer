import gradio as gr
import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import nltk
import moviepy.editor as mp
import speech_recognition as sr

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the saved model and vectorizer
model = joblib.load('lgbm_multi_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define the labels
label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Function to predict toxicity for a given comment
def classify_comment(comment_text):
    preprocessed_text = preprocess_text(comment_text)
    X_tfidf = vectorizer.transform([preprocessed_text])
    y_pred_proba = model.predict_proba(X_tfidf)

    results = []

    for i, label in enumerate(label_columns):
        prob = y_pred_proba[i][0][1]
        results.append((label, prob * 100))

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    is_toxic = sorted_results[0][1] > 50

    return is_toxic, sorted_results

# Function to extract audio from video and transcribe using SpeechRecognition library
def extract_and_transcribe(video_path):
    # Extract audio from video
    video = mp.VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path)

    # Initialize the recognizer
    r = sr.Recognizer()

    # Open the audio file
    with sr.AudioFile(audio_path) as source:
        # Listen for the data (load audio to memory)
        audio_data = r.record(source)
        # Recognize (convert from speech to text)
        transcription = r.recognize_google(audio_data)
        print(transcription)

    # Save the transcription to a file
    file_path = "transcription.txt"
    with open(file_path, "w") as text_file:
        text_file.write(transcription)

    print("Text has been saved to:", file_path)

    return transcription

# Function to classify toxicity for video transcription
def classify_video(video_file):
    transcription = extract_and_transcribe(video_file)
    is_toxic, results = classify_comment(transcription)
    return transcription, is_toxic, results

# HTML template for circular progress bar
def circular_progress_html(label, percentage):
    color = f"rgba(255, {255 - int(2.55 * percentage)}, 0, 1)"
    return f"""
    <div style="display: inline-block; margin: 10px; text-align: center;">
        <div style="position: relative; display: inline-block; width: 100px; height: 100px; border-radius: 50%; background: conic-gradient({color} {percentage}%, #ddd {percentage}%);">
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 24px; color: black;">{percentage:.2f}%</div>
        </div>
        <div style="margin-top: 5px; font-size: 20px;">{label}</div>
    </div>
    """

# Define the Gradio interface for text classification
def text_ui(comment_text):
    is_toxic, results = classify_comment(comment_text)
    if is_toxic:
        result_text = '<span style="color: red; font-size: 24px;">The comment is toxic.</span>'
        result_details = "".join([circular_progress_html(label, prob) for label, prob in results])
    else:
        result_text = '<span style="color: green; font-size: 24px;">The comment is non-toxic.</span>'
        result_details = ""
    return result_text, result_details

def video_ui(video_file):
    transcription, is_toxic, results = classify_video(video_file)
    if is_toxic:
        result_text = '<span style="color: red; font-size: 24px;">The comment is toxic.</span>'
        result_details = "".join([circular_progress_html(label, prob) for label, prob in results])
    else:
        result_text = '<span style="color: green; font-size: 24px;">The comment is non-toxic.</span>'
        result_details = ""
    return transcription, result_text, result_details

with gr.Blocks() as demo:
    with gr.Tab("Text Classification"):
        with gr.Row():
            with gr.Column():
                comment_input = gr.Textbox(label="Enter a comment", lines=5)
                classify_button = gr.Button("Classify")
            with gr.Column():
                result_text = gr.HTML(label="Result")
                result_details = gr.HTML(label="Details")

        classify_button.click(text_ui, inputs=comment_input, outputs=[result_text, result_details])

    with gr.Tab("Video Classification"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload a short video")
                classify_button = gr.Button("Classify")
            with gr.Column():
                transcription_text = gr.Textbox(label="Transcription", interactive=False, lines=5)
                result_text = gr.HTML(label="Result")
                result_details = gr.HTML(label="Details")

        classify_button.click(video_ui, inputs=video_input, outputs=[transcription_text, result_text, result_details])

demo.launch()
