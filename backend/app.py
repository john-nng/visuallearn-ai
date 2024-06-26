import requests
from flask import Flask, render_template, send_from_directory, request, redirect, url_for, jsonify
import os
import threading
from nltk.tokenize import sent_tokenize
from PIL import Image
from io import BytesIO
import openai
import nltk
from celery_worker import app as celery_app

nltk.download('punkt')

app = Flask(__name__)

# Configure paths for generated content
IMAGE_FOLDER = 'image_files'
SUMMARY_FOLDER = 'summary_files'
AUDIO_FOLDER = 'wav_files'

app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['SUMMARY_FOLDER'] = SUMMARY_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER

# Ensure directories exist
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

#TODO when using, un comment and insert your own openai key
#openai.api_key = ''

def get_text_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=4096):
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

def generate_image_from_description(description, size="512x512"):
    print("generating image----------------------------")
    response = openai.images.generate(
        prompt=description,
        n=1,
        size=size,
    )
    return response.data[0].url

def summary_to_speech(input, file_name):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input= input,
    )
    response.stream_to_file(file_name)


def get_summary(input_message, message_type=0, prev=''):
    added = 'This may include notable visual and auditory detail.'
    genre = "narrative" if message_type == 0 else "storytelling"
    delimiter = "####"

    system_message = f"""
    Summarize this text referencing quotes and details from it.\
    {added}
    The summary should also be in the same {genre} format as the original text, but should be either one-third its length\
    or less than 50 words
    The user input \
    This is the context behind the text: {prev}. Continue the summary \
    so that it flows with the previous.
    message will be delimited with {delimiter} characters.
    """
    input_message = input_message.replace(delimiter, "")

    user_message_for_model = f"""
    {delimiter}{input_message}{delimiter}
    """

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message_for_model},
    ]
    response = get_text_from_messages(messages)
    return response

def get_description(input_message):
    delimiter = "####"

    system_message = f"""
    Create a short description of an image that represents the main action of the text.
    """
    input_message = input_message.replace(delimiter, "")

    user_message_for_model = f"""
    {delimiter}{input_message}{delimiter}
    """

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message_for_model},
    ]
    response = get_text_from_messages(messages)
    return response

def get_image(input_message):
    description = get_description(input_message)
    image_url = generate_image_from_description(description)
    return image_url

def generate_slide(slide_number, input_text, sentences_per_image=30):
    try:
        sentences = sent_tokenize(input_text)
        start = slide_number * sentences_per_image
        end = start + sentences_per_image

        if end > len(sentences):
            end = len(sentences)

        input_user_message = ' '.join(sentences[start:end])

        summary = get_summary(input_user_message, 0)

        summary_file = os.path.join(SUMMARY_FOLDER, f'summary_file_{slide_number}.txt')
        with open(summary_file, 'w') as file:
            file.write(summary)

        audio_file = os.path.join(AUDIO_FOLDER, f'audio_file_{slide_number}.wav')
        summary_to_speech(summary, audio_file)

        description = get_description(input_user_message)
        image_url = get_image(input_user_message)
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image_file = os.path.join(IMAGE_FOLDER, f'image_file_{slide_number}.png')
        image.save(image_file, "PNG")
    except Exception as e:
        print(f"Error generating slide {slide_number}: {e}")

def generate_slides(input_text):
    sentences = sent_tokenize(input_text)
    num_slides = (len(sentences) + 29) // 30

    threads = []
    for i in range(num_slides):
        thread = threading.Thread(target=generate_slide, args=(i, input_text))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_text', methods=['POST'])
def submit_text():
    input_text = request.form['input_text']

    # Clear existing files
    clear_directory(IMAGE_FOLDER)
    clear_directory(SUMMARY_FOLDER)
    clear_directory(AUDIO_FOLDER)

    with open('sample_story.txt', 'w') as file:
        file.write(input_text)

    # Start generating slides in a separate thread
    threading.Thread(target=generate_slides, args=(input_text,)).start()

    return redirect(url_for('slides'))

@app.route('/api/slide_status/<int:slide_number>')
def slide_status(slide_number):
    image_file = os.path.join(app.config['IMAGE_FOLDER'], f'image_file_{slide_number}.png')
    summary_file = os.path.join(app.config['SUMMARY_FOLDER'], f'summary_file_{slide_number}.txt')
    audio_file = os.path.join(app.config['AUDIO_FOLDER'], f'audio_file_{slide_number}.wav')

    exists = os.path.exists(image_file) and os.path.exists(summary_file) and os.path.exists(audio_file)
    return jsonify({'exists': exists})

@app.route('/api/slide/<int:slide_number>')
def api_slide(slide_number):
    image_file = os.path.join(app.config['IMAGE_FOLDER'], f'image_file_{slide_number}.png')
    summary_file = os.path.join(app.config['SUMMARY_FOLDER'], f'summary_file_{slide_number}.txt')
    audio_file = os.path.join(app.config['AUDIO_FOLDER'], f'audio_file_{slide_number}.wav')

    if not (os.path.exists(image_file) and os.path.exists(summary_file) and os.path.exists(audio_file)):
        return {'exists': False}, 404

    with open(summary_file, 'r') as file:
        summary = file.read()

    return {
        'slide_number': slide_number + 1,
        'image_file': url_for('serve_image', filename=f'image_file_{slide_number}.png'),
        'summary': summary,
        'audio_file': url_for('serve_audio', filename=f'audio_file_{slide_number}.wav'),
        'exists': True
    }

@app.route('/slides')
def slides():
    return render_template('slides.html')

@app.route('/image_files/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

@app.route('/summary_files/<filename>')
def serve_summary(filename):
    return send_from_directory(app.config['SUMMARY_FOLDER'], filename)

@app.route('/wav_files/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['AUDIO_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)