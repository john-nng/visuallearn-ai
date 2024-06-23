from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusion3Pipeline
import torch
from huggingface_hub import login
import os
import os
import openai
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import requests
from PIL import Image
from io import BytesIO

openai.api_key = 'sk-proj-1yMUbKq5IQkoGyAYabkwT3BlbkFJSpNbtD4ot4AFoVqfGe7f'
client = openai.OpenAI(api_key = 'sk-proj-1yMUbKq5IQkoGyAYabkwT3BlbkFJSpNbtD4ot4AFoVqfGe7f')
def get_text_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=4096):
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

def generate_image_from_description(description, size="1024x1024"):
    response = openai.images.generate(
        prompt=description,
        n=1,
        size=size,
    )
    return response.data[0].url

def get_summary(input_message, message_type=0, prev = ''):
    added = 'This may include '
    if message_type == 0:
        added += 'notable visual and auditory detail'
    

    genre = "narrative"
    if message_type == 0:
        genre = 'storytelling'
    else:
        genre = 'narrative'
    delimiter = "####"

    system_message = f"""
    Summarize this text referencing quotes and details from it.\
    {added}\
    The summary should also be in the same {genre} format as the original text. \
    The user input \
    This is the context behind the text: {prev}. Continue the summary \
    so that it flows with the previous.
    message will be delimited with {delimiter} characters.
    """
    # remove possible delimiters in the user's message
    input_message = input_message.replace(delimiter, "")

    user_message_for_model = f"""
    {delimiter}{input_message}{delimiter}
    """

    messages =  [  
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
    # remove possible delimiters in the user's message
    input_message = input_message.replace(delimiter, "")

    user_message_for_model = f"""
    {delimiter}{input_message}{delimiter}
    """

    messages =  [  
        {'role': 'system', 'content': system_message},    
        {'role': 'user', 'content': user_message_for_model},  
    ] 
    response = get_text_from_messages(messages)
    return response

def get_image(input_message):
    description = get_description(input_message) 
    print('----------------------------------------------------------------------------------------------------------------------------------------------------------')
    print(description)
    image_url = generate_image_from_description(description)
    return image_url

def summary_to_speech(input, file_name):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input= input,
    )
    response.stream_to_file(file_name)


with open('sample_story.txt', 'r') as file:
    input_user_message = file.read()


def generate_loop(text, sentences_per_image = 30):
    sentences = sent_tokenize(text)
    i = len(sentences)
    images = []
    prev = ''
    iteration = 0

    while len(sentences) > 0:
        audio_file = f"wav_files/audio_file_{iteration}.wav"
        summary_file = f"summary_files/summary_file_{iteration}.txt"
        image_file = f"image_files/image_file_{iteration}.png"
        
        iteration +=1

        input_user_message = ''
        i -= sentences_per_image
        if len(sentences) < sentences_per_image:
            input_user_message = ''.join(sentences[0:])
            sentences = []
        else:
            input_user_message = ''.join(sentences[0:sentences_per_image])
            sentences = sentences[sentences_per_image:]
        summary = get_summary(input_user_message, 0, prev = prev)

        with open(summary_file, 'w') as file:
            file.write(summary)


        summary_to_speech(summary, audio_file)
        prev = summary
        url = get_image(get_description(input_user_message))

        response = requests.get(url)
        response.raise_for_status()  # Ensure we got a valid response
        
        # Open the image and save it as a PNG file
        image = Image.open(BytesIO(response.content))
        image.save(image_file, "PNG")
        
        print(summary)
        print(image)
        images.append(image)
        print('------------------------------------------------------------------------------------')
    return images
        

def create_html_with_images(image_urls, output_file="output.html"):
    with open(output_file, 'w') as f:
        f.write("<html><head><title>Image Gallery</title></head><body>\n")
        f.write("<div style='display: flex; flex-wrap: wrap;'>\n")
        for url in image_urls:
            f.write(f"<div style='margin: 10px;'><img src='{url}' style='max-width: 200px; max-height: 200px;'/></div>\n")
        f.write("</div>\n")
        f.write("</body></html>\n")

# Example usage
image_urls = generate_loop(input_user_message)

create_html_with_images(image_urls)

