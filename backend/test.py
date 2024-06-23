import openai
# Set your API key
openai.api_key = 'sk-proj-1yMUbKq5IQkoGyAYabkwT3BlbkFJSpNbtD4ot4AFoVqfGe7f'

try:
    # Make a simple API request to verify the key using the new API methods
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say this is a test"}
        ]
    )
    # Print the response
    print(response.choices[0].message['content'].strip())
except openai.error.OpenAIError as e:
    print(f"An error occurred: {e}")