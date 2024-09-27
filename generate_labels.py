import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
import random

# Load the trained model and tokenizer
model_path = "./meme_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Load the original jokes
df = pd.read_csv('data/shortjokes.csv')
jokes = df['Joke'].tolist()

def generate_meme_label(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    
    # Get the model's prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted label index
    predicted_label_index = outputs.logits.argmax().item()
    
    # Get the corresponding joke from the dataset
    meme_label = jokes[predicted_label_index % len(jokes)]
    
    return meme_label

def generate_meme_label_wrapper(user_input):
    return generate_meme_label(user_input)

if __name__ == "__main__":
    while True:
        user_input = input("Enter a prompt (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        meme_label = generate_meme_label_wrapper(user_input)
        print(f"Generated Meme Label: {meme_label}")

    print("Thank you for using the Meme Generator!")