import time
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
import torch
import os

# Load the dataset
df = pd.read_csv('data/shortjokes.csv')
jokes = df['Joke'].tolist()

# Split the data into training and validation sets
train_texts, val_texts = train_test_split(jokes, test_size=0.1)

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(jokes))

# Tokenize the data
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Create torch datasets
class JokeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = JokeDataset(train_encodings, list(range(len(train_texts))))
val_dataset = JokeDataset(val_encodings, list(range(len(val_texts))))

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Define the callback
class TimeLimitCallback(TrainerCallback):
    def __init__(self, time_limit):
        self.start_time = time.time()
        self.time_limit = time_limit

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - self.start_time > self.time_limit:
            control.should_training_stop = True
        return control

# Add the callback to the trainer
time_limit_callback = TimeLimitCallback(time_limit=3600)  # 1 hour
trainer.add_callback(time_limit_callback)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./meme_model")
tokenizer.save_pretrained("./meme_model")

print(f"Training completed in {time.time() - time_limit_callback.start_time:.2f} seconds")