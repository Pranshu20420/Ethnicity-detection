#%% md
# Data Analysis
#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('names.csv')

# Combine smaller categories into 'Other' for Ethnicity
top_ethnicities = data['Ethnicity'].value_counts().nlargest(5).index
data['Ethnicity Simplified'] = data['Ethnicity'].apply(lambda x: x if x in top_ethnicities else 'Other')

# Combine smaller categories into 'Other' for Country
top_countries = data['Country'].value_counts().nlargest(5).index
data['Country Simplified'] = data['Country'].apply(lambda x: x if x in top_countries else 'Other')

sns.set_style("whitegrid")

# Ethnicity Distribution (Simplified)
plt.figure(figsize=(10, 8))
data['Ethnicity Simplified'].value_counts().plot(kind='barh')
plt.title('Ethnicity Distribution (Simplified)')
plt.xlabel('Count')
plt.ylabel('Ethnicity')
plt.show()

# Country Distribution (Simplified)
plt.figure(figsize=(10, 8))
data['Country Simplified'].value_counts().plot(kind='barh')
plt.title('Country Distribution (Simplified)')
plt.xlabel('Count')
plt.ylabel('Country')
plt.show()

# Ethnicity by Gender (Simplified)
plt.figure(figsize=(12, 8))
sns.countplot(y='Ethnicity Simplified', hue='Gender', data=data)
plt.title('Ethnicity by Gender (Simplified)')
plt.xlabel('Count')
plt.ylabel('Ethnicity')
plt.legend(title='Gender')
plt.show()

#%% md
# Load and Prepare the Data
#%%

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('names.csv')

# Preprocess names by concatenating them for tokenization
df['full_name'] = df['Name']  # Assuming 'Name' is already a concatenated full name

# Encode your labels
gender_encoder = LabelEncoder()
ethnicity_encoder = LabelEncoder()
country_encoder = LabelEncoder()

df['Gender'] = gender_encoder.fit_transform(df['Gender'])
df['Ethnicity'] = ethnicity_encoder.fit_transform(df['Ethnicity'])
df['Country'] = country_encoder.fit_transform(df['Country'])

num_genders = df['Gender'].nunique()
num_ethnicities = df['Ethnicity'].nunique()
num_countries = df['Country'].nunique()

print(f"Number of unique genders: {num_genders}")
print(f"Number of unique ethnicities: {num_ethnicities}")
print(f"Number of unique countries: {num_countries}")

#%%

from sklearn.model_selection import train_test_split

# Split the data into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

#%%

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch

# Tokenize the full names
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
train_tokens = tokenizer(list(train_df['full_name'].values), padding=True, truncation=True, return_tensors='pt')
val_tokens = tokenizer(list(val_df['full_name'].values), padding=True, truncation=True, return_tensors='pt')

class NamesDataset(Dataset):
    def __init__(self, tokens, genders, ethnicities, countries):
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        self.genders = genders
        self.ethnicities = ethnicities
        self.countries = countries

    def __len__(self):
        return len(self.genders)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'genders': self.genders[idx],
            'ethnicities': self.ethnicities[idx],
            'countries': self.countries[idx]
        }

# Create the train and validation datasets
train_dataset = NamesDataset(train_tokens, torch.tensor(train_df['Gender'].values), torch.tensor(train_df['Ethnicity'].values), torch.tensor(train_df['Country'].values))
val_dataset = NamesDataset(val_tokens, torch.tensor(val_df['Gender'].values), torch.tensor(val_df['Ethnicity'].values), torch.tensor(val_df['Country'].values))

# Create the train and validation dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

#%% md
# Transformer Model
#%%

from transformers import AutoModel
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

class NameClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_genders=num_genders, num_ethnicities=num_ethnicities, num_countries=num_countries, dropout_rate=0.1):
        super(NameClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.gender_classifier = nn.Linear(self.transformer.config.hidden_size, num_genders)
        self.ethnicity_classifier = nn.Linear(self.transformer.config.hidden_size, num_ethnicities)
        self.country_classifier = nn.Linear(self.transformer.config.hidden_size, num_countries)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        gender = self.gender_classifier(pooled_output)
        ethnicity = self.ethnicity_classifier(pooled_output)
        country = self.country_classifier(pooled_output)
        return gender, ethnicity, country

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = NameClassifier(model_name='bert-base-uncased', num_genders=num_genders, num_ethnicities=num_ethnicities, num_countries=num_countries, dropout_rate=0.1)

#%% md
# LSTM Model & Separate Pre-Processing
#%%

from torchtext.vocab import build_vocab_from_iterator

def yield_tokens(df):
    for name in df['full_name']:
        yield tokenizer.encode(name, add_special_tokens=False)

vocab = build_vocab_from_iterator(yield_tokens(train_df), specials=['<pad>'])
vocab.set_default_index(vocab['<pad>'])

def encode_name(name):
    tokens = vocab(tokenizer.encode(name, add_special_tokens=False))
    return torch.tensor(tokens, dtype=torch.long)

# Encode names in the train and validation sets
train_df['input_ids'] = train_df['full_name'].apply(encode_name)
val_df['input_ids'] = val_df['full_name'].apply(encode_name)

class NamesDataset(Dataset):
    def __init__(self, input_ids, genders, ethnicities, countries):
        self.input_ids = input_ids
        self.genders = genders
        self.ethnicities = ethnicities
        self.countries = countries

    def __len__(self):
        return len(self.genders)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = torch.ones(input_ids.size(0), dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'genders': self.genders[idx],
            'ethnicities': self.ethnicities[idx],
            'countries': self.countries[idx]
        }

#%% md
# LSTM Model
#%%

import torch.nn.utils.rnn as rnn_utils

class LSTMNameClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_genders, num_ethnicities, num_countries, dropout_rate=0.1):
        super(LSTMNameClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.gender_classifier = nn.Linear(hidden_size, num_genders)
        self.ethnicity_classifier = nn.Linear(hidden_size, num_ethnicities)
        self.country_classifier = nn.Linear(hidden_size, num_countries)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        packed_embedded = rnn_utils.pack_padded_sequence(embedded, attention_mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_embedded)
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        output = output[:, -1, :]  # Take the last hidden state
        output = self.dropout(output)
        gender = self.gender_classifier(output)
        ethnicity = self.ethnicity_classifier(output)
        country = self.country_classifier(output)
        return gender, ethnicity, country

vocab_size = len(vocab)
embedding_dim = 128
hidden_size = 256
lstm_model = LSTMNameClassifier(vocab_size, embedding_dim, hidden_size, num_genders, num_ethnicities, num_countries, dropout_rate=0.1)

#%% md
# RNN Model
#%%

class RNNNameClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_genders, num_ethnicities, num_countries, dropout_rate=0.1):
        super(RNNNameClassifier, self
