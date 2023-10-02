import pandas as pd
import numpy as np
import re

from sklearn.metrics import accuracy_score

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from transformers import AutoConfig

def load_model(model):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    return tokenizer, model, config

def preprocess_tweet(text):
    
    # remove hashtags but keep the word
    filtered_text = re.sub(r'\s?@\w+', '', text).replace('#', '')
    
    # remove http links
    filtered_text = re.sub(r'http\S+', '', filtered_text)
    
    # remove random strings (errors from scraping, i.e. r\&?)
    filtered_text = re.sub(r'\&\S+\s', '', filtered_text)
    
    #clean double spaces and floating whitespaces
    filtered_text = re.sub(r'\s+', ' ', filtered_text).strip().lstrip().rstrip()
    
    return filtered_text

def get_sentiment(text, tokenizer, model, config):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    
    # negative, neutral, positive
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    #print(filtered_text)

    # we only need the last item
    ranking = np.argsort(scores)
    
    # return the correct label and the corresponding score    
    return config.id2label[ranking[2]], scores[ranking[2]]

def run_sentiment_on_test(tokenizer, model, config):
    # Reading a CSV file using pandas
    df = pd.read_csv('sentiment_test_cases.csv')

    pred_sentiment = []
    pred_score = []

    for x in range(len(df)):
        text = preprocess_tweet(df.loc[x]['text'])
        label, score = get_sentiment(text, tokenizer, model, config)

        pred_sentiment.append(label)
        pred_score.append(score)

    actual = df['expected_sentiment'].tolist()
    accuracy = accuracy_score(actual, pred_sentiment)

    print('Computed accuracy:', "{:.2f}%".format(accuracy*100))

def run_sentiment_on_input(sent, tokenizer, model, config):
    output = {}

    text = preprocess_tweet(sent)
    label, score = get_sentiment(text, tokenizer, model, config)

    output['model_output'] = label
    output['confidence_score'] = score

    return output

if __name__ == "__main__":

    # load roberta model
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer, model, config = load_model(model_name)

    sent = input('\nEnter input sentence: ')

    if sent == 'run_test':
        run_sentiment_on_test(tokenizer, model, config)
    else:
        output = run_sentiment_on_input(sent, tokenizer, model, config)

        print(output)

    
   

    