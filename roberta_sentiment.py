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

    # we only need the last item
    ranking = np.argsort(scores)
    
    # return the correct label and the corresponding score    
    return config.id2label[ranking[2]], scores[ranking[2]]

# Function for running sentiment analysis on test cases
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

    # Export the model output and confidence scores to csv
    df_final = pd.DataFrame({
        'text': df['text'].tolist(),
        'expected_sentiment': df['expected_sentiment'].tolist(),
        'model_output': pred_sentiment,
        'confidence_score': pred_score
    })

    df_final.to_csv('output_sentiment_test.csv')
    print('Exported results to output_sentiment_test.csv')

    # Compute and print the accuracy
    actual = df['expected_sentiment'].tolist()
    accuracy = accuracy_score(actual, pred_sentiment)

    print('Computed accuracy:', "{:.2f}%".format(accuracy*100))

# Function for running sentiment analysis on input sentence
def run_sentiment_on_input(sent, tokenizer, model, config):
    output = {}

    text = preprocess_tweet(sent)
    label, score = get_sentiment(text, tokenizer, model, config)

    output['model_output'] = label
    output['confidence_score'] = score

    return output

if __name__ == "__main__":

    # Load roberta model
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer, model, config = load_model(model_name)

    # Accept input sentence from user
    sent = input('\nEnter input sentence: ')

    # Using the special string 'run_test', run the model on test cases
    # else run sentiment analysis on text user has given
    while(1):
        sent = input('\nEnter input sentence: ')

        if sent == 'run_test':
            run_sentiment_on_test(tokenizer, model, config)
        elif sent == 'end_test':
            break
        else:
            output = run_sentiment_on_input(sent, tokenizer, model, config)

            print(output)

    
   

    