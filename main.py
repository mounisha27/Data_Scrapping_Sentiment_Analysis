import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to extract article text from a given URL
def extract_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the article title
    title = soup.find('title').get_text()

    # Find and extract the article text
    article_body = soup.find('div', class_='article-body')
    if article_body:
        article_text = ''
        for paragraph in article_body.find_all('p'):
            article_text += paragraph.get_text() + '\n'
        return title, article_text
    else:
        return None, None

# Function to clean the text
def clean_text(text):
    # Tokenize the text into words
    words = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]

    return ' '.join(words)

# Function to perform sentiment analysis
def perform_sentiment_analysis(cleaned_text):
    # Load positive and negative words
    positive_words = set()
    negative_words = set()

    master_dictionary_folder = 'MasterDictionary'

    # List all files in the MasterDictionary folder
    file_names = os.listdir(master_dictionary_folder)

    # Read the content of each file
    for file_name in file_names:
        file_path = os.path.join(master_dictionary_folder, file_name)
        with open(file_path, 'r') as file:
            content = file.read()
            if 'positive' in file_name.lower():
                positive_words.update(content.split('\n'))
            elif 'negative' in file_name.lower():
                negative_words.update(content.split('\n'))

    # Tokenize the cleaned text
    words = word_tokenize(cleaned_text.lower())

    # Calculate positive and negative scores
    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)

    # Calculate polarity score
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)

    # Calculate subjectivity score
    subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)

    return positive_score, negative_score, polarity_score, subjectivity_score

# Function to perform readability analysis
def perform_readability_analysis(text):
    # Tokenize text into sentences
    sentences = sent_tokenize(text)

    # Calculate average sentence length
    total_words = sum(len(word_tokenize(sentence)) for sentence in sentences)
    average_sentence_length = total_words / len(sentences)

    # Count complex words (words with more than 2 syllables)
    complex_words = [word for word in word_tokenize(text) if syllable_count(word) > 2]
    percentage_complex_words = len(complex_words) / len(word_tokenize(text)) * 100

    # Calculate fog index
    fog_index = 0.4 * (average_sentence_length + percentage_complex_words)

    # Calculate average number of words per sentence
    average_words_per_sentence = total_words / len(sentences)

    # Count total words
    word_count = len(word_tokenize(text))

    # Calculate syllable per word
    total_syllables = sum(syllable_count(word) for word in word_tokenize(text))
    syllable_per_word = total_syllables / word_count

    # Count personal pronouns
    personal_pronouns = sum(1 for word in word_tokenize(text) if word.lower() in ['i', 'we', 'my', 'ours', 'us'])

    # Calculate average word length
    average_word_length = sum(len(word) for word in word_tokenize(text)) / word_count

    return average_sentence_length, percentage_complex_words, fog_index, average_words_per_sentence, word_count, syllable_per_word, personal_pronouns, average_word_length

# Function to count syllables in a word
def syllable_count(word):
    word = word.lower()
    if word in ['a', 'i']:
        return 1
    count = 0
    vowels = 'aeiouy'
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if count == 0:
        count += 1
    return count

# Main function
def main():
    # Read input URLs from Excel file
    input_file = 'Input.xlsx'
    df = pd.read_excel(input_file)

    # Create a new DataFrame for storing the results
    columns = ['URL_ID', 'URL', 'Title', 'Article_Text', 'Positive_Score', 'Negative_Score', 'Polarity_Score', 'Subjectivity_Score',
               'Average_Sentence_Length', 'Percentage_of_Complex_Words', 'Fog_Index', 'Average_Number_of_Words_Per_Sentence',
               'Word_Count', 'Syllable_Per_Word', 'Personal_Pronouns', 'Average_Word_Length']
    results_df = pd.DataFrame(columns=columns)

    # Iterate over each URL in the input file
    for index, row in df.iterrows():
        url_id = row['URL_ID']
        url = row['URL']

        print(f"Processing URL: {url}")

        # Extract article text from the URL
        title, article_text = extract_article_text(url)

        if article_text:
            # Clean the article text
            cleaned_text = clean_text(article_text)

            # Perform sentiment analysis
            positive_score, negative_score, polarity_score, subjectivity_score = perform_sentiment_analysis(cleaned_text)

            # Perform readability analysis
            (average_sentence_length, percentage_complex_words, fog_index, average_words_per_sentence,
             word_count, syllable_per_word, personal_pronouns, average_word_length) = perform_readability_analysis(article_text)

            # Append the results to the DataFrame
            results_df = results_df.append({'URL_ID': url_id, 'URL': url, 'Title': title, 'Article_Text': article_text,
                                            'Positive_Score': positive_score, 'Negative_Score': negative_score,
                                            'Polarity_Score': polarity_score, 'Subjectivity_Score': subjectivity_score,
                                            'Average_Sentence_Length': average_sentence_length,
                                            'Percentage_of_Complex_Words': percentage_complex_words,
                                            'Fog_Index': fog_index, 'Average_Number_of_Words_Per_Sentence': average_words_per_sentence,
                                            'Word_Count': word_count, 'Syllable_Per_Word': syllable_per_word,
                                            'Personal_Pronouns': personal_pronouns, 'Average_Word_Length': average_word_length}, 
                                           ignore_index=True)

    # Save the results to an Excel file
    output_file = 'Output.xlsx'
    results_df.to_excel(output_file, index=False)

if __name__ == "__main__":
    main()
