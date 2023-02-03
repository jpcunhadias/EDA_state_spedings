import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from collections import Counter


def get_most_common_nouns(df, column_name) -> list:
    """
    The get_most_common_nouns function takes a dataframe and column name as input.
    It then extracts nouns from the text in that column, flattens the list of verbs into a single list,
    and counts the frequency of each verb. It returns this most common verbs as an output.

    :param df: Pass in the dataframe that contains the text
    :param column_name: Specify the column name of the text data
    :return: A list of tuples containing the most common verbs and their frequencys
    :doc-author: Trelent
    """

    def extract_nouns(text):


        words = nltk.word_tokenize(text)
        tagged = pos_tag(words)

        #
        verbs = [word for word, pos in tagged if pos == 'NNP']
        return verbs

    # Apply the function to the text column
    df['verbs'] = df[column_name].apply(extract_nouns)

    # Flatten the list of verbs into a single list
    flat_list = [item for sublist in df['verbs'].tolist() for item in sublist]

    # Count the frequency of each verb
    verb_counts = Counter(flat_list)

    # Get the most common verbs
    most_common_verbs = verb_counts.most_common()

    return most_common_verbs