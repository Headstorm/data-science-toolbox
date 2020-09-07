from sklearn.feature_extraction.text import CountVectorizer
import re
import string

# import and download neccessary nltk package
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


def import_file(filename):
    return open(filename, 'rt').read()


def tokenize(text):
    return text.split()


def to_lowercase(text):
    return text.lower()


def replace_newline_by_space(text):
    return text.replace('\n', ' ').replace('\r', '')


def replace_x_by_tag(text):
    # remove URL from text. substitute the matched string in URL with 'URL' tag.
    URL = re.compile(
        '(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?')
    text = URL.sub('URL', text)

    # remove email from text. substitute the matched string in EMAIL with 'EMAIL' tag.
    EMAIL = re.compile('^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$')
    text = EMAIL.sub('EMAIL', text)

    return text


def remove_punctuation(text):
    # remove string punctuation
    PUNCTUATION = str.maketrans("", "", string.punctuation)
    return text.translate(PUNCTUATION)


def remove_stopwords(tokens):
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if not t in stop_words]


def stemming(tokens):
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    return [ps.stem(t) for t in tokens]


def lemmatizing(tokens):
    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    return [lem.lemmatize(t) for t in tokens]


def count_vector(text):
    vectorizer = CountVectorizer()
    vectorizer.fit(text)
    # summarize
    print(vectorizer.vocabulary_)
    # encode document
    vector = vectorizer.transform(text)
    return vector


if __name__ == '__main__':
    # load text
    text = import_file('pg5200.txt')

    # replace newline by space
    text = replace_newline_by_space(text)

    # to lowercase
    text = to_lowercase(text)

    # replace x by space
    text = replace_x_by_tag(text)

    # remove punctuation
    text = remove_punctuation(text)

    # tokenize text
    tokens = tokenize(text)

    # remove stopwords
    tokens = remove_stopwords(tokens)

    # stemming the words
    tokens = stemming(tokens)

    # lemmatizing the words
    tokens = lemmatizing(tokens)

    # bag of words
    vector = count_vector(tokens)

    # summarize encoded vector
    print(vector.shape)
    print(type(vector))
    print(vector.toarray())
