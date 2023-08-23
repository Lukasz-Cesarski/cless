from typing import Tuple, Optional, List
from gensim.models import Phrases
import pyLDAvis.gensim_models
from pyLDAvis._prepare import PreparedData
from wordcloud import WordCloud, STOPWORDS
from nltk import RegexpTokenizer, WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from datetime import datetime


import seaborn as sns
sns.set_theme(style="whitegrid")

def docs_preprocessor(docs: List[str]) -> List[List[str]]:
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    new_docs = []
    for doc in docs:
        doc = doc.lower()
        doc = tokenizer.tokenize(doc)
        # Remove numbers, but not words that contain numbers.
        doc = [token for token in doc if not token.isdigit()]
        # Remove words that are only one character.
        doc = [token for token in doc if len(token) > 3]
        doc = [lemmatizer.lemmatize(token) for token in doc]
        doc = [token for token in doc if token not in STOPWORDS]
        new_docs.append(doc)
    return new_docs


def extend_with_ngrams(docs: List[List[str]]) -> List[List[str]]:
    bigram = Phrases(docs, min_count=5)
    trigram = Phrases(bigram[docs])
    docs_ngrams = []
    for doc in docs:
        p_doc_ngrams = []
        p_doc_ngrams.extend(doc)
        ngrams = []
        for token in bigram[doc]:
            if '_' in token:
                # Token is a bigram, add to document.
                ngrams.append(token)
        for token in trigram[bigram[doc]]:
            if '_' in token:
                # Token is a ngram (bigram or trigram), add to document.
                ngrams.append(token)
        p_doc_ngrams.extend(ngrams)
        docs_ngrams.append(p_doc_ngrams)
    return docs_ngrams


def run_lda(docs: List[str]) -> PreparedData:
    """https://www.kaggle.com/code/subinium/showusthedata-topic-modeling-with-lda/notebook"""

    p_docs = docs_preprocessor(docs)
    ngram_docs = extend_with_ngrams(p_docs)

    # Create a dictionary representation of the documents.
    dictionary = Dictionary(ngram_docs)
    print('Number of unique words in initital documents:', len(dictionary))

    # Filter out words that occur less than 2 documents, or more than 80% of the documents.
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    print('Number of unique words after removing rare and common words:', len(dictionary))
    corpus = [dictionary.doc2bow(doc) for doc in ngram_docs]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    # Set training parameters.
    num_topics = 10
    chunksize = 500  # size of the doc looked at every pass
    passes = 20  # number of passes through documents
    iterations = 100
    eval_every = 1  # Don't evaluate model perplexity, takes too much time.

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    start = datetime.now()
    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto', eta='auto',
        iterations=iterations, num_topics=num_topics,
        passes=passes, eval_every=eval_every
    )
    print(f"LDA time: {datetime.now() - start}")
    pyLDAvis.enable_notebook()

    return pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
