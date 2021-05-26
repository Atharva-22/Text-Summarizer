import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from string import punctuation
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem.wordnet import WordNetLemmatizer

from collections import defaultdict
from heapq import nlargest
import operator


def nltk_summarizer(raw_text):
	stopWords = set(stopwords.words("english") + list(punctuation))
	sentences = sent_tokenize(raw_text)

	text = raw_text
	def tokenize(text):
		tokens = nltk.word_tokenize(raw_text)
		stems = []
		for item in tokens:
			stems.append(WordNetLemmatizer().lemmatize(item))
		return stems

	tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words=stopWords)
	tfs = tfidf.fit_transform([raw_text])

	freqs = {}
	feature_names = tfidf.get_feature_names()
	for col in tfs.nonzero()[1]:
		freqs[feature_names[col]] = tfs[0, col]

	important_sentences = defaultdict(int)

	for i, sentence in enumerate(sentences):
		for token in word_tokenize(sentence.lower()):
			if token in freqs:
				important_sentences[i] += freqs[token]
				
	number_sentences = int(len(sentences) * 0.3)
	index_important_sentences = nlargest(number_sentences,important_sentences,important_sentences.get)
	
	Summary = []
	for i in sorted(index_important_sentences):
		Summary.append(sentences[i])

	Summarized = TreebankWordDetokenizer().detokenize(Summary)

	return Summarized
