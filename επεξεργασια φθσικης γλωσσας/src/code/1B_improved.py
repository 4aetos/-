import nltk
import torch
import numpy as np
import re

from nltk.corpus import wordnet, stopwords, words
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    pipeline,
    T5Tokenizer,
    BertTokenizer,
    BertModel
)
from sentence_transformers import SentenceTransformer, util
from gensim.models import Word2Vec, FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import spacy
import Levenshtein

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

# Load language models
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Paraphrasing models
t5_paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")
pegasus_tokenizer = PegasusTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
pegasus_model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
pegasus_pipeline = pipeline("text2text-generation", model=pegasus_model, tokenizer=pegasus_tokenizer, truncation=True)

# Sentence transformer
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Input texts
texts = [
    """Today is our dragon boat festival...""",
    """During our final discuss, I told him about the new submission..."""
]


def simplify_text(text):
    doc = nlp(text)
    simplified = []
    for sent in doc.sents:
        words = [token.lemma_ if token.pos_ in ["VERB", "NOUN"] else token.text for token in sent]
        cleaned = ' '.join(sorted(set(words), key=words.index))
        simplified.append(cleaned.strip().capitalize() + '.')
    return ' '.join(simplified)


def paraphrase_t5(text):
    return ' '.join([
        t5_paraphraser(f"paraphrase: {sent}", max_length=512, num_return_sequences=1, do_sample=True)[0]['generated_text']
        for sent in sent_tokenize(text)
    ])


def paraphrase_pegasus(text):
    return ' '.join([
        pegasus_pipeline(sent)[0]['generated_text']
        for sent in sent_tokenize(text)
    ])


def cosine_sim(a, b):
    return cosine_similarity([a], [b])[0][0]


def levenshtein_similarity(a, b):
    return 1 - Levenshtein.distance(a, b) / max(len(a), len(b))


def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def preprocess(text):
    doc = nlp(text.lower())
    return [t.lemma_ for t in doc if not t.is_stop and not t.is_punct] or [t.text for t in doc if not t.is_punct]


def mean_embedding(tokens, model, normalize_embeddings=True):
    vectors = [model[tok] for tok in tokens if tok in model]
    if not vectors:
        return np.zeros(model.vector_size)
    mean_vec = np.mean(vectors, axis=0)
    return normalize(mean_vec.reshape(1, -1)).flatten() if normalize_embeddings else mean_vec


def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = bert_model(**inputs).last_hidden_state.mean(dim=1)
    return output.squeeze().cpu().numpy()


# Run paraphrasing
paraphrased_versions = []
for text in texts:
    paraphrased_versions.extend([
        simplify_text(text),
        paraphrase_t5(text),
        paraphrase_pegasus(text)
    ])

# Compute similarities (originals vs paraphrases)
all_versions = paraphrased_versions + texts  # Last two are originals
originals = texts
paraphrases = paraphrased_versions

# Sentence-BERT similarities
sbert_embeddings = sbert_model.encode(all_versions, convert_to_tensor=True)
for i in range(3):
    print(f"Cosine SBERT Similarity (Text1 vs Paraphrase {i+1}): {util.cos_sim(sbert_embeddings[-2], sbert_embeddings[i]).item():.4f}")
    print(f"Cosine SBERT Similarity (Text2 vs Paraphrase {i+1}): {util.cos_sim(sbert_embeddings[-1], sbert_embeddings[i+3]).item():.4f}")

print()

# Levenshtein similarities
for i in range(3):
    print(f"Levenshtein (Text1 vs Paraphrase {i+1}): {levenshtein_similarity(texts[0], paraphrases[i]):.4f}")
    print(f"Levenshtein (Text2 vs Paraphrase {i+1}): {levenshtein_similarity(texts[1], paraphrases[i+3]):.4f}")

print()

# BERT cosine similarities
bert_embeddings = [get_bert_embedding(t) for t in all_versions]
for i in range(3):
    print(f"BERT Cosine (Text1 vs Paraphrase {i+1}): {cosine_sim(bert_embeddings[-2], bert_embeddings[i]):.4f}")
    print(f"BERT Cosine (Text2 vs Paraphrase {i+1}): {cosine_sim(bert_embeddings[-1], bert_embeddings[i+3]):.4f}")

print()

# Word embeddings (Word2Vec, FastText)
corpus = [preprocess(t) for t in originals + paraphrases]
w2v = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1)
ft = FastText(vector_size=100, window=3, min_count=1)
ft.build_vocab(corpus)
ft.train(corpus, total_examples=len(corpus), epochs=5)

methods = {
    "Word2Vec": lambda tokens: mean_embedding(tokens, w2v.wv),
    "FastText": lambda tokens: mean_embedding(tokens, ft.wv),
    "BERT": lambda tokens: get_bert_embedding(" ".join(tokens)),
    "SBERT": lambda tokens: sbert_model.encode(tokens, convert_to_tensor=False),
    "Levenshtein": None  # special case
}

for name, embed_func in methods.items():
    print(f"\n{name} Similarities:")
    for i, orig in enumerate(originals):
        for j in range(3):
            para = paraphrases[i * 3 + j]
            if name == "Levenshtein":
                sim = levenshtein_similarity(" ".join(preprocess(orig)), " ".join(preprocess(para)))
            else:
                emb_o = embed_func(preprocess(orig))
                emb_p = embed_func(preprocess(para))
                sim = cosine_sim(emb_o, emb_p)
            print(f"Text{i+1} vs {['Spacy','T5','Pegasus'][j]}: {sim:.4f}")
