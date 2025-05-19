from sentence_transformers import SentenceTransformer, util
import Levenshtein
from transformers import BertTokenizer, BertModel
import torch
import re
import torch
import nltk
import numpy as np
from gensim.models import KeyedVectors, FastText
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import spacy
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Λεξικό συνωνύμων / παραφρασμένων εκφράσεων
replacements = {
    "dragon boat festival": "Dragon Boat Festival",
    "in our Chinese culture": "in Chinese tradition",
    "to celebrate it with all safe and great in our lives": "to celebrate it with safety and joy in our lives",
    "I am very appreciated the full support of the professor": "I truly appreciate the professor's full support",
    "for our Springer proceedings publication": "regarding the publication in the Springer proceedings"
}

# Συνάρτηση για ανακατασκευή
def simple_paraphrase(text: str) -> str:
    for phrase, replacement in replacements.items():
        if phrase in text:
            text = text.replace(phrase, replacement)
    return text

# Κείμενο εισόδου
text1 = "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives."
text2 = "I am very appreciated the full support of the professor, for our Springer proceedings publication."
text1_b=simple_paraphrase(text1)
text2_b=simple_paraphrase(text2)


# Εφαρμογή παραφραστικού αλγορίθμου
print("🔹 Original 1:", text1)
print("🔁 Paraphrased 1:", text1_b)
print()
print("🔹 Original 2:", text2)
print("🔁 Paraphrased 2:", text2_b)


embeddings = model.encode([text1, text1_b, text2,text2_b], convert_to_tensor=True)


sim_1 = util.cos_sim(embeddings[0], embeddings[1]).item()
sim_2 = util.cos_sim(embeddings[2], embeddings[3]).item()
print()
print()

print(f"Similarity (Original1 vs Paraphrase 1): {sim_1:.4f}")
print(f"Similarity (Original2 vs Paraphrase 2): {sim_2:.4f}")

def levenshtein_similarity(text1, text2):
    # Υπολογισμός της Levenshtein Distance
    lev_distance = Levenshtein.distance(text1, text2)
    # Κανονικοποιημένη ομοιότητα (0 - 1)
    similarity = 1 - lev_distance / max(len(text1), len(text2))
    return similarity

dis_1 = levenshtein_similarity(text1, text1_b)
dis_2 = levenshtein_similarity(text2, text2_b)
print()

print(f"Levenshtein Distance: (Text1 vs  Paraphrase 1): {dis_1:.4f}")
print(f"Levenshtein Distance: (Text1 vs  Paraphrase 1): {dis_2:.4f}")
print()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def encode_bert(texts):
    # Tokenize all texts and apply padding & truncation
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)

    with torch.no_grad():
        outputs = model(**tokens)

    # Use mean of the last hidden states to create embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings



embeddings = torch.stack([encode_bert(text1), encode_bert(text1_b), encode_bert(text2), encode_bert(text2_b)])


sim_1 = torch.cosine_similarity(embeddings[0], embeddings[1]).item()
sim_2 = torch.cosine_similarity(embeddings[2], embeddings[3]).item()



print(f"BERT Similarity (Text1 vs  Paraphrase 1): {sim_1:.4f}")
print(f"BERT Similarity (Text1 vs  Paraphrase 1): {sim_2:.4f}")


originals, paraphrases=[],[]


originals.append(text1)
originals.append(text2)

paraphrases.append(text1_b)
paraphrases.append(text2_b)

w2v = Word2Vec([t.split() for t in originals+paraphrases], vector_size=100, window=5, min_count=1)


ft = FastText(vector_size=100, window=3, min_count=1)
ft.build_vocab([t.split() for t in originals+paraphrases])
ft.train([t.split() for t in originals+paraphrases], total_examples=len(originals+paraphrases), epochs=5)
#   3.4 BERT embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


def get_bert_emb(text):
    toks = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        out = bert_model(**toks)
    return out.last_hidden_state.mean(dim=1).cpu().numpy()[0]

# Συνάρτηση μέσου όρου embeddings λέξεων για word-level μεθόδους
def mean_embedding(words, model_wv):
    vecs = [model_wv[w] for w in words if w in model_wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model_wv.vector_size)

methods = {}


for name, embed_func in [
    ("Word2Vec", lambda txt: mean_embedding(txt.split(), w2v.wv)),

    ("FastText", lambda txt: mean_embedding(txt.split(), ft.wv))

]:
    sims = []
    for orig, para in zip(originals, paraphrases):
        emb_o = embed_func(orig)
        emb_p = embed_func(para)
        sim = cosine_similarity([emb_o], [emb_p])[0][0]
        sims.append(sim)
    methods[name] = sims

print()
for name, sims in methods.items():
    print(f"\n– {name}:")
    for i, sim in enumerate(sims, 1):
        print(f"  Pair {i}: similarity = {sim:.4f}")


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()






def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and t.isalpha()]
    return ' '.join(lemmas)


all_texts = [preprocess(t) for t in originals+paraphrases]
tfidf = TfidfVectorizer()
tfidf.fit(all_texts)
vocab = tfidf.get_feature_names_out()


# 4. Υπολογισμός cosine similarity
methods = {}
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

for name, embed_func in [
    ("Word2Vec", lambda txt: mean_embedding(txt.split(), w2v.wv)),
    #("GloVe",  lambda txt: mean_embedding(txt.split(), glove)),
    ("FastText", lambda txt: mean_embedding(txt.split(), ft.wv)),
    ("BERT", get_bert_emb)
]:
    sims = []
    for orig, para in zip(originals, paraphrases):
        emb_o = embed_func(preprocess(orig))
        emb_p = embed_func(preprocess(para))
        sim = cosine_similarity([emb_o], [emb_p])[0][0]
        sims.append(sim)
    methods[name] = sims

    # Sentence-BERT cosine similarity
    sbert_sims = []
    for orig, para in zip(originals, paraphrases):
        emb1 = sbert_model.encode(orig, convert_to_tensor=True)
        emb2 = sbert_model.encode(para, convert_to_tensor=True)
        sim = util.cos_sim(emb1, emb2).item()
        sbert_sims.append(sim)
    methods["cos_sim"] = sbert_sims

    # Levenshtein Distance normalized
    lev_sims = []
    for orig, para in zip(originals, paraphrases):
        lev = levenshtein_similarity(orig, para)

        lev_sims.append(lev)
    methods["Levenshtein"] = lev_sims
# 5. Παρουσίαση Αποτελεσμάτων

print("Cosine similarity (original vs paraphrase) after preprocess:\n")
for name, sims in methods.items():
    print(f"\n– {name}:")
    for i, sim in enumerate(sims, 1):
        print(f"  Pair {i}: similarity = {sim:.4f}")


