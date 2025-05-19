import nltk
from nltk.corpus import wordnet,words
from nltk.tokenize import word_tokenize
import random
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import pipeline
from transformers import T5Tokenizer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk import pos_tag
import spacy
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
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.preprocessing import normalize

nlp = spacy.load("en_core_web_sm")
# Ensure required NLTK corpora are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

texts = [
    """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives.
    Hope you too, to enjoy it as my deepest wishes. Thank your message to show our words to the doctor, as his next contract checking, to all of us.
    I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago.
    I am very appreciated the full support of the professor, for our Springer proceedings publication.""",

    """During our final discuss, I told him about the new submission — the one we were waiting since last autumn,
    but the updates was confusing as it not included the full feedback from reviewer or maybe editor?
    Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation.
    We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think.
    Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again.
    Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
    Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets."""
]



def simplify_text(text):
    if isinstance(text, list):
        text = " ".join(text)  # Αν κατά λάθος δώσεις λίστα

    doc = nlp(text)
    new_sentences = []

    for sent in doc.sents:
        words = [token.lemma_ if token.pos_ in ["VERB", "NOUN"] else token.text for token in sent]
        cleaned = ' '.join(sorted(set(words), key=words.index))
        new_sentences.append(cleaned.strip().capitalize() + '.')

    return ' '.join(new_sentences)



# Load the T5 paraphraser model from Hugging Face
paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

# Load the Pegasus model for paraphrasing
model_name = "tuner007/pegasus_paraphrase"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

nlp_p = pipeline("text2text-generation", model=model,tokenizer=tokenizer,truncation=True)

All=[]
# Example usage
for i, text in enumerate(texts, 1):
    print(f"Text {i}: {text}")
    print()
    # Synonym paraphrasing
    All.append(simplify_text(text))
    print("Spacy Paraphrase:", All[-1])
    print()

    # T5 Model paraphrasing


    sentences=nltk.sent_tokenize(text)

    paraphrases = []

    for sentence in sentences:
      paraphrase = paraphraser(f"paraphrase: {sentence}", max_length=512, num_return_sequences=3, do_sample=True)
      paraphrases.append(paraphrase[0]['generated_text'])

    paraphrased_text = ' '.join(paraphrases)
    print("Τ5 Paraphrase:", paraphrased_text)
    All.append(paraphrased_text)
    print()


    array1=[]
    array2=[]
    for i in range (0,len(sentences)-1):
        array1.append(nlp_p(sentences[i]))
    # Pegasus Model paraphrasing

    for i in range (len(array1)):
          array2.append(array1[i][0]['generated_text'])



    para=' '.join(array2)
    print(f"Pegasus Paraphrase: {para}")
    All.append(para)

    print()


All.append(texts[0])
All.append(texts[1])

model = SentenceTransformer('all-MiniLM-L6-v2')


embeddings = model.encode(All, convert_to_tensor=True)

sim_1 = util.cos_sim(embeddings[-2], embeddings[0]).item()
sim_2 = util.cos_sim(embeddings[-2], embeddings[1]).item()
sim_3 = util.cos_sim(embeddings[-2], embeddings[2]).item()



sim_1b = util.cos_sim(embeddings[-1], embeddings[3]).item()
sim_2b = util.cos_sim(embeddings[-1], embeddings[4]).item()
sim_3b = util.cos_sim(embeddings[-1], embeddings[5]).item()


print(f"Cosine Similarity (Text1 vs Spacy Paraphrase 1): {sim_1:.4f}")
print(f"Cosine Similarity (Text1 vs T5 Paraphrase 1): {sim_2:.4f}")
print(f"Cosine Similarity (Text1 vs Pegasus Paraphrase 1): {sim_3:.4f}")

print()

print(f"Cosine Similarity (Text2 vs Spacy Paraphrase 2): {sim_1b:.4f}")
print(f"Cosine Similarity (Text2 vs T5 Paraphrase 2): {sim_2b:.4f}")
print(f"Cosine Similarity (Text2 vs Pegasus Paraphrase 2): {sim_3b:.4f}")

print()
print()

def levenshtein_similarity(text1, text2):
    # Υπολογισμός της Levenshtein Distance
    lev_distance = Levenshtein.distance(text1, text2)
    # Κανονικοποιημένη ομοιότητα (0 - 1)
    similarity = 1 - lev_distance / max(len(text1), len(text2))
    return similarity


dis_1=levenshtein_similarity(All[-2],All[0])
dis_2=levenshtein_similarity(All[-2],All[1])
dis_3=levenshtein_similarity(All[-2],All[2])


dis_1b=levenshtein_similarity(All[-1],All[3])
dis_2b=levenshtein_similarity(All[-1],All[4])
dis_3b=levenshtein_similarity(All[-1],All[5])

print(f"Levenshtein Distance: (Text1 vs Spacy Paraphrase 1): {dis_1:.4f}")
print(f"Levenshtein Distance: (Text1 vs T5 Paraphrase 1): {dis_2:.4f}")
print(f"Levenshtein Distance: (Text1 vs Pegasus Paraphrase 1): {dis_3:.4f}")

print()

print(f"Levenshtein Distance: (Text2 vs Spacy Paraphrase 2): {dis_1b:.4f}")
print(f"Levenshtein Distance: (Text2 vs T5 Paraphrase 2): {dis_2b:.4f}")
print(f"Levenshtein Distance: (Text2 vs Pegasus Paraphrase 2): {dis_3b:.4f}")


print()
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



embeddings = torch.stack([encode_bert(text) for text in All])

sim_1 = torch.cosine_similarity(embeddings[-2], embeddings[0]).item()
sim_2 = torch.cosine_similarity(embeddings[-2], embeddings[1]).item()
sim_3 = torch.cosine_similarity(embeddings[-2], embeddings[2]).item()



sim_1b = torch.cosine_similarity(embeddings[-1], embeddings[3]).item()
sim_2b = torch.cosine_similarity(embeddings[-1], embeddings[4]).item()
sim_3b = torch.cosine_similarity(embeddings[-1], embeddings[5]).item()


print(f"BERT Similarity (Text1 vs Spacy Paraphrase 1): {sim_1:.4f}")
print(f"BERT Similarity (Text1 vs T5 Paraphrase 1): {sim_2:.4f}")
print(f"BERT Similarity (Text1 vs Pegasus Paraphrase 1): {sim_3:.4f}")

print()

print(f"BERT Similarity (Text2 vs Spacy Paraphrase 2): {sim_1b:.4f}")
print(f"BERT Similarity (Text2 vs T5 Paraphrase 2): {sim_2b:.4f}")
print(f"BERT Similarity (Text2 vs Pegasus Paraphrase 2): {sim_3b:.4f}")

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

originals, paraphrases=[],[]


originals = [texts[0], texts[1]]
paraphrases = [All[0], All[1], All[2], All[3], All[4], All[5]]

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


def mean_embedding(tokens, model, normalize_embeddings=True, min_word_count=1):
    """
    Υπολογίζει τη μέση τιμή των word embeddings για μία λίστα λέξεων.
    """
    valid_embeddings = []
    found_words = []  # Λίστα για τις λέξεις που βρέθηκαν στο μοντέλο

    for token in tokens:
        if token in model:
            valid_embeddings.append(model[token])
            found_words.append(token)  # Αποθηκεύουμε τις λέξεις που βρέθηκαν στο μοντέλο

    if not valid_embeddings:

        return np.zeros(model.vector_size)

    # Μέσος όρος των embeddings
    mean_vector = np.mean(valid_embeddings, axis=0)

    # Κανονικοποίηση, αν χρειάζεται
    if normalize_embeddings:
        mean_vector = normalize(mean_vector.reshape(1, -1)).flatten()




    return mean_vector


methods = {}


for name, embed_func in [
    ("Word2Vec", lambda txt:mean_embedding(simple_tokenize(txt), w2v.wv)),

    ("FastText", lambda txt: mean_embedding(simple_tokenize(txt), w2v.wv))

]:
    sims = []
    for i, orig in enumerate(originals):
        for j in range(3):
            para = paraphrases[i * 3 + j]
            emb_o = embed_func(orig)
            emb_p = embed_func(para)
            sim = cosine_similarity([emb_o], [emb_p])[0][0]
            sims.append(sim)
    methods[name] = sims

k_model=["Spacy Paraphrase","T5 Paraphrase","Pegasus Paraphrase"]
print()

for name, sims in methods.items():
    print()
    for i, sim in enumerate(sims, 1):

        if i <= 3:
            print(f"{name} Similarity (Text1 vs {k_model[i-1]} 1):  {sim:.4f}")
        else:
            if i==4:
                print()
            model_index = (i - 4) % 3  # για να ξεκινάει πάλι από 0
            print(f"{name} Similarity (Text2 vs {k_model[model_index]} 2):  {sim:.4f}")



stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()







def preprocess(text):
    doc = nlp(text.lower())
    tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct]
    if not tokens:
        tokens = [t.text for t in doc if not t.is_punct]
    return tokens


all_texts = [" ".join(preprocess(t)) for t in originals + paraphrases]
tfidf = TfidfVectorizer()
tfidf.fit(all_texts)
vocab = tfidf.get_feature_names_out()

corpus = [preprocess(t) for t in originals + paraphrases]

w2v = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1)

# Εκπαίδευση FastText
ft = FastText(vector_size=100, window=3, min_count=1)
ft.build_vocab(corpus)
ft.train(corpus, total_examples=len(corpus), epochs=5)


# 4. Υπολογισμός cosine similarity
methods = {}
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

for name, embed_func in [
    ("Word2Vec", lambda toks: mean_embedding(toks, w2v.wv)),
    ("FastText", lambda toks: mean_embedding(toks, ft.wv)),
    ("BERT", lambda text: get_bert_emb(" ".join(text)))
]:
    sims = []
    for i, orig in enumerate(originals):
        for j in range(3):
            para = paraphrases[i * 3 + j]
            emb_o = embed_func(preprocess(orig))
            emb_p = embed_func(preprocess(para))
            sim = cosine_similarity([emb_o], [emb_p])[0][0]
            sims.append(sim)
    methods[name] = sims

# Similarity
sbert_sims = []
for i, orig in enumerate(originals):
    for j in range(3):
        para = paraphrases[i * 3 + j]
        emb1 = sbert_model.encode(preprocess(orig), convert_to_tensor=True)
        emb2 = sbert_model.encode(preprocess(para), convert_to_tensor=True)
        sim = util.cos_sim(emb1, emb2).cpu().numpy()[0][0]
        sbert_sims.append(sim)
methods["Cosine"] = sbert_sims

# Levenshtein Distance
lev_sims = []
for i, orig in enumerate(originals):
    for j in range(3):
        para = paraphrases[i * 3 + j]
        lev = levenshtein_similarity(" ".join(preprocess(orig)), " ".join(preprocess(para)))
        lev_sims.append(lev)
methods["Levenshtein"] = lev_sims
# 5. Παρουσίαση Αποτελεσμάτων

print("After preprocess")
print()
print()

for name, sims in methods.items():
    print()
    for i, sim in enumerate(sims, 1):

        if i <= 3:
            print(f"{name}  Similarity (Text1 vs {k_model[i-1]} 1):  {sim:.4f}")
        else:
            if i==4:
                print()
            model_index = (i - 4) % 3  # για να ξεκινάει πάλι από 0
            print(f"{name} Similarity (Text2 vs {k_model[model_index]} 2):  {sim:.4f}")


