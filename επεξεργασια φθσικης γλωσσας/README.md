Α)Σε αυτό το έργο, υλοποιείται μια διαδικασία επεξεργασίας και αξιολόγησης παραφρασμένων κειμένων, με στόχο να υπολογιστεί η ομοιότητά τους με τα αρχικά, χρησιμοποιώντας διάφορες μεθόδους NLP.

🔁 1. Παραφραστική Αντικατάσταση
Χρησιμοποιείται ένα λεξικό (replacements) με προκαθορισμένες εκφράσεις και τις αντίστοιχες παραφράσεις τους. Αυτές εφαρμόζονται στα αρχικά κείμενα μέσω της συνάρτησης simple_paraphrase.
(για το Β απλά χρησιμοποιούμε βιβλιοθήκες για να ανακατασκεβάζουν το κείμενο)

📐 2. Υπολογισμός Ομοιότητας με Sentence-BERT
Χρησιμοποιείται το μοντέλο Sentence-BERT (all-MiniLM-L6-v2) για τη δημιουργία embeddings ολόκληρων προτάσεων και την εξαγωγή της ομοιότητας cosine (cos_sim) μεταξύ των αρχικών και παραφρασμένων εκδοχών.

🔤 3. Levenshtein Distance (Λεξιλογική Ομοιότητα)
Η απόσταση Levenshtein μετρά πόσες αλλαγές χαρακτήρων χρειάζονται για να μετατραπεί η μία πρόταση στην άλλη. Η τιμή κανονικοποιείται ώστε να κυμαίνεται από 0 έως 1, με το 1 να σημαίνει απόλυτη ταύτιση.

🤖 4. BERT-based Embeddings
Χρησιμοποιείται το προκαθορισμένο μοντέλο BERT για τη δημιουργία ενσωματώσεων μέσω του μέσου όρου των τελευταίων κρυφών καταστάσεων. Υπολογίζεται η ομοιότητα cosine μεταξύ των προτάσεων.

🧬 5. Word Embedding Models (Word2Vec & FastText)
Τα κείμενα χωρίζονται σε λέξεις και δημιουργούνται ενσωματώσεις λέξεων με τα μοντέλα Word2Vec και FastText. Για κάθε πρόταση, υπολογίζεται το μέσο διάνυσμα, και από αυτό η ομοιότητα cosine.

🧹 6. Προεπεξεργασία Κειμένου
Πραγματοποιείται λεξικογραφική καθαριότητα με:

Αφαίρεση σημείων στίξης

Μετατροπή σε πεζά

Λεμματοποίηση με χρήση του WordNet Lemmatizer

Αφαίρεση stopwords (π.χ. "the", "is")

Τα καθαρισμένα κείμενα χρησιμοποιούνται σε επιπλέον συγκρίσεις ομοιότητας.

📊 7. Παρουσίαση Αποτελεσμάτων
Για κάθε μέθοδο, εμφανίζονται οι βαθμολογίες ομοιότητας για κάθε ζευγάρι αρχικού-παραφρασμένου κειμένου, ώστε να αξιολογηθεί η επίδοση των διαφορετικών τεχνικών:

cos_sim

Levenshtein

BERT Embeddings

Word2Vec

FastText
