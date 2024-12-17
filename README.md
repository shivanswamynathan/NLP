# **Documentation for IMDB Review Classification**

## **1. Objective**
The primary objective of this project is to classify movie reviews from the IMDB dataset as either positive or negative using Natural Language Processing (NLP) techniques. The focus is on implementing and comparing different text vectorization methods—Bag of Words (BoW), GloVe (Global Vectors for Word Representation), and Word2Vec—along with their performance in terms of prediction outputs and evaluation metrics.

---

## **2. Dataset**
The dataset used for this project consists of labeled IMDB movie reviews.

- **Features**: Raw text of reviews  
- **Target**: Binary labels (Positive = 1, Negative = 0)

---

## **3. Methodology**

### **Data Preprocessing**
#### **Text Cleaning**:
- Removal of punctuation, numbers, and special characters.
- Conversion of text to lowercase.
- Tokenization and removal of stopwords.
- Stemming and Lemmatization (where applicable).

#### **Splitting**:
- Train-test split of the dataset to evaluate the model performance.

### **Feature Extraction Methods**
Three feature extraction methods were implemented to convert text into numerical representations for classification:

1. **Bag of Words (BoW)**
   - Represents text as a sparse matrix of word frequencies or binary values.
   - Captures term occurrence but ignores word semantics and context.

2. **GloVe**
   - Pre-trained word embeddings that capture semantic meaning.
   - Embedding vectors are trained on a large corpus (e.g., Wikipedia) and used to represent words in a dense vector space.

3. **Word2Vec**
   - Embeddings generated using Skip-Gram or CBOW (Continuous Bag of Words).
   - Captures semantic relationships between words through unsupervised learning.

### **Classification**
- A classification model (e.g., Logistic Regression, Random Forest, or any other specified) was trained using the vectorized features from each method.
- Predictions (`y_pred`) were compared across methods.

### **Evaluation Metrics**
- Accuracy
- Precision
- Recall
- F1-score

---

## **4. Results and Comparison**

### **1. Bag of Words (BoW)**
- **Matrix Size**: High-dimensional sparse matrix, resulting in large memory usage.
- **Output**: Performs well for simpler tasks but lacks semantic understanding.
- **Prediction (`y_pred`)**: Captures word frequency but ignores context, leading to potential misclassifications.
![image](https://github.com/user-attachments/assets/83522595-0acc-4dc0-9bee-2345644eb811)
![image](https://github.com/user-attachments/assets/f9b16b1a-5b01-4273-93da-0649c9a364fc)

### **2. GloVe**
- **Matrix Size**: Dense, lower-dimensional vectors compared to BoW.
- **Output**: Captures semantic relationships effectively.
- **Prediction (`y_pred`)**: Demonstrates improved performance in sentiment detection, especially for context-dependent words.
![image](https://github.com/user-attachments/assets/0963f590-3424-469e-bdb6-23f3da066d5f)
![image](https://github.com/user-attachments/assets/f760f4e4-889b-4c73-a302-c90bd3fb51fc)

### **3. Word2Vec**
- **Matrix Size**: Dense vectors, customizable for task-specific training.
- **Output**: Captures semantic and syntactic information in text.
- **Prediction (`y_pred`)**: Slightly outperforms GloVe in certain cases due to its training methodology, particularly in context-heavy reviews.
![image](https://github.com/user-attachments/assets/4614c9fc-b445-4650-92eb-5cbbf8e998c4)
![image](https://github.com/user-attachments/assets/cc9ec8d5-0eda-44ee-bf7a-eb2a981e56a7)


## **5. Comparison of `y_pred` Values**

| **Method**   | **Key Strength**                      | **Limitation**                            | **Accuracy** |
|--------------|---------------------------------------|-------------------------------------------|--------------|
| **BoW**      | Simplicity and fast computation       | Ignores word context and semantics        | 0.85         |
| **GloVe**    | Pre-trained embeddings, semantic info | Limited to fixed embeddings               | 0.73         |
| **Word2Vec** | Task-specific embeddings, contextual  | Computationally expensive for training    | 0.81         |

---

## **6. Conclusion**

### **Summary of Findings**:
- **BoW** is suitable for straightforward classification tasks but lacks the sophistication required for complex datasets.
- **GloVe** and **Word2Vec** leverage semantic meaning, improving model performance for nuanced reviews.

### **Recommendation**:
- **GloVe** is a robust option when pre-trained embeddings suffice.
- **Word2Vec** is better for custom, task-specific embeddings when computational resources allow.

**Final Model**: The best-performing method based on accuracy and evaluation metrics was `<Best Method>`.

---

## **7. Future Work**

- Experiment with advanced models like BERT or GPT-based embeddings for further improvement.
- Explore hyperparameter tuning to optimize the classification model.

---

# Natural Language Processing (NLP) Techniques

## 1. Bag of Words (BoW)

### Description
The text is represented as a sparse vector of word counts or binary values. It does not consider word order or semantics.

### Disadvantages
- High-dimensional for large vocabularies (sparse representation).
- Struggles with contextual understanding.

### When to Use
- **Structured Text Data**: Works well with small or moderately sized datasets with predictable and structured text (e.g., reviews, simple classification tasks).  
  **Example**: Spam detection, sentiment analysis.
- **Text with Simple Patterns**: When the frequency of words matters more than their order or context.  
  **Example**: Classifying product reviews based on specific keywords (e.g., "good", "bad").

### When NOT to Use
- **Large and Complex Datasets**: Struggles with high-dimensional data due to sparse representation.  
  **Example**: Datasets with millions of unique words (e.g., scientific articles or legal documents).
- **Context-Dependent Tasks**: When understanding word relationships or context is essential.  
  **Example**: Tasks involving polysemous words like "bank" (river bank vs financial bank).

---

## 2. GloVe (Global Vectors for Word Representation)

### Description
GloVe creates pre-trained word embeddings based on co-occurrence statistics of words in a corpus. Each word is represented as a dense vector in a fixed-dimensional space.

### Key Features
- Trains on a co-occurrence matrix from a large corpus.
- Captures semantic relationships between words (e.g., "king - man + woman = queen").

### Disadvantages
- **Static Embeddings**: Each word has a single fixed vector, losing context (e.g., "bank" in different sentences).

### When to Use
- **Pre-trained Embeddings**: Effective for datasets where you don’t want to train embeddings from scratch.  
  **Example**: Similarity search, clustering words, or classification tasks where semantic relationships are important.
- **Moderately Large Datasets**: Works well when the dataset size aligns with the pre-trained embeddings’ corpus.  
  **Example**: Movie reviews, FAQs.

### When NOT to Use
- **Domain-Specific Datasets**: Pre-trained GloVe embeddings may not align with niche or specialized domains (e.g., medical or legal text).  
  **Solution**: Fine-tune pre-trained GloVe on domain-specific data or use domain-specific embeddings like BioWordVec or ClinicalBERT.
- **Contextual Tasks**: Fails when a single word has multiple meanings in different contexts.

---

## 3. Word2Vec

### Description
Word2Vec uses neural networks to learn dense (continuous) vector representations of words. These vectors capture semantic relationships, so words with similar meanings or contexts have similar vector representations.

### Methods
1. **CBOW (Continuous Bag of Words)**: Predicts the target word (center word) based on surrounding context words.  
   **Example**: In the sentence "The cat sits on the mat," CBOW uses context words ("The", "cat", "on", "the", "mat") to predict the target word "sits".

2. **Skip-gram**: Predicts context words based on a given word (target word).  
   **Example**: In the sentence "The cat sits on the mat," Skip-gram uses the target word "sits" to predict the context words ("The", "cat", "on", "the", "mat").

### Key Differences
- **CBOW**: Faster and works well for smaller datasets.
- **Skip-gram**: Better for larger datasets, works well for rare words, and captures detailed word relationships.

### When to Use
- **Domain-Specific Datasets**: When you can train embeddings on a large, domain-specific corpus.  
  **Example**: E-commerce product reviews, medical terminology, or scientific papers.
- **Semantic Understanding**: For applications requiring semantic and syntactic relationships between words.  
  **Example**: Word analogy tasks or language modeling.

### When NOT to Use
- **Small Datasets**: Word2Vec requires a large corpus to generate meaningful embeddings.  
  **Example**: Small datasets with a limited vocabulary.

---

## 4. BERT (Bidirectional Encoder Representations from Transformers)

### Description
BERT uses the Transformer architecture to generate contextualized embeddings. Each word’s embedding depends on the entire sentence. It is pre-trained using:
- **Masked Language Modeling (MLM)**: Predicts missing words in a sentence.
- **Next Sentence Prediction (NSP)**: Learns relationships between sentences.

### Key Features
- **Contextual Embeddings**: Same word has different vectors depending on the context.
- Supports fine-tuning for specific tasks.

### When to Use
- **Contextual Understanding**: For datasets requiring understanding of word meaning based on surrounding text.  
  **Example**: Question answering, sentiment analysis, named entity recognition.
- **Advanced NLP Tasks**: When the task involves complex relationships between words and context.  
  **Example**: Summarization, translation, text generation.
- **Pre-Trained Knowledge**: For fine-tuning on tasks with limited data but rich context.  
  **Example**: Small datasets with specific tasks, leveraging transfer learning.

### When NOT to Use
1. **Simple or Structured Datasets**: Overkill for simple tasks like word frequency analysis or basic classification.  
   **Example**: Binary classification for spam detection based on a small vocabulary.
2. **Computational Constraints**: BERT is computationally intensive, requiring significant memory and processing power.  
   **Example**: Real-time inference on edge devices like smartphones or IoT devices.
3. **Data-Specific Contexts Not Captured in Pre-Trained Models**: BERT may miss domain-specific nuances unless fine-tuned.  
   **Example**: Legal or medical datasets without further fine-tuning.

---

## Summary: Comparison of NLP Models

| Model    | Key Features                                  | Disadvantages                            | Best Use Cases                          |
|----------|----------------------------------------------|------------------------------------------|-----------------------------------------|
| BoW      | Sparse vector representation of word counts. | High-dimensional and lacks context.      | Simple text data, spam detection.       |
| GloVe    | Pre-trained dense embeddings; semantic focus.| Static embeddings; lacks context.        | General NLP tasks, similarity search.   |
| Word2Vec | Learns dense vectors; context-aware.         | Requires large datasets for training.    | Domain-specific corpora, rare words.    |
| BERT     | Contextual embeddings; Transformer-based.    | Computationally intensive; overkill.     | Advanced NLP tasks, contextual analysis.|



# Summary Evaluation Metrics

## 1. ROUGE Scores
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures how much overlap exists between the generated (candidate) summary and the reference (original/ideal) summary.

### ROUGE Metrics:
- **ROUGE-1**: Measures the overlap of unigrams (single words) between the candidate summary and the reference text.
- **ROUGE-2**: Measures the overlap of bigrams (two consecutive words) between the candidate and the reference.
- **ROUGE-L**: Measures the longest common subsequence (LCS) of words between the candidate and the reference. It evaluates fluency and coherence.

### ROUGE Metrics Explained:
Each ROUGE metric provides three values:
- **Precision**: Percentage of words in the candidate summary that are also in the reference.
- **Recall**: Percentage of words in the reference that appear in the candidate.
- **F-measure**: Harmonic mean of Precision and Recall, balancing both.

### Your ROUGE Scores:
#### ROUGE-1:
- **Precision**: 1.0 (Every unigram in the candidate exists in the reference.)
- **Recall**: 0.1337 (Only 13.37% of the unigrams in the reference appear in the candidate.)
- **F-measure**: 0.2358 (Moderate overlap, slightly favoring precision.)

#### ROUGE-2:
- **Precision**: 0.9079 (Most bigrams in the candidate are in the reference.)
- **Recall**: 0.12 (Only 12% of reference bigrams appear in the candidate.)
- **F-measure**: 0.2120 (Similar pattern as ROUGE-1, slightly favoring precision.)

#### ROUGE-L:
- **Precision**: 0.8831 (Most of the candidate's sequence is found in the reference.)
- **Recall**: 0.1181 (Only 11.81% of the reference's sequence appears in the candidate.)
- **F-measure**: 0.2083 (Again, precision is higher than recall.)

### Observations:
- **High precision** across all ROUGE scores means the candidate summary contains relevant and correct information.
- **Low recall** indicates the candidate summary is much shorter than the reference, missing some important details.

---

## 2. BLEU Score
BLEU (Bilingual Evaluation Understudy) measures how close the generated summary is to the reference summary based on n-gram overlap. 

### BLEU Metrics Explained:
- The BLEU score compares the tokenized reference and candidate summaries.
- It calculates **precision** for n-grams (default: unigrams and bigrams) while penalizing overly short candidates using a **brevity penalty**.

### BLEU Score Results:
- BLEU scores range from **0** (no overlap) to **1.0** (perfect match).
- A **low BLEU score** suggests that the generated summary lacks sufficient overlap with the reference.

---

## Comparison and Insights

### ROUGE vs BLEU:
- **ROUGE** is recall-oriented, meaning it evaluates how much of the reference's content is covered by the candidate.
- **BLEU** is precision-oriented, focusing on how accurately the candidate matches the reference.

### Your Results:
- **ROUGE Precision** (1.0 for ROUGE-1) indicates the candidate summary uses terms found in the reference but doesn't capture all the content (low recall).
- The **low BLEU score** further suggests that while the candidate summary contains correct phrases, it fails to align strongly with the reference text in n-gram sequence and length.

---

## Practical Analysis
- If the candidate summary is very short, it can achieve high ROUGE precision but fail in BLEU and ROUGE recall because it omits many details.

### Improvement Tips:
1. Generate a slightly longer summary to increase recall and BLEU scores.
2. Use a model trained for more comprehensive summarization, such as fine-tuned **T5** or **PEGASUS** models.






