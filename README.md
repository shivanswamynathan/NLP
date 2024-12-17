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


## Comparison Between Bow,Glove and Word2vec: IMDB review dataset
## BOW
![image](https://github.com/user-attachments/assets/83522595-0acc-4dc0-9bee-2345644eb811)
![image](https://github.com/user-attachments/assets/f9b16b1a-5b01-4273-93da-0649c9a364fc)

## Glove
![image](https://github.com/user-attachments/assets/0963f590-3424-469e-bdb6-23f3da066d5f)
![image](https://github.com/user-attachments/assets/f760f4e4-889b-4c73-a302-c90bd3fb51fc)

## Word2vec
![image](https://github.com/user-attachments/assets/4614c9fc-b445-4650-92eb-5cbbf8e998c4)
![image](https://github.com/user-attachments/assets/cc9ec8d5-0eda-44ee-bf7a-eb2a981e56a7)





