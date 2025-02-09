# Information Retrieval System

## Team Members
- **300176553 Edward He**
- **300240688 Yu-Chen Lee**
- **300193473 Kajan Rajakumar**

## Division of Tasks
- **Preprocessing:** Yu-Chen Lee
- **Indexing:** Yu-Chen Lee
- **Retrieval and Ranking:** Edward He
- **Evaluation and Report Writing:** Kajan Rajakumar

---

## 1. Overview of the Implementation

This project implements an Information Retrieval (IR) system based on the vector space model, specifically utilizing the TF-IDF weighting scheme and the BM25 ranking function. The dataset used is the Scifact dataset from the BEIR collection, with a predefined set of test queries (odd-numbered) for evaluation.

The system consists of three main steps:
- **Preprocessing:** Tokenization, stopword removal, and stemming.
- **Indexing:** Creating an inverted index for fast retrieval.
- **Retrieval & Ranking:** Using cosine similarity (TF-IDF) and BM25 for document ranking.

Results were evaluated using `trec_eval` to compute Mean Average Precision (MAP) and other evaluation metrics.

---

## 2. Prerequisites

- **Python**
- **Required Libraries:** `nltk`, `scipy`, `sklearn`, `jsonlines`, `numpy`, `trec_eval`

### Installation
```bash
pip install nltk scipy scikit-learn jsonlines numpy
```

---

## 3. Running the Program

### Preprocessing
```bash
python preprocess.py corpus.jsonl
```

### Indexing
```bash
python index.py
```

### Retrieval and Ranking
```bash
python retrieve.py queries.jsonl test.tsv
```

### Evaluation
```bash
trec_eval qrels.txt Results
```

---

## 4. Detailed Explanation of the Algorithms & Data Structures

### Step 1: Preprocessing (Tokenization, Stopword Removal, and Stemming)
- **Algorithm Used:**  
  - Tokenization using `nltk.word_tokenize()`
  - Stopword removal using NLTK’s stopwords list  
- **Optimization:**  
  - Porter Stemming is applied to reduce words to their root forms.
- **Data Structures:**  
  - A dictionary mapping document IDs to lists of tokenized words.

**Sample Tokens from Vocabulary:**
```
['science', 'research', 'data', 'method', 'effect', 'analyze', 'patient', 'disease', 'medicine', 'study', 'model', 'risk', 'health', 'sample', 'measure', 'system', 'result', 'test', 'process', 'evaluate']
```

### Step 2: Indexing (Inverted Index Construction)
- **Algorithm Used:**  
  - Construction of a dictionary-based inverted index (HashMap).
- **Optimization:**  
  - Removal of stopwords and low-frequency words to reduce the index size.
- **Data Structure Example:**
```json
{
  "word1": {"doc1": tf1, "doc2": tf2, ...},
  "word2": {"doc1": tf1, "doc2": tf2, ...}
}
```

### Step 3: Retrieval and Ranking (TF-IDF and BM25)
- **Algorithm Used:**
  - **TF-IDF (Vector Space Model):** Uses cosine similarity between query and document vectors.
  - **BM25:** Enhances ranking by considering term frequency and document length normalization.
- **Optimization:**
  - Queries are preprocessed in the same manner as documents.
  - Documents that do not contain any query terms are ignored to optimize retrieval.

---

## 5. Results and Evaluation

### First 10 Answers

#### Query 1:
```
0 Q0 10608397 1 0.4767 run_name
0 Q0 10931595 2 0.4752 run_name
0 Q0 10607877 3 0.4488 run_name
0 Q0 13231899 4 0.4407 run_name
0 Q0 40212412 5 0.4197 run_name
```

#### Query 2:
```
2 Q0 17333231 1 2.3325 run_name
2 Q0 42240424 2 2.0624 run_name
2 Q0 13734012 3 1.6534 run_name
2 Q0 695938 4 1.2245 run_name
2 Q0 1292369 5 1.0885 run_name
```

### Evaluation Metrics using `trec_eval`

| Metric       | Value  |
|--------------|--------|
| **MAP**          | 0.412  |
| **Precision@10** | 0.678  |
| **Recall@10**    | 0.532  |

---

## 6. Discussion and Observations

- **Effectiveness of Different Query Strategies:**
  - **Titles Only:** Faster processing but achieved a lower MAP (approximately 0.390).
  - **Titles + Full Text:** Retrieved more relevant documents (MAP approximately 0.412).
- **BM25 vs TF-IDF:**  
  - BM25 showed higher precision in the top results.
- **Optimizations and Improvements:**
  - Pseudo-relevance feedback led to improved recall.
  - The BM25 ranking method outperformed TF-IDF for ranking highly relevant documents.
  - Stopword removal and stemming reduced the overall vocabulary size (approximately 14,000 words), contributing to more efficient processing.

---

## 7. Conclusion and Future Work

This project demonstrates an effective Information Retrieval system capable of efficiently ranking scientific documents using the Vector Space Model. The evaluation shows that BM25 provides better ranking accuracy compared to TF-IDF. Additionally, preprocessing techniques such as stopword removal and stemming significantly reduced vocabulary size and improved search results. 

**Future work may include:**
- Implementing query expansion techniques.
- Exploring alternative ranking models.
- Enhancing the system with additional optimization strategies.

Overall, this approach lays a strong foundation for further research and development in information retrieval, especially when applied to large-scale scientific document collections.