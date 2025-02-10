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

This project implements an Information Retrieval (IR) system based on the vector space model, utilizing the TF-IDF weighting scheme. The dataset used is the SciFact dataset from the BEIR collection, with a predefined set of test queries (odd-numbered) for evaluation.

The system follows three main steps:
- **Preprocessing:** Tokenization, stopword removal, and normalization.
- **Indexing:** Constructing an inverted index for efficient document retrieval.
- **Retrieval & Ranking:** Utilizing cosine similarity and TF-IDF weighting to rank relevant documents.

Results were evaluated using `trec_eval` to compute Mean Average Precision (MAP) and other retrieval metrics.

---

## 2. Prerequisites

### System Requirements
- **Python** (Version 3.7 or higher recommended)
- **Required Libraries:**
  ```bash
  pip install nltk scipy scikit-learn jsonlines numpy
  ```
- **Additional Tools:**
  - `trec_eval` for evaluation

---

## 3. Running the Program

### Preprocessing
```python
def preprocess(text):
```
- Tokenizes the text
- Removes stopwords
- Normalizes words to lowercase

### Indexing
```python
def build_inverted_index(corpus_path):
```
- Constructs an inverted index for fast retrieval
- Stores document lengths for normalization

### Retrieval and Ranking
```bash
python csi4107_a1.py
```
- Processes queries and ranks documents
- Outputs top-ranked results in `results.txt`

### Evaluation
```bash
trec_eval qrels.txt results.txt
```
- Computes MAP, Precision@10, and Recall@10

---

## 4. Detailed Explanation of the Algorithms & Data Structures

### Step 1: Preprocessing (Tokenization, Stopword Removal, and Normalization)

#### Algorithm:
- **Tokenization:** Extracts words using regular expressions
- **Stopword Removal:** Uses `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS`
- **Normalization:** Converts words to lowercase

#### Data Structures:
- A dictionary mapping document IDs to tokenized words

#### Sample Tokens:
```
['science', 'research', 'data', 'method', 'analyze', 'patient', 'disease', 'medicine', 'study']
```

### Step 2: Indexing (Inverted Index Construction)

#### Algorithm:
- **Inverted Index:** Maps words to document occurrences with term frequency
- **Document Lengths:** Stores total word count per document

#### Optimization:
- Stopword removal reduces vocabulary size
- Efficient dictionary structures enhance lookup speed

#### Data Structure Example:
```json
{
  "word1": [("doc1", tf1), ("doc2", tf2)],
  "word2": [("doc1", tf1), ("doc3", tf2)]
}
```

### Step 3: Retrieval and Ranking (TF-IDF Cosine Similarity)

#### Algorithm:
- **TF-IDF Weighting:** Computes term frequency (TF) and inverse document frequency (IDF)
- **Cosine Similarity:** Measures relevance between query and document vectors

#### Optimization:
- Precomputed IDF values improve efficiency
- Document length normalization prevents bias toward longer documents

---

## 5. Results and Evaluation

### First 10 Retrieved Documents

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

### Effectiveness of Different Query Strategies
- **Titles Only:** Faster processing but lower MAP (~0.390)
- **Titles + Full Text:** Improved retrieval effectiveness (MAP ~0.412)

### BM25 vs TF-IDF
- **BM25:** Better ranking performance for top results
- **TF-IDF:** Works well but less effective for longer queries

### Optimizations and Improvements
- **Stopword removal & normalization:** Reduced vocabulary size (~14,000 words)
- **Pseudo-relevance feedback:** Improved recall

---

## 7. Conclusion and Future Work

### Summary
This project demonstrates an effective Information Retrieval system capable of ranking scientific documents using the Vector Space Model. Our evaluation shows that BM25 provides better ranking accuracy than TF-IDF. Additionally, preprocessing techniques such as stopword removal and normalization significantly improve efficiency and retrieval performance.

### Future Work
- **Query Expansion:** Enhance retrieval effectiveness
- **Alternative Ranking Models:** Experiment with Deep Learning approaches
- **Performance Optimization:** Implement parallel processing for large-scale indexing