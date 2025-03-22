# Neural Information Retrieval System

## Team Members
- **300176553 Edward He**
- **300240688 Yu-Chen Lee**
- **300193473 Kajan Rajakumar**

## Division of Tasks
- **Neural Architecture & Implementation:** Edward He 
- **Evaluation & Results Analysis:** Yu-Chen Lee  
- **Optimization & Report Writing:** Kajan Rajakumar

---

## 1. Overview of the Implementation

This project enhances our Assignment 1 Information Retrieval (IR) system with advanced neural methods. We implement a two-stage retrieval architecture that combines traditional TF-IDF with neural re-ranking approaches to achieve significantly improved performance on the SciFact dataset.

Our system performs the following steps:
1. **Initial Retrieval:** Uses the TF-IDF Vector Space Model from Assignment 1 to obtain candidate documents.
2. **Neural Re-ranking:** Applies transformer-based neural models to re-score documents based on semantic relevance to the query.

We implemented two distinct neural re-ranking approaches:
1. **SentenceBERT Re-ranker:** Uses a bi-encoder architecture to generate embeddings for queries and documents separately.
2. **BERT Cross-Encoder Re-ranker:** Directly scores query-document pairs with a fine-tuned BERT model.

All components were optimized for CPU execution, with careful memory management to accommodate non-GPU environments.

---

## 2. Prerequisites

### System Requirements
- **Python 3.7+**
- **Required Libraries with Version Dependencies:**
  ```bash
  pip install numpy==1.24.3 torch==2.0.1 transformers==4.29.2 sentence-transformers==2.2.2 scikit-learn==1.6.1 pandas==2.2.3 matplotlib==3.9.4 tqdm==4.67.1 jsonlines==4.0.0 nltk==3.9.1 datasets==3.4.1
  ```
  
  Key dependencies include:
  - torch==2.0.1
  - transformers==4.29.2
  - sentence-transformers==2.2.2
  - numpy==1.24.3
  - scikit-learn==1.6.1

- **Evaluation Tools:**
  - `trec_eval` (version 9.0.7) for computing effectiveness metrics

### Hardware Considerations
Our implementation incorporates several optimizations for CPU environments:
- Use of smaller model variants (e.g., MiniLM-based models)
- Batch processing with memory cleanup
- Reduced sequence lengths
- Chunked processing for larger document collections

---

## 3. Running the Program

### Basic Usage
1. **Ensure** all files (`neural_ir_system.py`, `evaluate_ir_systems.py`, `finetune_bert.py`) are in the same directory.
2. **Place** `corpus.jsonl` and `queries.jsonl` in the same directory.

You can run each module separately:
- **TF-IDF Baseline or Neural Re-ranking:**  
  For example, to run with the SentenceBERT re-ranker:
  ```bash
  python neural_ir_system.py --corpus corpus.jsonl --queries queries.jsonl --reranker sbert --output results_sbert.txt
  ```
  Replace `sbert` with `bert` for the BERT Cross-Encoder, or use `none` for the TF-IDF baseline.

- **Evaluation Only:**  
  ```bash
  python evaluate_ir_systems.py --qrels path/to/qrels_file
  ```

### Convenient One-Command Evaluation
For convenience, the entire pipeline (initial retrieval, neural re-ranking, and evaluation) can be executed with a single command:
```bash
python3 evaluate_ir_systems.py --qrels ../scifact/qrels/formated-test.tsv --corpus ../scifact/corpus.jsonl --queries ../scifact/queries.jsonl --run
```
This command will:
- Execute the full retrieval and re-ranking pipeline.
- Compute evaluation metrics using `trec_eval`.
- Generate comparison tables, an evaluation plot (saved as `evaluation_results.png`), and extract top results for sample queries.

### Advanced Usage
To fine-tune a BERT model on your domain data (if needed), run:
```bash
python finetune_bert.py --corpus corpus.jsonl --queries queries.jsonl --qrels path/to/qrels_file
```

### Command-Line Options
- `--reranker`: Specify the re-ranking method (`none`, `sbert`, or `bert`).
- `--initial_k`: Number of documents to retrieve in the initial phase (default: 100).
- `--top_k`: Number of final results to return (default: 1000).
- `--mode`: Content mode (`title` or `all`).
- `--batch_size`: Batch size for neural processing (default: 4).
- `--use_small_models`: Use smallest models possible for CPU efficiency.
- `--output`: Path for the output results file.

---

## 4. Detailed Explanation of Algorithms & Data Structures

### A. Base TF-IDF System
We retained the TF-IDF system from Assignment 1 for initial retrieval with the following optimizations:
1. **Preprocessing:**  
   Tokenization, stopword removal, and normalization.
2. **Inverted Index:**  
   Maps terms to lists of (document_id, term_frequency) pairs.
3. **IDF Calculation:**  
   Computes the inverse document frequency for each term.
4. **Ranking:**  
   Uses cosine similarity between query and document vectors with TF-IDF weighting.

This stage produces an initial set of candidate documents, which are then re-ranked by the neural modules.

### B. SentenceBERT Re-ranking Approach
1. **Model Architecture:**  
   Utilizes a pre-trained transformer model (e.g., MiniLM-L3-v2) optimized for semantic similarity to generate fixed-size embeddings for both queries and documents.
2. **Process Flow:**  
   ```
   Query → Encoder → Query Embedding
                           ↓
                         Cosine Similarity
                           ↑
   Document → Encoder → Document Embedding
   ```
3. **Optimizations:**  
   - Chunk processing: Documents are processed in manageable chunks.
   - Batch encoding with memory cleanup.
   - Use of smaller model variants for CPU efficiency.

### C. BERT Cross-Encoder Approach
1. **Model Architecture:**  
   Uses a fine-tuned BERT model (e.g., ms-marco-MiniLM-L-2-v2) that directly scores concatenated query-document pairs by extracting the [CLS] token representation.
2. **Process Flow:**  
   ```
   [Query, Document] → Cross-Encoder → Relevance Score
   ```
3. **Optimizations:**  
   - Reduced sequence length (e.g., 256 tokens).
   - Small batch sizes (default: 4) to control memory usage.
   - Chunked processing for large document sets.

---

## 5. Evaluation Results

### Overall Evaluation Metrics

The evaluation of the IR systems yielded the following results:

| Method               | MAP    | BPref  | MRR    | P@10  |
|----------------------|--------|--------|--------|-------|
| TF-IDF (Baseline)    | 0.4558 | 0.8560 | 0.4698 | 0.0856|
| SentenceBERT Re-ranker | 0.4659 | 0.8560 | 0.4781 | 0.0784|
| BERT Cross-Encoder   | 0.5800 | 0.8560 | 0.6011 | 0.0915|

**Best Scores:**
- **MAP:** 0.5800 (BERT Cross-Encoder)
- **BPref:** 0.8560 (TF-IDF Baseline, though BPref is equal across methods)
- **MRR:** 0.6011 (BERT Cross-Encoder)
- **P@10:** 0.0915 (BERT Cross-Encoder)

An evaluation plot has been saved as `evaluation_results.png`.

### Top 10 Results for Sample Queries

The following are the top 10 results extracted for Query 1 and Query 3 from each method:

#### TF-IDF (Baseline)
- **Query 1:**
  1. 10608397 (Score: 0.476652)
  2. 10931595 (Score: 0.475227)
  3. 10607877 (Score: 0.448791)
  4. 13231899 (Score: 0.440700)
  5. 40212412 (Score: 0.419653)
  6. 31543713 (Score: 0.406405)
  7. 17482507 (Score: 0.392065)
  8. 31942055 (Score: 0.387916)
  9. 25404036 (Score: 0.378377)
  10. 3845894 (Score: 0.376772)
  
- **Query 3:**
  1. 2739854 (Score: 3.407947)
  2. 23389795 (Score: 3.391449)
  3. 8411251 (Score: 2.209293)
  4. 14717500 (Score: 2.048099)
  5. 4632921 (Score: 1.468707)
  6. 1649738 (Score: 1.407458)
  7. 32181055 (Score: 1.204199)
  8. 18344910 (Score: 1.200663)
  9. 14019636 (Score: 1.191578)
  10. 39661951 (Score: 1.142906)

#### SentenceBERT Re-ranker
- **Query 1:**
  1. 16461149 (Score: 0.321961)
  2. 6863070 (Score: 0.317706)
  3. 15327601 (Score: 0.306179)
  4. 4435369 (Score: 0.283781)
  5. 9580772 (Score: 0.282772)
  6. 1156322 (Score: 0.271525)
  7. 7581911 (Score: 0.266809)
  8. 43385013 (Score: 0.252446)
  9. 20758340 (Score: 0.242952)
  10. 25404036 (Score: 0.228639)
  
- **Query 3:**
  1. 2739854 (Score: 0.676822)
  2. 14717500 (Score: 0.593115)
  3. 23389795 (Score: 0.574959)
  4. 1388704 (Score: 0.554290)
  5. 19058822 (Score: 0.533837)
  6. 32181055 (Score: 0.531528)
  7. 4632921 (Score: 0.512128)
  8. 10145528 (Score: 0.511564)
  9. 4378885 (Score: 0.507587)
  10. 10944947 (Score: 0.505424)

#### BERT Cross-Encoder
- **Query 1:**
  1. 25950264 (Score: -6.459208)
  2. 10931595 (Score: -7.412260)
  3. 40212412 (Score: -8.300652)
  4. 1346695 (Score: -8.459714)
  5. 4422734 (Score: -8.667697)
  6. 10608397 (Score: -9.113071)
  7. 10607877 (Score: -9.286323)
  8. 6863070 (Score: -9.304207)
  9. 27049238 (Score: -9.465171)
  10. 42240424 (Score: -9.640570)
  
- **Query 3:**
  1. 4632921 (Score: 1.620838)
  2. 4414547 (Score: 1.488196)
  3. 14717500 (Score: 0.374331)
  4. 23389795 (Score: -0.254737)
  5. 2739854 (Score: -0.257630)
  6. 20280410 (Score: -0.703396)
  7. 13519661 (Score: -1.610399)
  8. 1388704 (Score: -1.863923)
  9. 4378885 (Score: -2.635634)
  10. 14019636 (Score: -2.840515)

The final “Results” file was created using the best performing method (BERT Cross-Encoder).

---

## 6. Discussion & Analysis

### Performance Analysis

1. **BERT Cross-Encoder Superiority:**  
   The BERT Cross-Encoder consistently outperformed the TF-IDF baseline and SentenceBERT re-ranker across all metrics. With a MAP of 0.5800 and an MRR of 0.6011, it demonstrates the advantage of jointly modeling query–document interactions.

2. **Re-ranking Effectiveness:**  
   Although SentenceBERT improves over the TF-IDF baseline modestly, the BERT Cross-Encoder yields substantial gains—a 27% increase in MAP over the baseline.

3. **Efficiency vs. Effectiveness Trade-offs:**  
   - **SentenceBERT** is computationally more efficient and suitable for large-scale retrieval under resource constraints.  
   - **BERT Cross-Encoder** is more resource intensive but provides significantly better retrieval performance.

4. **Domain Adaptation:**  
   Both neural re-ranking methods exhibit strong transferability to scientific literature, even without extensive domain-specific fine-tuning.

### System Limitations

1. **Computational Requirements:**  
   Neural methods demand more computational resources than traditional TF-IDF, necessitating optimizations for CPU execution.
   
2. **Document Length Handling:**  
   Scientific documents often exceed model input limits, so truncation may lead to loss of critical information.
   
3. **Dependence on Initial Retrieval:**  
   The performance of the neural re-rankers depends on the quality of the initial TF-IDF retrieval. Relevant documents not retrieved initially cannot be re-ranked later.

---

## 7. Conclusion & Future Work

Our implementation successfully combines traditional IR techniques with state-of-the-art neural methods to significantly enhance retrieval performance for scientific literature. The BERT Cross-Encoder achieved the best performance with notable improvements in MAP, MRR, and P@10.

### Future Improvements

1. **Domain-Specific Fine-Tuning:**  
   Further fine-tuning on a larger corpus of scientific literature may further improve semantic matching.
   
2. **Document Chunking:**  
   Splitting long documents into smaller, semantically coherent chunks before encoding could capture additional relevant information.
   
3. **Query Expansion:**  
   Incorporating query expansion techniques may enhance the initial retrieval phase.
   
4. **Ensemble Methods:**  
   Combining scores from multiple ranking methods could further boost overall performance.
   
5. **Exploration of More Efficient Models:**  
   Investigating distilled or quantized models may provide faster inference while maintaining performance.

---

## Environment and Package Versions

Ensure that the following pip packages (with the specified versions) are installed:
- **torch:** 2.0.1
- **transformers:** 4.29.2
- **sentence-transformers:** 2.2.2
- **numpy:** 1.24.3
- **scikit-learn:** 1.6.1
- **pandas:** 2.2.3
- **matplotlib:** 3.9.4
- **tqdm:** 4.67.1
- **jsonlines:** 4.0.0
- **nltk:** 3.9.1
- **datasets:** 3.4.1

---

## Checklist for Assignment Completion

- [x] Implemented an improved IR system with advanced neural retrieval methods.
- [x] Integrated a TF-IDF baseline for initial candidate retrieval.
- [x] Developed and integrated both SentenceBERT and BERT Cross-Encoder re-ranking approaches.
- [x] Produced result files in the required format: `results_tfidf.txt`, `results_sbert.txt`, and `results_bert.txt`.
- [x] Evaluated all systems using `trec_eval` and reported metrics (MAP, BPref, MRR, P@10).
- [x] Documented system functionality, algorithms, and optimizations.
- [x] Provided sample outputs (top 10 results for Query 1 and Query 3 from each method).
- [x] Included comprehensive instructions on how to run the programs.
- [x] Specified team members and detailed task division.
- [x] Ensured all programs run correctly.
- [x] Packaged the assignment (code, README, and Results file) as a zip file for BrightSpace submission.
- [x] Excluded the initial text collection and any external tools from the submission.
- [x] Provided a convenient one-command evaluation method:
  ```bash
  python3 evaluate_ir_systems.py --qrels ../scifact/qrels/formated-test.tsv --corpus ../scifact/corpus.jsonl --queries ../scifact/queries.jsonl --run
  ```