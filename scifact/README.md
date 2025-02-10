# Information Retrieval System

## Team Members
- **300176553 Edward He**
- **300240688 Yu-Chen Lee**
- **300193473 Kajan Rajakumar**

## Division of Tasks
- **Preprocessing and Indexing:** Yu-Chen Lee 
- **Retrieval and Ranking:** Edward He  
- **Evaluation and Report Writing:** Kajan Rajakumar

---

## 1. Overview of the Implementation

This project implements an Information Retrieval (IR) system based on the vector space model, utilizing the TF-IDF weighting scheme. The dataset used is the SciFact dataset from the BEIR collection. We process a predefined set of test queries (odd-numbered queries only) for evaluation.

The system performs three main steps:
1. **Preprocessing:** Tokenization, stopword removal, and normalization.
2. **Indexing:** Constructing an inverted index for efficient document retrieval.
3. **Retrieval & Ranking:** Utilizing cosine similarity with TF-IDF weighting to rank documents.

The results were evaluated using `trec_eval` to compute Mean Average Precision (MAP) and other retrieval metrics. Two separate experiments were conducted: one using only the document titles and another using both the titles and the full text.

---

## 2. Prerequisites

### System Requirements
- **Python** (Version 3.7 or higher recommended)
- **Required Libraries:**
  ```bash
  pip install nltk scipy scikit-learn jsonlines numpy
  ```
- **Additional Tools:**
  - `trec_eval` (version 9.0.7) for evaluation

---

## 3. Running the Program

1. **Place** `csi4107_a1.py`, `corpus.jsonl`, and `queries.jsonl` in the same directory.
2. **Install** prerequisites (see above).
3. **Run**:
   ```bash
   python csi4107_a1.py
   ```
   This will:
   - Build the inverted index.
   - Print a random sample of 100 tokens from the vocabulary.
   - Calculate IDF values.
   - Process queries (only odd-numbered queries) and rank the top documents.
   - Write ranked results to `results_title.txt` (for title-only) and `results_all.txt` (for title + full text).

4. **Evaluate** with `trec_eval`:
   ```bash
   ./trec_eval-9.0.7/trec_eval [qrels_file] results_title.txt
   ./trec_eval-9.0.7/trec_eval [qrels_file] results_all.txt
   ```
   For example:
   ```bash
   ./trec_eval-9.0.7/trec_eval ./qrels/formated-test.tsv ./results_all.txt
   ./trec_eval-9.0.7/trec_eval ./qrels/formated-test.tsv ./results_title.txt
   ```

---

## 4. Detailed Explanation of the Algorithms & Data Structures

### Step 1: Preprocessing

#### Algorithm
- **Tokenization:** Uses regex-based tokenization to split text into words.
- **Stopword Removal:** Filters out common words using sklearn’s `ENGLISH_STOP_WORDS`.
- **Normalization:** Converts all tokens to lowercase.

#### Data Structures
- A list of tokens is generated for each document, which is then used in the indexing process.

---

### Step 2: Indexing (Inverted Index)

#### Algorithm
1. **Reading:** Each JSON line in `corpus.jsonl` is processed.
2. **Preprocessing:** The title and text are combined (or just the title in the "title" mode).
3. **Term Frequency Calculation:** Computes term frequencies (TF) for each document.
4. **Inverted Index Construction:** Stores entries in the form `{token: [(doc_id, tf), ...]}`.
5. **Document Length:** Stores the total token count for normalization during ranking.

#### Example
```json
{
  "study": [("doc001", 4), ("doc002", 1)],
  "cancer": [("doc003", 2), ("doc050", 3)]
}
```

---

### Step 3: Retrieval and Ranking (TF-IDF & Cosine Similarity)

#### Algorithm
1. **IDF Calculation:** For each term, IDF is computed as:
   \[
   \text{idf}[term] = \ln\left(\frac{N}{\text{df}[term]}\right)
   \]
   where \( N \) is the total number of documents.
2. **Query Processing:** Queries are preprocessed in the same manner as the corpus.
3. **TF-IDF Vector Construction:**
   - **Query weight:** \( \text{query\_tf} \times \text{idf} \)
   - **Document weight:** \( \text{tf} \times \text{idf} \)
4. **Scoring:** For each query, scores are computed by summing the products of query and document weights over all matching terms.
5. **Normalization:** Document scores are normalized by document length to approximate cosine similarity.
6. **Ranking:** Documents are sorted in descending order of their scores.

---

## 5. Sample of 100 Vocabulary Tokens

Below is a sample of 100 vocabulary tokens printed by the system during the inverted index building process for both **Title-Only** and **Title+Full Text** modes:

```bash
========== SAMPLE OF 100 VOCABULARY TOKENS (title mode) ==========
1. pd
2. cream
3. 34
4. dnmt1
5. cocaine
6. orange
7. shift
8. celiac
9. amazonensis
10. poland
11. sign
12. s
13. elucidation
14. colorectal
15. forks
16. remodeller
17. collaborates
18. succession
19. benzopyran
20. env
21. minimally
22. epidural
23. cause
24. biphasically
25. jack
26. dup
27. tangles
28. composition
29. decreased
30. repetition
31. homeobox
32. determinant
33. spermatogenic
34. exposure
35. main
36. 1996
37. heparan
38. supplemented
39. castration
40. sediments
41. transporters
42. infections
43. survivin
44. neuroinflammatory
45. lac
46. cortisol
47. alts
48. environmental
49. microflora
50. administering
51. keeping
52. ability
53. 1α
54. collegiate
55. promoter
56. nonuniform
57. preventing
58. entity
59. trophoblastic
60. formylated
61. module
62. recessive
63. fc
64. cytidine
65. infecting
66. myc
67. starches
68. experiments
69. drug
70. 1c
71. wolcott
72. phenix
73. findings
74. regulator
75. commensal
76. plus
77. wuschel
78. ikkα
79. fast
80. sphingosine
81. j4
82. msin3a
83. benzodiazepines
84. according
85. brucei
86. mycn
87. rhythmic
88. coxsackieviral
89. triglyceride
90. probe
91. pecking
92. nemo
93. wip1
94. glun2b
95. pot1b
96. rnap
97. hog1
98. pericyte
99. globulin
100. p190
====================================================


========== SAMPLE OF 100 VOCABULARY TOKENS (all mode) ==========
1. rs1982073
2. ihepscs
3. balanced
4. educational
5. human
6. 2
7. groove
8. 1920s
9. pristine
10. decades
11. escort
12. dia
13. anti
14. whale
15. hct116
16. agouti
17. oesophageal
18. reliably
19. brake
20. races
21. occur
22. holoendemic
23. 1993a
24. equivalents
25. dexamethasone
26. 32s
27. surfactin
28. transit
29. variably
30. contributor
31. surprised
32. massively
33. maximal
34. coexpression
35. contamination
36. ha2
37. sort1
38. cxc
39. carbinol
40. logically
41. foxk
42. frontline
43. oxidizing
44. rembrandt
45. sensitizer
46. ki67
47. absolute
48. balb
49. sbts
50. insertion
51. 3985
52. bridge
53. menarche
54. 220
55. ovulation
56. 2f5
57. migrate
58. ilds
59. metabololipidomics
60. reg
61. respectively
62. silently
63. collective
64. acgh
65. rad51
66. able
67. cigr1
68. aist
69. martinique
70. secdfyajc
71. all1
72. sulfonate
73. biloba
74. neurosurgeons
75. beset
76. 092
77. generalizable
78. phagolysosomal
79. cytosine
80. specifies
81. cytostatic
82. hsp
83. dihydrokainate
84. prostanoid
85. thrombus
86. parasitemia
87. infusion
88. muckle
89. wrongly
90. microcin
91. pm
92. rois
93. 13
94. d13s263
95. 688
96. polychemotherapy
97. cns
98. disability
99. dusp1
100. 1533
====================================================
```

---

## 6. Results & Evaluation

### Top 10 Results for Two Sample Odd Queries

#### Query 9 (Title-Only Mode)
```
9 Q0 44265107 1 13.1443 run_name
9 Q0 24700152 2 12.4589 run_name
9 Q0 9056874 3 7.3151 run_name
9 Q0 45461275 4 6.9956 run_name
9 Q0 16322674 5 6.6506 run_name
9 Q0 13380980 6 5.6857 run_name
9 Q0 14647747 7 5.2577 run_name
9 Q0 3419709 8 5.2570 run_name
9 Q0 25853741 9 4.9724 run_name
9 Q0 33684572 10 4.9724 run_name
```

#### Query 11 (Title-Only Mode)
```
11 Q0 25510546 1 23.2622 run_name
11 Q0 8453819 2 17.5327 run_name
11 Q0 38477436 3 17.5327 run_name
11 Q0 19561411 4 16.5684 run_name
11 Q0 29459383 5 14.8354 run_name
11 Q0 26121646 6 7.9523 run_name
11 Q0 16712164 7 7.2294 run_name
11 Q0 41781905 8 6.6270 run_name
11 Q0 1710116 9 6.2915 run_name
11 Q0 17656445 10 6.2915 run_name
```

#### Query 9 (Title+Full Text Mode)
```
9 Q0 44265107 1 3.3429 run_name
9 Q0 24700152 2 2.9995 run_name
9 Q0 32787042 3 1.0355 run_name
9 Q0 19511011 4 1.0326 run_name
9 Q0 27188320 5 0.9184 run_name
9 Q0 11527199 6 0.8871 run_name
9 Q0 13030852 7 0.8293 run_name
9 Q0 11255504 8 0.7793 run_name
9 Q0 17119869 9 0.7566 run_name
9 Q0 25182647 10 0.7537 run_name
```

#### Query 11 (Title+Full Text Mode)
```
11 Q0 29459383 1 2.4232 run_name
11 Q0 13780287 2 2.2968 run_name
11 Q0 4399311 3 2.0883 run_name
11 Q0 1887056 4 1.7418 run_name
11 Q0 25510546 5 1.7206 run_name
11 Q0 19708993 6 1.6983 run_name
11 Q0 16712164 7 1.5732 run_name
11 Q0 3591070 8 1.5053 run_name
11 Q0 20904154 9 1.4901 run_name
11 Q0 32587939 10 1.4409 run_name
```

### TREC-Eval Metrics

After running the evaluation with:
```bash
./trec_eval-9.0.7/trec_eval ./qrels/formated-test.tsv ./results_all.txt
```

the following metrics were obtained:

#### **For Title+Full Text (results_all.txt)**
| Metric               | Value   |
|----------------------|---------|
| **MAP**              | 0.4563  |
| **GM_MAP**           | 0.1366  |
| **Rprec**            | 0.3280  |
| **bpref**            | 0.9423  |
| **Recip_Rank**       | 0.4702  |
| **P@5**              | 0.1386  |
| **P@10**             | 0.0856  |
| **P@15**             | 0.0610  |
| **P@20**             | 0.0461  |
| **P@30**             | 0.0318  |
| **P@100**            | 0.0105  |
| **P@200**            | 0.0054  |
| **P@500**            | 0.0022  |
| **P@1000**           | 0.0011  |

#### **For Title-Only Mode (results_title.txt)**
| Metric               | Value   |
|----------------------|---------|
| **MAP**              | 0.3132  |
| **GM_MAP**           | 0.0216  |
| **Rprec**            | 0.2221  |
| **bpref**            | 0.7730  |
| **Recip_Rank**       | 0.3225  |
| **P@5**              | 0.0889  |
| **P@10**             | 0.0556  |
| **P@15**             | 0.0436  |
| **P@20**             | 0.0353  |
| **P@30**             | 0.0255  |
| **P@100**            | 0.0088  |
| **P@200**            | 0.0047  |
| **P@500**            | 0.0019  |
| **P@1000**           | 0.0010  |

**Interpretation:**
- **MAP** for **Title+Full Text** mode is 0.4563, which is higher than the MAP for **Title-Only** mode (0.3132), indicating that incorporating the full text results in better performance.
- The **bpref** is higher in **Title+Full Text** (0.9423) compared to **Title-Only** (0.7730), indicating a better ranking of relevant documents.

---

## 7. Discussion & Observations

- **Query Filtering:** The system correctly processes only the odd-numbered queries (test queries), as specified.
- **Vocabulary Diversity:** Both modes reflect a diverse vocabulary, but the **Title+Full Text** mode has a broader range of tokens due to the inclusion of the full text.
- **Performance Analysis:**  
  - **Title+Full Text** provides a higher MAP and better ranking quality (bpref) compared to **Title-Only**, indicating that using the full text improves retrieval performance.
  - **Title-Only** mode achieves lower MAP, but it still provides useful information, though with fewer relevant results in the top ranks.

---

## 8. Conclusion & Future Work

We successfully developed an IR system using the vector space model with TF-IDF weighting and cosine similarity for document ranking. After running experiments with both **Title-Only** and **Title+Full Text** modes, we concluded that incorporating the full text provides better retrieval performance. Future work includes:
- **Enhancing preprocessing** (e.g., stemming or lemmatization).
- **Experimenting with advanced ranking models** like BM25.
- **Exploring query expansion and pseudo-relevance feedback** to improve precision and recall.

This work lays a strong foundation for further improvements and explorations in Information Retrieval systems.
