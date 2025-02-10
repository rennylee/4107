"""
CSI4107 Assignment 1 - Vector Space Model IR System

This script implements a simple Information Retrieval system that:
  1. Preprocesses the corpus (tokenizes, removes stopwords, normalizes).
  2. Builds an inverted index to store term frequencies per document.
  3. Calculates IDF for each term in the vocabulary.
  4. Processes queries and computes TF-IDF based cosine similarity scores.
  5. Outputs ranked results (top-1000 by default) in TREC format.
  6. Prints a random sample of 100 tokens from the vocabulary (for documentation).

The script runs two experiments:
  - Title-Only: Uses only the document titles.
  - Title+Full Text: Uses both the title and the full text.

File Requirements:
  - corpus.jsonl (the SciFact corpus)
  - queries.jsonl (list of queries)
  - The results files are generated automatically by the script.

Usage:
  python csi4107_a1.py
"""

import jsonlines
import os
import re
import math
from collections import defaultdict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import random

def preprocess(text):
    """
    Step 1: Preprocessing
    - Tokenizes the text to extract words.
    - Removes standard English stopwords.
    - Converts everything to lowercase for uniformity.

    :param text: Raw string of text.
    :return: List of filtered tokens.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    filtered_tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
    return filtered_tokens


def build_inverted_index(corpus_path, mode='all'):
    """
    Step 2: Build the Inverted Index
    - Reads the corpus (JSONL) line by line.
    - For each document, preprocesses the text to extract tokens.
    - If mode is 'title', only the title is used; if mode is 'all', title and text are combined.
    - Calculates term frequencies (TF) for each token.
    - Updates the inverted index with (doc_id, count).
    - Stores document lengths for normalization.

    :param corpus_path: Path to the corpus.jsonl file.
    :param mode: 'title' for Title-Only, 'all' for Title+Full Text.
    :return: (inverted_index, document_lengths)
    """
    inverted_index = defaultdict(list)
    document_lengths = {}

    with jsonlines.open(corpus_path) as reader:
        for doc in reader:
            doc_id = doc["_id"]
            title = doc.get("title", "")
            text = doc.get("text", "")

            # Choose content based on mode
            if mode == 'title':
                content = title
            else:
                content = title + " " + text

            tokens = preprocess(content)

            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1

            for token, count in tf.items():
                inverted_index[token].append((doc_id, count))

            document_lengths[doc_id] = len(tokens)

    return inverted_index, document_lengths


def calculate_idf(inverted_index, total_docs):
    """
    Step 3: Calculate IDF Values
    - IDF(term) = ln(total_docs / doc_freq(term))

    :param inverted_index: dict {token -> [(doc_id, tf), ...]}
    :param total_docs: Total number of documents in the corpus.
    :return: dict {term -> idf_value}
    """
    idf = {}
    for term, postings in inverted_index.items():
        idf[term] = math.log(total_docs / len(postings))
    return idf


def rank_documents(query, inverted_index, idf, document_lengths):
    """
    Step 4: Query Processing and Ranking
    - Preprocesses the query.
    - Computes a TF-IDF vector for the query.
    - For each term in the query, iterates over all documents that contain it,
      accumulating a partial score (query_weight * document_weight).
    - Normalizes the document scores by document length (approximating cosine similarity).
    - Returns a sorted list of (doc_id, score) in descending order of score.

    :param query: Query text.
    :param inverted_index: Inverted index dictionary.
    :param idf: IDF values for each term.
    :param document_lengths: Dictionary mapping doc_id to token count.
    :return: List of (doc_id, score), sorted descending by score.
    """
    query_tokens = preprocess(query)
    query_tf = defaultdict(int)
    for token in query_tokens:
        query_tf[token] += 1

    query_vector = {}
    for token, count in query_tf.items():
        if token in idf:
            query_vector[token] = count * idf[token]

    scores = defaultdict(float)
    for token, weight in query_vector.items():
        if token in inverted_index:
            for doc_id, tf in inverted_index[token]:
                scores[doc_id] += weight * (tf * idf[token])

    for doc_id in scores:
        scores[doc_id] /= max(document_lengths[doc_id], 1e-6)

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs


def main():
    """
    Main Script:
    1. Locates corpus.jsonl and queries.jsonl.
    2. Runs experiments for both modes: Title-Only and Title+Full Text.
    3. For each mode, builds the inverted index, calculates IDF, and processes queries.
    4. Writes results in TREC format to separate output files.
    """
    # File paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(current_dir, "corpus.jsonl")
    queries_path = os.path.join(current_dir, "queries.jsonl")

    modes = ["title", "all"]

    for mode in modes:
        print(f"\n================ Running experiment for mode: {mode} ================\n")

        # Build the inverted index based on the selected mode
        inverted_index, document_lengths = build_inverted_index(corpus_path, mode)
        total_docs = len(document_lengths)

        vocabulary = list(inverted_index.keys())
        sample_size = min(100, len(vocabulary))
        sample_tokens = random.sample(vocabulary, sample_size)
        print(f"\n========== SAMPLE OF 100 VOCABULARY TOKENS ({mode} mode) ==========")
        for i, token in enumerate(sample_tokens, start=1):
            print(f"{i}. {token}")
        print("====================================================\n")

        print("Calculating IDF values...")
        idf = calculate_idf(inverted_index, total_docs)

        print("Processing queries...")
        with jsonlines.open(queries_path) as reader:
            queries = list(reader)

        # Define the output result file name based on mode
        results_file_name = f"results_{mode}.txt"
        results_path = os.path.join(current_dir, results_file_name)

        # Open the result file for writing, this will overwrite existing files
        with open(results_path, "w") as results_file:
            for query in queries:
                query_id = query["_id"]

                try:
                    if int(query_id) % 2 == 0:
                        continue  # Skip even queries
                except ValueError:
                    continue

                query_text = query.get("text", "")
                if not query_text:
                    continue

                ranked_docs = rank_documents(query_text, inverted_index, idf, document_lengths)

                # Write the top-1000 results in TREC format
                for rank, (doc_id, score) in enumerate(ranked_docs[:1000], start=1):
                    results_file.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} run_name\n")

        print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
