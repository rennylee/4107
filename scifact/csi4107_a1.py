import jsonlines
import os
import re
import math
from collections import defaultdict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Step 1: Preprocessing
def preprocess(text):
    """
    Tokenizes, removes stopwords, and normalizes text.
    """
    # Tokenization
    tokens = re.findall(r"\b\w+\b", text.lower())

    # Remove stopwords
    filtered_tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]

    return filtered_tokens


# Step 2: Build the inverted index
def build_inverted_index(corpus_path):
    """
    Builds an inverted index from the corpus.
    """
    inverted_index = defaultdict(list)
    document_lengths = {}  # To store the length of each document (for normalization)

    with jsonlines.open(corpus_path) as reader:
        for doc in reader:
            doc_id = doc["_id"]
            title = doc.get("title", "")
            text = doc.get("text", "")

            # Combine title and text
            content = title + " " + text
            tokens = preprocess(content)

            # Calculate term frequency (tf) for this document
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1

            # Update the inverted index
            for token, count in tf.items():
                inverted_index[token].append((doc_id, count))

            # Store document length
            document_lengths[doc_id] = len(tokens)

    return inverted_index, document_lengths


# Step 3: Calculate IDF values
def calculate_idf(inverted_index, total_docs):
    """
    Calculates inverse document frequency (IDF) for each term.
    """
    idf = {}
    for term, postings in inverted_index.items():
        idf[term] = math.log(total_docs / len(postings))
    return idf


# Step 4: Query processing and ranking
def rank_documents(query, inverted_index, idf, document_lengths):
    """
    Ranks documents based on cosine similarity with the query.
    """
    # Preprocess the query
    query_tokens = preprocess(query)

    # Calculate query term frequency (tf)
    query_tf = defaultdict(int)
    for token in query_tokens:
        query_tf[token] += 1

    # Compute query weights (tf-idf)
    query_vector = {}
    for token, count in query_tf.items():
        if token in idf:
            query_vector[token] = count * idf[token]

    # Calculate scores
    scores = defaultdict(float)
    for token, weight in query_vector.items():
        if token in inverted_index:
            for doc_id, tf in inverted_index[token]:
                scores[doc_id] += weight * (tf * idf[token])

    # Normalize scores by document length
    for doc_id in scores:
        scores[doc_id] /= max(document_lengths[doc_id], 1e-6)

    # Sort documents by score in descending order
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs


# Main script
def main():
    # File paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(current_dir, "corpus.jsonl")
    queries_path = os.path.join(current_dir, "queries.jsonl")
    results_path = os.path.join(current_dir, "results.txt")

    # Build inverted index
    print("Building inverted index...")
    inverted_index, document_lengths = build_inverted_index(corpus_path)
    total_docs = len(document_lengths)

    # Calculate IDF values
    print("Calculating IDF values...")
    idf = calculate_idf(inverted_index, total_docs)

    # Process queries and rank documents
    print("Processing queries...")
    with jsonlines.open(queries_path) as reader:
        queries = list(reader)

    with open(results_path, "w") as results_file:
        for query in queries:
            query_id = query["_id"]
            query_text = query.get("text", "")

            # Skip if query_text is empty
            if not query_text:
                continue

            ranked_docs = rank_documents(query_text, inverted_index, idf, document_lengths)

            # Write top-1000 results to the results file
            for rank, (doc_id, score) in enumerate(ranked_docs[:1000], start=1):
                results_file.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} run_name\n")

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
