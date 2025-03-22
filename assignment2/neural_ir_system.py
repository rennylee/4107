"""
CSI4107 Assignment 2 - Neural Information Retrieval System (CPU-Optimized)

This script implements an enhanced Information Retrieval system optimized for CPU usage:
  1. Uses the TF-IDF system from Assignment 1 for initial retrieval
  2. Applies neural re-ranking methods to improve results
  3. Outputs ranked results in TREC format for evaluation

Usage:
  python neural_ir_system.py
"""

import os
import json
import numpy as np
import jsonlines
from tqdm import tqdm
from collections import defaultdict
import re
import math
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import random
import argparse
import gc
import time

from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
NEURAL_AVAILABLE = True

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device('cpu')
print(f"Using device: {device}")

#------------------------------------------------------------
# TF-IDF System from Assignment 1 (with some optimizations)
#------------------------------------------------------------

def preprocess(text):
    """
    Preprocessing: tokenize, remove stopwords, convert to lowercase
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    filtered_tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
    return filtered_tokens


def build_inverted_index(corpus_path, mode='all'):
    """
    Build the inverted index from corpus, storing term frequencies
    """
    inverted_index = defaultdict(list)
    document_lengths = {}
    document_texts = {}  # Store document texts for re-ranking phase

    with jsonlines.open(corpus_path) as reader:
        for doc in reader:
            doc_id = doc["_id"]
            title = doc.get("title", "")
            text = doc.get("text", "")

            # Choose content based on mode
            if mode == 'title':
                content = title
                # Still store full content for re-ranking
                document_texts[doc_id] = title
            else:
                content = title + " " + text
                document_texts[doc_id] = title + " " + text

            tokens = preprocess(content)

            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1

            for token, count in tf.items():
                inverted_index[token].append((doc_id, count))

            document_lengths[doc_id] = len(tokens)

    return inverted_index, document_lengths, document_texts


def calculate_idf(inverted_index, total_docs):
    """
    Calculate IDF values for all terms in the vocabulary
    """
    idf = {}
    for term, postings in inverted_index.items():
        idf[term] = math.log(total_docs / len(postings))
    return idf


def rank_documents(query, inverted_index, idf, document_lengths):
    """
    Rank documents using TF-IDF with cosine similarity
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

    # Normalize scores by document length (approximates cosine similarity)
    for doc_id in scores:
        scores[doc_id] /= max(document_lengths[doc_id], 1e-6)

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs


#------------------------------------------------------------
# Neural Re-ranking: SentenceBERT Approach (CPU-Optimized)
#------------------------------------------------------------

class SentenceBERTReranker:
    """
    Re-ranks documents using Sentence Transformers for semantic similarity
    Optimized for CPU usage
    """
    def __init__(self, model_name=None, use_small_model=False):
        # Use smaller model by default for CPU
        if use_small_model or model_name is None:
            model_name = "paraphrase-MiniLM-L3-v2"  # Very small, CPU-friendly model
        else:
            model_name = "all-MiniLM-L6-v2"  # Default model

        print(f"Loading SentenceBERT model: {model_name}")
        start_time = time.time()
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    def encode_batch(self, texts, batch_size=8):  # Reduced batch size for CPU
        """Encode a large list of texts in batches with memory cleanup"""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding documents"):
            batch = texts[i:i+batch_size]
            with torch.no_grad():  # Ensure no gradients are computed
                batch_embeddings = self.model.encode(batch, convert_to_tensor=True)
                all_embeddings.append(batch_embeddings.cpu())  # Move to CPU immediately
                
            # Force garbage collection after each batch
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        # Concatenate all batch embeddings
        return torch.cat(all_embeddings) if all_embeddings else torch.tensor([])
    
    def rerank(self, query, candidates, doc_texts, top_k=1000, batch_size=8):
        """
        Re-rank candidates using semantic similarity, optimized for CPU
        """
        # Get documents to re-rank (only from candidates)
        doc_ids = [doc_id for doc_id, _ in candidates]
        texts = [doc_texts[doc_id] for doc_id in doc_ids]
        
        # Create query embedding
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Process document embeddings in smaller chunks to manage memory
        chunk_size = 500  # Process at most 500 docs at a time
        reranked_results = []
        
        for chunk_start in range(0, len(texts), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(texts))
            chunk_texts = texts[chunk_start:chunk_end]
            chunk_ids = doc_ids[chunk_start:chunk_end]
            
            # Create embeddings for this chunk
            doc_embeddings = self.encode_batch(chunk_texts, batch_size=batch_size)
            
            # Calculate cosine similarities
            similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
            
            # Convert to list of (doc_id, score) tuples for this chunk
            chunk_results = [(chunk_ids[i], similarities[i].item()) 
                             for i in range(len(chunk_ids))]
            
            # Add to overall results
            reranked_results.extend(chunk_results)
            
            # Clean up to free memory
            del doc_embeddings, similarities
            gc.collect()
        
        # Sort by score in descending order and return top_k
        reranked_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)
        return reranked_results[:top_k]


#------------------------------------------------------------
# Neural Re-ranking: BERT Cross-Encoder Approach (CPU-Optimized)
#------------------------------------------------------------

class BERTCrossEncoder:
    """
    Re-ranks documents using a BERT model for query-document relevance
    Optimized for CPU usage
    """
    def __init__(self, model_name=None, use_small_model=False):
        # Use smaller model by default for CPU
        if use_small_model or model_name is None:
            model_name = "cross-encoder/ms-marco-MiniLM-L-2-v2"  # Smaller model
        elif model_name != "cross-encoder/ms-marco-MiniLM-L-6-v2":
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Default model
        
        print(f"Loading BERT Cross-Encoder model: {model_name}")
        start_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    def score_pairs(self, query, documents, batch_size=4):  # Very small batch size for CPU
        """Score a list of query-document pairs with memory optimization"""
        # Prepare input pairs
        pairs = [[query, doc] for doc in documents]
        
        scores = []
        # Process in very small batches to avoid memory pressure
        for i in tqdm(range(0, len(pairs), batch_size), desc="Scoring document pairs"):
            batch_pairs = pairs[i:i+batch_size]
            
            # Tokenize with reduced max length for CPU
            features = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=256  # Reduced from 512 to save memory
            )
            
            # Get scores
            with torch.no_grad():
                outputs = self.model(**features)
                batch_scores = outputs.logits.squeeze(-1).numpy()
                scores.extend(batch_scores)
            
            # Clean up memory
            del features, outputs
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        return scores
    
    def rerank(self, query, candidates, doc_texts, top_k=1000, batch_size=4):
        """
        Re-rank candidates using BERT cross-encoder, optimized for CPU
        """
        # Get documents to re-rank
        doc_ids = [doc_id for doc_id, _ in candidates]
        documents = [doc_texts[doc_id] for doc_id in doc_ids]
        
        # Process in chunks to manage memory
        chunk_size = 200  # Process at most 200 docs at a time
        reranked_results = []
        
        for chunk_start in range(0, len(documents), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(documents))
            chunk_docs = documents[chunk_start:chunk_end]
            chunk_ids = doc_ids[chunk_start:chunk_end]
            
            # Score each query-document pair in this chunk
            scores = self.score_pairs(query, chunk_docs, batch_size=batch_size)
            
            # Create list of (doc_id, score) tuples for this chunk
            chunk_results = [(chunk_ids[i], scores[i]) for i in range(len(chunk_ids))]
            reranked_results.extend(chunk_results)
            
            # Clean up
            del scores
            gc.collect()
        
        # Sort by score in descending order and return top_k
        reranked_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)
        return reranked_results[:top_k]


#------------------------------------------------------------
# Main Execution Functions
#------------------------------------------------------------

def load_queries(queries_path):
    """Load queries from JSONL file"""
    queries = {}
    with jsonlines.open(queries_path) as reader:
        for query in reader:
            query_id = query["_id"]
            try:
                # Skip even queries if needed (as in Assignment 1)
                if int(query_id) % 2 == 0:
                    continue
            except ValueError:
                continue

            query_text = query.get("text", "")
            if query_text:
                queries[query_id] = query_text
    
    return queries


def process_queries(queries, inverted_index, idf, document_lengths, document_texts, 
                    reranker=None, top_k=1000, initial_k=100, batch_size=4):
    """
    Process all queries using initial retrieval and optional neural re-ranking
    
    Args:
        queries: Dictionary mapping query_id to query_text
        inverted_index, idf, document_lengths: TF-IDF system components
        document_texts: Dictionary mapping doc_ids to document texts
        reranker: Neural re-ranker object (optional)
        top_k: Number of final results to return
        initial_k: Number of documents to retrieve in initial phase
        batch_size: Batch size for neural processing
    """
    results = {}
    
    for query_id, query_text in tqdm(queries.items(), desc="Processing queries"):
        # Initial retrieval using TF-IDF
        initial_results = rank_documents(query_text, inverted_index, idf, document_lengths)
        
        # Take top initial_k results
        initial_results = initial_results[:initial_k]
        
        # Neural re-ranking if available
        if reranker is not None:
            reranked_results = reranker.rerank(
                query_text, 
                initial_results, 
                document_texts,
                top_k=min(top_k, len(initial_results)),
                batch_size=batch_size
            )
            results[query_id] = reranked_results
        else:
            # Just use TF-IDF results
            results[query_id] = initial_results[:top_k]
    
    return results


def write_results(results, output_path, run_name="neural_ir"):
    """Write results in TREC format"""
    with open(output_path, "w") as f:
        for query_id, docs in results.items():
            for rank, (doc_id, score) in enumerate(docs, start=1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")


def main():
    """Main execution function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Neural IR System (CPU-Optimized)")
    parser.add_argument("--corpus", default="corpus.jsonl", help="Path to corpus file")
    parser.add_argument("--queries", default="queries.jsonl", help="Path to queries file")
    parser.add_argument("--mode", default="all", choices=["title", "all"], help="Content mode (title or all)")
    parser.add_argument("--reranker", default="none", 
                      choices=["none", "sbert", "bert"], 
                      help="Re-ranker type: none (TF-IDF only), sbert, or bert")
    parser.add_argument("--initial_k", type=int, default=100, help="Number of initial documents to retrieve")
    parser.add_argument("--top_k", type=int, default=1000, help="Number of final results per query")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for neural processing")
    parser.add_argument("--output", default="results_output.txt", help="Output file path")
    parser.add_argument("--use_small_models", action="store_true", help="Use smallest models possible for CPU")
    args = parser.parse_args()
    
    # Verify neural libraries are available if needed
    if args.reranker != "none" and not NEURAL_AVAILABLE:
        print("Error: Neural libraries not available. Defaulting to TF-IDF only.")
        args.reranker = "none"
    
    # Build index and calculate IDF
    print(f"Building inverted index (mode: {args.mode})...")
    start_time = time.time()
    inverted_index, document_lengths, document_texts = build_inverted_index(args.corpus, mode=args.mode)
    print(f"Index built in {time.time() - start_time:.2f} seconds")
    
    print("Calculating IDF values...")
    total_docs = len(document_lengths)
    idf = calculate_idf(inverted_index, total_docs)
    
    print("Loading queries...")
    queries = load_queries(args.queries)
    
    # Initialize reranker if needed
    reranker = None
    run_name = "tfidf"
    
    if args.reranker == "sbert":
        reranker = SentenceBERTReranker(use_small_model=args.use_small_models)
        run_name = "neural_ir_sbert"
    elif args.reranker == "bert":
        reranker = BERTCrossEncoder(use_small_model=args.use_small_models)
        run_name = "neural_ir_bert"
    
    # Process queries
    print(f"Processing queries with {args.reranker} approach...")
    start_time = time.time()
    results = process_queries(
        queries, 
        inverted_index, 
        idf, 
        document_lengths, 
        document_texts, 
        reranker=reranker,
        top_k=args.top_k,
        initial_k=args.initial_k,
        batch_size=args.batch_size
    )
    print(f"Query processing completed in {time.time() - start_time:.2f} seconds")
    
    # Write results
    print(f"Writing results to {args.output}...")
    write_results(results, args.output, run_name=run_name)
    
    print("Done!")


if __name__ == "__main__":
    main()