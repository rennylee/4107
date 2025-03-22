"""
Evaluation script for the Neural IR System (CPU-Optimized)

This script runs the different IR methods and evaluates their performance using
the trec_eval tool. It compares:
1. Baseline TF-IDF system
2. SentenceBERT re-ranking approach
3. BERT Cross-Encoder re-ranking approach

Usage:
  python evaluate_ir_systems.py --qrels path/to/qrels_file
"""

import os
import subprocess
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def run_trec_eval(results_file, qrels_file, metrics=None):
    """
    Run trec_eval and parse the results
    
    Args:
        results_file: Path to results file in TREC format
        qrels_file: Path to qrels file
        metrics: List of metrics to evaluate (None for all)
        
    Returns:
        Dictionary of metric -> value
    """
    # Build command
    cmd = ["./trec_eval-9.0.7/trec_eval"]
    if metrics:
        for metric in metrics:
            cmd.extend(["-m", metric])
    cmd.extend([qrels_file, results_file])
    
    # Run command
    try:
        output = subprocess.check_output(cmd, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running trec_eval: {e}")
        print("Trying alternate trec_eval path...")
        # Try alternate path
        cmd[0] = "trec_eval"
        try:
            output = subprocess.check_output(cmd, universal_newlines=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e2:
            print(f"Error running trec_eval from alternate path: {e2}")
            return {}
    except FileNotFoundError:
        print("trec_eval not found at ./trec_eval-9.0.7/trec_eval. Trying 'trec_eval' directly...")
        cmd[0] = "trec_eval"
        try:
            output = subprocess.check_output(cmd, universal_newlines=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error: Could not find or run trec_eval: {e}")
            return {}
    
    # Parse output
    results = {}
    for line in output.strip().split("\n"):
        parts = line.split()
        if len(parts) == 3:
            metric, _, value = parts
            results[metric] = float(value)
    
    return results


def run_evaluation(methods, qrels_file):
    """
    Run evaluation for multiple methods
    
    Args:
        methods: Dictionary mapping method names to result file paths
        qrels_file: Path to qrels file
        
    Returns:
        DataFrame with evaluation results
    """
    metrics = ["map", "P.10", "bpref", "recip_rank"]
    results = {}
    
    for method_name, results_file in tqdm(methods.items(), desc="Evaluating methods"):
        # Skip if results file doesn't exist
        if not os.path.exists(results_file):
            print(f"Warning: Results file not found for {method_name}: {results_file}")
            continue
        
        # Run trec_eval
        eval_results = run_trec_eval(results_file, qrels_file, metrics)
        if eval_results:
            results[method_name] = eval_results
    
    # Convert to DataFrame
    df = pd.DataFrame(results).T if results else pd.DataFrame()
    
    # Rename columns for readability
    column_mapping = {
        "map": "MAP",
        "P.10": "P@10",
        "bpref": "BPref",
        "recip_rank": "MRR"
    }
    df = df.rename(columns=column_mapping)
    
    return df


def plot_results(df, output_file="evaluation_results.png"):
    """Plot evaluation results"""
    try:
        ax = df.plot(kind="bar", figsize=(10, 6))
        ax.set_title("IR System Performance Comparison")
        ax.set_ylabel("Score")
        ax.set_xlabel("Method")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Evaluation plot saved to {output_file}")
    except Exception as e:
        print(f"Warning: Could not generate plot: {e}")


def get_top_results(results_file, query_ids=None, top_n=10):
    """
    Get top N results for specified queries
    
    Args:
        results_file: Path to results file in TREC format
        query_ids: List of query IDs to extract (None for all)
        top_n: Number of top results to extract
        
    Returns:
        Dictionary mapping query_id to list of (rank, doc_id, score) tuples
    """
    query_results = {}
    
    try:
        with open(results_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    query_id, _, doc_id, rank, score, _ = parts
                    
                    # Skip if not in requested query_ids
                    if query_ids and query_id not in query_ids:
                        continue
                    
                    # Initialize query results if needed
                    if query_id not in query_results:
                        query_results[query_id] = []
                    
                    # Add result if within top_n
                    if int(rank) <= top_n:
                        query_results[query_id].append((int(rank), doc_id, float(score)))
    except Exception as e:
        print(f"Error reading results file {results_file}: {e}")
        return {}
    
    # Sort results by rank
    for query_id in query_results:
        query_results[query_id].sort(key=lambda x: x[0])
    
    return query_results


def run_neural_ir_system(method, corpus, queries, use_small_models=True):
    """Run the neural IR system with the specified method"""
    import subprocess
    import sys
    
    output_file = f"results_{method}.txt"
    
    # Build command
    cmd = [sys.executable, "neural_ir_system.py",
           "--corpus", corpus,
           "--queries", queries,
           "--reranker", method if method != "tfidf" else "none",
           "--output", output_file,
           "--batch_size", "4"]
    
    if use_small_models:
        cmd.append("--use_small_models")
    
    # Run command
    print(f"Running {method} method...")
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        print(f"{method} completed in {time.time() - start_time:.2f} seconds")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error running {method}: {e}")
        return None


def main():
    """Main execution function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate IR Systems")
    parser.add_argument("--qrels", required=True, help="Path to qrels file")
    parser.add_argument("--corpus", default="corpus.jsonl", help="Path to corpus file")
    parser.add_argument("--queries", default="queries.jsonl", help="Path to queries file")
    parser.add_argument("--run", action="store_true", help="Run the IR systems before evaluation")
    parser.add_argument("--baseline", default="results_tfidf.txt", help="Path to baseline results file")
    parser.add_argument("--sbert", default="results_sbert.txt", help="Path to SentenceBERT results file")
    parser.add_argument("--bert", default="results_bert.txt", help="Path to BERT Cross-Encoder results file")
    args = parser.parse_args()
    
    # Run IR systems if requested
    if args.run:
        print("Running IR systems...")
        tfidf_results = run_neural_ir_system("tfidf", args.corpus, args.queries, args.qrels)
        sbert_results = run_neural_ir_system("sbert", args.corpus, args.queries, args.qrels)
        bert_results = run_neural_ir_system("bert", args.corpus, args.queries, args.qrels)
        
        # Update file paths
        if tfidf_results:
            args.baseline = tfidf_results
        if sbert_results:
            args.sbert = sbert_results
        if bert_results:
            args.bert = bert_results
    
    # Define methods to evaluate
    methods = {
        "TF-IDF (Baseline)": args.baseline,
        "SentenceBERT Re-ranker": args.sbert,
        "BERT Cross-Encoder": args.bert
    }
    
    # Run evaluation
    print("\nEvaluating IR systems...")
    results_df = run_evaluation(methods, args.qrels)
    
    if results_df.empty:
        print("Error: No evaluation results were obtained.")
        return
    
    # Print results
    print("\nEvaluation Results:")
    print(results_df)
    
    # Highlight best scores
    print("\nBest Scores:")
    for column in results_df.columns:
        max_value = results_df[column].max()
        best_method = results_df[column].idxmax()
        print(f"{column}: {max_value:.4f} ({best_method})")
    
    # Plot results
    plot_results(results_df)
    
    # Extract top results for sample queries
    print("\nExtracting top 10 results for sample queries...")
    sample_queries = ["1", "3"]  # As requested in the assignment
    
    for method_name, results_file in methods.items():
        if os.path.exists(results_file):
            query_results = get_top_results(results_file, query_ids=sample_queries)
            
            # Print top 10 results for each sample query
            for query_id in sample_queries:
                if query_id in query_results:
                    print(f"\nTop 10 results for Query {query_id} using {method_name}:")
                    for rank, doc_id, score in query_results[query_id]:
                        print(f"{rank}. {doc_id} (Score: {score:.6f})")
    
    # Identify best method and prepare Results file
    if not results_df.empty and 'MAP' in results_df.columns:
        best_method = results_df['MAP'].idxmax()
        best_file = methods.get(best_method)
        if best_file and os.path.exists(best_file):
            print(f"\nCreating 'Results' file from best method: {best_method}")
            subprocess.run(["cp", best_file, "Results"])
            print("Results file created successfully!")


if __name__ == "__main__":
    main()