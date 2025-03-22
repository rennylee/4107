"""
Fine-tuning script for BERT Cross-Encoder (CPU-Optimized)

This script fine-tunes a small BERT model for query-document relevance prediction
optimized to run on CPU systems with limited memory.

Usage:
  python finetune_bert.py --corpus corpus.jsonl --queries queries.jsonl --qrels path/to/qrels
"""

import os
import json
import jsonlines
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import random
import gc
import time

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import neural libraries with error handling
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        TrainerCallback
    )
    from datasets import Dataset as HFDataset
    from sklearn.model_selection import train_test_split
    NEURAL_AVAILABLE = True
except ImportError:
    print("Warning: Neural libraries not fully available. Installing required packages...")
    import subprocess
    subprocess.call(["pip", "install", "torch", "transformers", "datasets", "scikit-learn"])
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            Trainer,
            TrainingArguments,
            TrainerCallback
        )
        from datasets import Dataset as HFDataset
        from sklearn.model_selection import train_test_split
        NEURAL_AVAILABLE = True
    except ImportError:
        print("Error: Could not install required packages.")
        NEURAL_AVAILABLE = False

print(f"Using device: CPU")


def load_corpus(corpus_path, max_docs=None):
    """Load corpus documents into memory with limit for testing"""
    documents = {}
    with jsonlines.open(corpus_path) as reader:
        for i, doc in enumerate(reader):
            if max_docs and i >= max_docs:
                break
                
            doc_id = doc["_id"]
            title = doc.get("title", "")
            text = doc.get("text", "")
            
            # For CPU efficiency, use only title + first 100 words of text
            if text:
                text_words = text.split()[:100]
                text = " ".join(text_words)
                
            documents[doc_id] = title + " " + text
    return documents


def load_queries(queries_path):
    """Load queries into memory"""
    queries = {}
    with jsonlines.open(queries_path) as reader:
        for query in reader:
            query_id = query["_id"]
            query_text = query.get("text", "")
            if query_text:
                queries[query_id] = query_text
    return queries


def load_qrels(qrels_path):
    """Load relevance judgments from qrels file"""
    qrels = {}
    try:
        with open(qrels_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    query_id, _, doc_id, relevance = parts
                    
                    # Initialize query qrels if needed
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    
                    # Store relevance (convert to int)
                    qrels[query_id][doc_id] = int(relevance)
    except Exception as e:
        print(f"Error loading qrels file: {e}")
        return {}
        
    return qrels


def create_training_data(queries, documents, qrels, num_negatives=1, max_examples=10000):
    """
    Create training data for BERT fine-tuning, limited for CPU usage
    
    Args:
        queries: Dictionary mapping query_id to query_text
        documents: Dictionary mapping doc_id to document_text
        qrels: Dictionary mapping query_id to a dict of doc_id->relevance
        num_negatives: Number of negative examples per positive example
        max_examples: Maximum number of training examples to create
        
    Returns:
        List of dictionaries with fields: query, document, label
    """
    training_data = []
    
    # Create positive and negative examples
    for query_id, query_text in tqdm(queries.items(), desc="Creating training data"):
        # Skip if no relevance judgments for this query
        if query_id not in qrels or not qrels[query_id]:
            continue
        
        # Get relevant documents for this query
        relevant_docs = [doc_id for doc_id, rel in qrels[query_id].items() if rel > 0]
        
        # Skip if no relevant documents
        if not relevant_docs:
            continue
        
        # Create positive examples
        for doc_id in relevant_docs:
            # Skip if document not in corpus
            if doc_id not in documents:
                continue
            
            # Add positive example
            training_data.append({
                "query": query_text,
                "document": documents[doc_id],
                "label": 1
            })
            
            # Check if we've reached the maximum
            if len(training_data) >= max_examples/2:
                break
        
        # Create negative examples
        if len(training_data) >= max_examples/2:
            break
            
        # Sample negative examples
        irrelevant_docs = [doc_id for doc_id in list(documents.keys())[:1000] 
                          if doc_id not in qrels[query_id] or qrels[query_id][doc_id] == 0]
        
        # Sample negative examples
        num_to_sample = min(len(relevant_docs) * num_negatives, len(irrelevant_docs), 10)
        sampled_negatives = random.sample(irrelevant_docs, num_to_sample)
        
        for doc_id in sampled_negatives:
            training_data.append({
                "query": query_text,
                "document": documents[doc_id],
                "label": 0
            })
            
            # Check if we've reached the maximum
            if len(training_data) >= max_examples:
                break
                
        if len(training_data) >= max_examples:
            break
    
    print(f"Created {len(training_data)} training examples (max: {max_examples})")
    return training_data


def tokenize_function(examples, tokenizer, max_length=128):
    """Tokenize query-document pairs for BERT (reduced length for CPU)"""
    queries = examples["query"]
    documents = examples["document"]
    
    # Tokenize query-document pairs with shorter max_length for CPU
    tokenized = tokenizer(
        queries,
        documents,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return tokenized


class MemoryCallback(TrainerCallback):
    """Callback to monitor and free memory during training"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Free memory more aggressively
        gc.collect()


def finetune_bert(train_dataset, val_dataset, model_name="prajjwal1/bert-tiny", output_dir="bert-finetuned-cpu"):
    """
    Fine-tune small BERT model for relevance ranking on CPU
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        model_name: Pre-trained model name (tiny model for CPU)
        output_dir: Directory to save fine-tuned model
        
    Returns:
        Fine-tuned model and tokenizer
    """
    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Prepare training arguments with minimal settings for CPU
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Reduce for CPU
        per_device_train_batch_size=4,  # Small batch size for CPU
        per_device_eval_batch_size=4,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch
        fp16=False,  # No mixed precision on CPU
        dataloader_num_workers=0  # No parallel data loading on CPU
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[MemoryCallback()]
    )
    
    # Train model
    print("Starting fine-tuning (reduced training for CPU)...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer


def main():
    """Main execution function"""
    # Verify neural libraries are available
    if not NEURAL_AVAILABLE:
        print("Error: Required neural libraries are not available.")
        return
        
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune BERT for relevance ranking (CPU-Optimized)")
    parser.add_argument("--corpus", default="corpus.jsonl", help="Path to corpus file")
    parser.add_argument("--queries", default="queries.jsonl", help="Path to queries file")
    parser.add_argument("--qrels", required=True, help="Path to qrels file")
    parser.add_argument("--model", default="prajjwal1/bert-tiny", help="Pre-trained model name")
    parser.add_argument("--output", default="bert-finetuned-cpu", help="Output directory for fine-tuned model")
    parser.add_argument("--max_docs", type=int, default=1000, help="Maximum documents to load for CPU efficiency")
    parser.add_argument("--max_examples", type=int, default=2000, help="Maximum training examples to create")
    args = parser.parse_args()
    
    # Load data with limits for CPU
    print(f"Loading corpus (limited to {args.max_docs} documents for CPU)...")
    documents = load_corpus(args.corpus, max_docs=args.max_docs)
    
    print("Loading queries...")
    queries = load_queries(args.queries)
    
    print("Loading qrels...")
    qrels = load_qrels(args.qrels)
    
    # Create training data
    print("Creating training data...")
    training_data = create_training_data(
        queries, documents, qrels, 
        num_negatives=1,  # Reduced for CPU
        max_examples=args.max_examples
    )
    
    if not training_data:
        print("Error: No training data could be created.")
        return
    
    # Split into train and validation sets
    train_data, val_data = train_test_split(training_data, test_size=0.1, random_state=42)
    
    # Convert to HuggingFace datasets
    print("Preparing datasets...")
    train_dataset = HFDataset.from_pandas(pd.DataFrame(train_data))
    val_dataset = HFDataset.from_pandas(pd.DataFrame(val_data))
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Apply tokenization with wrapper function for CPU optimization
    def tokenize_and_prepare(examples):
        return tokenize_function(examples, tokenizer, max_length=128)  # Reduced length for CPU
    
    print("Tokenizing training data...")
    train_dataset = train_dataset.map(
        tokenize_and_prepare,
        batched=True,
        batch_size=32,  # Smaller batch for CPU
        remove_columns=["query", "document"]
    )
    
    print("Tokenizing validation data...")
    val_dataset = val_dataset.map(
        tokenize_and_prepare,
        batched=True,
        batch_size=32,  # Smaller batch for CPU
        remove_columns=["query", "document"]
    )
    
    # Fine-tune model
    print(f"Fine-tuning {args.model} with CPU optimizations...")
    finetune_bert(
        train_dataset,
        val_dataset,
        model_name=args.model,
        output_dir=args.output
    )
    
    print(f"Fine-tuned model saved to {args.output}")
    print("Note: This is a minimally fine-tuned model due to CPU constraints.")
    print("For production use, consider using a GPU for more thorough training.")


if __name__ == "__main__":
    main()