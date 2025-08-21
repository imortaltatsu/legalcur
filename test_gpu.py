#!/usr/bin/env python3
"""
Test script to verify GPU acceleration for embeddings
"""

import torch
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer
import time

def test_gpu_acceleration():
    print("=== GPU Acceleration Test ===")
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    # Test Accelerator
    print("\n=== Accelerator Test ===")
    accelerator = Accelerator()
    print(f"Device: {accelerator.device}")
    print(f"Processes: {accelerator.num_processes}")
    
    # Test embedding model
    print("\n=== Embedding Model Test ===")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Test on CPU
    print("Testing on CPU...")
    start_time = time.time()
    model_cpu = SentenceTransformer(model_name, device='cpu')
    cpu_embeddings = model_cpu.encode(["This is a test sentence for embeddings."], show_progress_bar=False)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.3f}s")
    
    # Test on GPU if available
    if torch.cuda.is_available():
        print("\nTesting on GPU...")
        start_time = time.time()
        model_gpu = SentenceTransformer(model_name, device='cuda')
        gpu_embeddings = model_gpu.encode(["This is a test sentence for embeddings."], show_progress_bar=False)
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.3f}s")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
        
        # Test batch processing
        print("\nTesting batch processing on GPU...")
        test_sentences = ["Sentence " + str(i) + " for testing." for i in range(100)]
        start_time = time.time()
        batch_embeddings = model_gpu.encode(test_sentences, batch_size=32, show_progress_bar=False)
        batch_time = time.time() - start_time
        print(f"Batch processing time (100 sentences): {batch_time:.3f}s")
        print(f"Sentences per second: {100/batch_time:.1f}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_gpu_acceleration()
