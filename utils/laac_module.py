import numpy as np
import scipy.sparse as sp
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch
import sys
import time
from sentence_transformers import SentenceTransformer

def generate_text_for_node(node_id, dl, adjM, client, model_name):
    target_name = dl.nodes['name'][node_id]
    
    # Dynamically retrieve local topological context (limited to a maximum of 5 neighbors to avoid exceeding the context window) 
    neighbors = adjM[node_id].nonzero()[1][:5]
    neighbor_names = [dl.nodes['name'][n] for n in neighbors if n in dl.nodes['name']]
    

    prompt = f"""You are an expert in academic network analysis. Your task is to infer the semantic profile of a specific target node based on its heterogeneous network environment.
Topological Context: The Target Node is named '{target_name}'. It exhibits the following local structural connections:
Neighbors: {', '.join(neighbor_names)}

Task Instruction: Analyzing the heterogeneous connections provided above, generate a concise semantic summary describing the primary characteristics and features of the Target Node. Limit the output to a single comprehensive paragraph containing highly discriminative keywords. Do not include introductory or concluding conversational phrases."""

    try:
        # Real-time output of request information
        print(f"\n  [LLM Request] Node {node_id} ('{target_name}')", flush=True)
        print(f"    Neighbors ({len(neighbor_names)}): {', '.join(neighbor_names[:5])}{'...' if len(neighbor_names) > 5 else ''}", flush=True)
        sys.stdout.flush()
        
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.2
        )
        elapsed_time = time.time() - start_time
        
        result = response.choices[0].message.content.strip()
        
        # Real-time output of response result and time taken
        preview = result[:100] + '...' if len(result) > 100 else result
        print(f"    Response: {preview}", flush=True)
        sys.stdout.flush()
        
        return result
    except Exception as e:
        elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
        print(f"    [ERROR] LLM failed for node {node_id} after {elapsed_time:.2f}s: {e}", flush=True)
        return target_name # fallback 

def complete_attributes_laac(node_type_id, dl, adjM, base_url="http://localhost:8080/v1"):
    client = OpenAI(api_key="EMPTY", base_url=base_url)
    
    # Dynamically retrieve the first model from the model list (suitable for local vLLM/Ollama deployment)  
    models = client.models.list()
    llm_model = models.data[0].id

    start_idx = dl.nodes['shift'][node_type_id]
    count = dl.nodes['count'][node_type_id]
    end_idx = start_idx + count
    
    node_ids = list(range(start_idx, end_idx))
    inferred_texts = [""] * count

    print(f"[*] LAAC: Inferring semantics for Node Type {node_type_id} (Nodes: {count})...")
    print(f"[*] Using model: {llm_model} | Base URL: {base_url}")
    print(f"[*] Concurrent workers: 64 | Total nodes: {count}\n")
    sys.stdout.flush()
    
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(generate_text_for_node, nid, dl, adjM, client, llm_model): (i, nid) for i, nid in enumerate(node_ids)}
        
        completed_count = 0
        start_time = time.time()
        
        for future in as_completed(futures):
            idx, node_id = futures[future]
            try:
                result = future.result()
                inferred_texts[idx] = result
                completed_count += 1
                
                # Output the processing status of each node in real time
                if completed_count % 5 == 0 or completed_count == len(node_ids):
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed_count
                    remaining = avg_time * (len(node_ids) - completed_count)
                    print(f"\r[Progress] {completed_count}/{len(node_ids)} nodes | "
                          f"Elapsed: {elapsed:.1f}s | "
                          f"Avg: {avg_time:.2f}s/node | "
                          f"ETA: {remaining:.1f}s", flush=True)
                    sys.stdout.flush()
                    
            except Exception as e:
                print(f"\n[Error] Failed to process node {node_id}: {e}", flush=True)
                inferred_texts[idx] = dl.nodes['name'][node_id]  # fallback
                completed_count += 1
    
    total_time = time.time() - start_time
    print(f"\n[*] LAAC: Completed inference for all {len(node_ids)} nodes in {total_time:.2f}s")
    print(f"[*] Average time per node: {total_time/len(node_ids):.2f}s\n")
    sys.stdout.flush()
    print(f"[*] LAAC: Vectorizing semantics via Local Embedding Model...")
    sys.stdout.flush()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Use BGE large model as the text embedding engine, output dimension is 1024
    print(f"[*] Loading embedding model from: /your/local/path/bge-large-en-v1.5 on {device}")
    sys.stdout.flush()
    emb_model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device) #download from huggingface 
    
    print(f"[*] Encoding {len(inferred_texts)} texts with batch_size=256...")
    sys.stdout.flush()
    embeddings = emb_model.encode(inferred_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    print(f"[*] LAAC: Vectorization completed. Shape: {embeddings.shape}")
    sys.stdout.flush()
    
    return sp.csr_matrix(embeddings)