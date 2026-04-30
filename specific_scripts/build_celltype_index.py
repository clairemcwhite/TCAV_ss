"""
=============================================================================
BUILD CELLTYPE INDEX FOR SEMANTIC CELL TYPE SEARCH
=============================================================================

This script builds a FAISS index of all cell types from CellXGene (Tabula Sapiens).
The index can be used for semantic similarity search to retrieve cells of specific types
and controls for downstream analysis (e.g., geneformer input).

Output files:
- celltype_embeddings.npy: Embeddings for all cell types
- celltype_list.pkl: List of cell type names
- celltype_index.faiss: FAISS index for similarity search
"""

import os
import sys
import argparse
import numpy as np
import pickle
import hnswlib
from google.genai import Client
from cellxgene_census import open_soma
import cellxgene_census

def build_celltype_index(
    output_dir: str,
    api_key: str,
    embedding_model: str = "gemini-embedding-001",
    batch_size: int = 100,
    verbose: bool = True
) -> dict:
    """
    Build a FAISS index of all cell types from Tabula Sapiens.
    
    Parameters
    ----------
    output_dir : str
        Directory to save index files
    api_key : str
        Google API key for Gemini embeddings
    embedding_model : str, optional
        Model to use for embeddings (default: gemini-embedding-001)
    batch_size : int, optional
        Batch size for embedding generation (default: 100)
    verbose : bool, optional
        Print progress messages (default: True)
    
    Returns
    -------
    dict
        Summary of created index with file paths and stats
    """
    
    # Setup
    client = Client(api_key=api_key)
    census = open_soma()
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"✅ Output directory: {output_dir}")
    
    # ====================================================================
    # STEP 1: Get all unique cell types from Tabula Sapiens
    # ====================================================================
    
    if verbose:
        print("\n📋 STEP 1: Retrieving all cell types from Tabula Sapiens...")
    
    obs_metadata = cellxgene_census.get_obs(
        census,
        organism="homo_sapiens",
        value_filter="dataset_id == '53d208b0-2cfd-4366-9866-c3c6114081bc' and is_primary_data == True",
        column_names=["cell_type"]
    )
    
    celltype_list = sorted(obs_metadata["cell_type"].unique().tolist())
    
    if verbose:
        print(f"✅ Found {len(celltype_list)} unique cell types")
        print(f"   Sample types: {celltype_list[:5]}")
    
    # ====================================================================
    # STEP 2: Generate embeddings for each cell type
    # ====================================================================
    
    if verbose:
        print(f"\n🧠 STEP 2: Generating embeddings for all cell types...")
        print(f"   Using model: {embedding_model}")
    
    embeddings_list = []
    
    for i in range(0, len(celltype_list), batch_size):
        batch = celltype_list[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(celltype_list) - 1) // batch_size + 1
        
        if verbose:
            print(f"   Processing batch {batch_num}/{total_batches} ({len(batch)} cell types)...")
        
        try:
            result = client.models.embed_content(
                model=embedding_model,
                contents=batch
            )
            
            # Extract embeddings from result
            # New SDK returns EmbedContentResponse with embedding field
            if hasattr(result, 'embeddings'):
                embeddings_list.extend(result.embeddings)
            elif isinstance(result, dict):
                embeddings_list.extend(result.get('embeddings', []))
            
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Error embedding batch: {e}")
            # Fall back to individual embeddings
            for ct in batch:
                try:
                    result = client.models.embed_content(
                        model=embedding_model,
                        contents=[ct]
                    )
                    
                    # Extract embedding
                    if hasattr(result, 'embeddings') and result.embeddings:
                        embeddings_list.append(result.embeddings[0])
                    elif isinstance(result, dict):
                        embs = result.get('embeddings', [])
                        if embs:
                            embeddings_list.append(embs[0])
                            
                except Exception as e2:
                    if verbose:
                        print(f"   ❌ Failed to embed '{ct}': {e2}")
    
    if verbose:
        print(f"✅ Generated {len(embeddings_list)} embeddings")
    
    # Convert to numpy array
    # New SDK returns Embedding objects with values attribute
    embedding_values = []
    for emb in embeddings_list:
        if hasattr(emb, 'values'):
            # Embedding object with values attribute
            embedding_values.append(emb.values)
        elif isinstance(emb, (list, np.ndarray)):
            # Already a list/array
            embedding_values.append(emb)
        elif isinstance(emb, dict) and 'values' in emb:
            # Dict format with 'values' key
            embedding_values.append(emb['values'])
        else:
            # Try to convert to array
            embedding_values.append(np.array(emb))
    
    embeddings_array = np.array(embedding_values, dtype='float32')
    
    if verbose:
        print(f"   Shape: {embeddings_array.shape}")
    
    # ====================================================================
    # STEP 3: Create hnswlib index
    # ====================================================================
    
    if verbose:
        print("\n🔍 STEP 3: Creating hnswlib index...")
    
    dimension = embeddings_array.shape[1]
    num_items = embeddings_array.shape[0]
    
    # Create index with L2 distance
    # max_m controls the number of connections per node (higher = more accurate but slower)
    # ef_construction controls search breadth during construction (higher = more accurate)
    index = hnswlib.Index(space='l2', dim=dimension)
    index.init_index(max_elements=num_items, ef_construction=200, M=16)
    
    # Add all vectors with their indices
    item_ids = np.arange(num_items)
    index.add_items(embeddings_array, item_ids)
    
    if verbose:
        print(f"✅ Index created with {index.get_current_count()} cell type embeddings")
        print(f"   Dimension: {dimension}")
        print(f"   Search parameter (ef): 200")
    
    # ====================================================================
    # STEP 4: Save index and metadata
    # ====================================================================
    
    if verbose:
        print("\n💾 STEP 4: Saving index and metadata...")
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, "celltype_embeddings.npy")
    np.save(embeddings_path, embeddings_array)
    if verbose:
        print(f"✅ Saved embeddings: {embeddings_path}")
    
    # Save cell type list
    celltype_list_path = os.path.join(output_dir, "celltype_list.pkl")
    with open(celltype_list_path, 'wb') as f:
        pickle.dump(celltype_list, f)
    if verbose:
        print(f"✅ Saved cell type list: {celltype_list_path}")
    
    # Save hnswlib index
    index_path = os.path.join(output_dir, "celltype_index.bin")
    index.save_index(index_path)
    if verbose:
        print(f"✅ Saved hnswlib index: {index_path}")
    
    # ====================================================================
    # SUMMARY
    # ====================================================================
    
    result_summary = {
        "status": "success",
        "output_dir": output_dir,
        "num_cell_types": len(celltype_list),
        "embedding_shape": embeddings_array.shape,
        "files": {
            "celltype_list": celltype_list_path,
            "embeddings": embeddings_path,
            "index": index_path
        }
    }
    
    if verbose:
        print("\n" + "="*70)
        print("✅ CELLTYPE INDEX BUILD COMPLETE")
        print("="*70)
        print(f"\nFiles created in {output_dir}:")
        print(f"  1. celltype_list.pkl          - {len(celltype_list)} cell types")
        print(f"  2. celltype_embeddings.npy    - {embeddings_array.shape}")
        print(f"  3. celltype_index.bin         - hnswlib index for semantic search")
        print("\nYou can now use these files in retrieve_cells_and_controls.py")
        print("="*70)
    
    return result_summary


def main():
    """Command-line interface for building celltype index."""
    
    parser = argparse.ArgumentParser(
        description="Build a FAISS index of all cell types from CellXGene (Tabula Sapiens)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings with environment API key
  python build_celltype_index.py --output-dir ./celltype_index
  
  # Specify API key explicitly
  python build_celltype_index.py --output-dir ./celltype_index --api-key YOUR_KEY
  
  # Custom batch size and model
  python build_celltype_index.py --output-dir ./idx --batch-size 50 \\
    --embedding-model gemini-embedding-001
        """
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save index files"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Google API key (default: GOOGLE_API_KEY env var)"
    )
    
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="gemini-embedding-001",
        help="Embedding model to use (default: gemini-embedding-001)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding generation (default: 100)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress messages (default: True)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: No API key provided")
        print("   Set GOOGLE_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Build index
    result = build_celltype_index(
        output_dir=args.output_dir,
        api_key=api_key,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        verbose=args.verbose
    )
    
    return result


if __name__ == "__main__":
    main()
