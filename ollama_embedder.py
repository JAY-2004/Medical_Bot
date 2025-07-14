'''#actual

import os
import pandas as pd
import numpy as np
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from sklearn.metrics.pairwise import cosine_similarity

class CDTEmbedder:
    def __init__(self, cdt_path="New_CDT.xlsx", model_name="nomic-embed-text:latest", index_path="faiss_cdt.index"):
        self.cdt_path = cdt_path
        self.model_name = model_name
        self.index_path = index_path

        # Load and preprocess CDT data
        self.df = pd.read_excel(cdt_path)
        self.df = self.df.rename(columns={
            "Procedure Code": "Code",
            "Description of Service": "Description",
            "Keywords": "Keywords"
        })
        self.df = self.df.dropna(subset=['Code', 'Description'])
        self.df['Keywords'] = self.df['Keywords'].fillna("").astype(str).str.lower().str.strip()

        # Format text better before embedding
        def format_text(row):
            return (
                f"CDT Code: {row['Code']}\n"
                f"Service: {row['Description']}\n"
                f"Keywords: {row['Keywords']}"
            )

        self.texts = self.df.apply(format_text, axis=1).tolist()

        # Initialize Ollama embedding model
        self.embedding = OllamaEmbeddings(model=model_name)

        # Load or create FAISS index
        if os.path.exists(self.index_path):
            print(f"Loading FAISS index from {self.index_path}...")
            self.vectorstore = FAISS.load_local(self.index_path, self.embedding, allow_dangerous_deserialization=True)
        else:
            print("Creating FAISS index from documents, this may take a moment...")
            docs = [Document(page_content=text) for text in self.texts]
            self.vectorstore = FAISS.from_documents(docs, self.embedding)
            print(f"Saving FAISS index to {self.index_path}...")
            self.vectorstore.save_local(self.index_path)

        # Precompute embeddings for re-ranking
        self.embeddings_matrix = self.embedding.embed_documents(self.texts)

    def _jaccard_similarity(self, set1, set2):
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union else 0.0

    def retrieve_best_match(self, query, k=10):
        # Step 1: FAISS similarity search for initial candidates
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        # Embed query for exact cosine similarity
        query_embedding = np.array(self.embedding.embed_query(query)).reshape(1, -1)

        re_ranked = []
        query_keywords = set(query.lower().split())

        for doc, score in results:
            lines = doc.page_content.split('\n')
            if len(lines) >= 2:
                # Extract code from first line "CDT Code: 1234"
                code = lines[0].replace("CDT Code:", "").strip()
                # Extract description from second line "Service: ... "
                desc = lines[1].replace("Service:", "").strip()
                # Extract keywords from third line if present
                keywords_line = lines[2] if len(lines) > 2 else ""
                candidate_keywords = set(keywords_line.replace("Keywords:", "").strip().split())

                # Find index in original dataframe
                idx = self.df[self.df['Code'].astype(str) == code].index[0]

                # Calculate exact cosine similarity with precomputed embedding
                candidate_emb = np.array(self.embeddings_matrix[idx]).reshape(1, -1)
                cos_sim = cosine_similarity(query_embedding, candidate_emb)[0][0]

                # Jaccard similarity on keywords
                jac_sim = self._jaccard_similarity(query_keywords, candidate_keywords)

                # Combine scores (weights can be tuned)
                combined_score = 0.7 * cos_sim + 0.3 * jac_sim

                re_ranked.append({
                    "Code": code,
                    "Description": desc,
                    "Cosine": cos_sim,
                    "Jaccard": jac_sim,
                    "Score": combined_score
                })

        # Sort by combined score descending
        re_ranked_sorted = sorted(re_ranked, key=lambda x: x['Score'], reverse=True)[:k]

        # Convert to DataFrame
        return pd.DataFrame(re_ranked_sorted), re_ranked_sorted

'''

import os
import time
import pandas as pd
import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS  # Updated import path
from langchain_core.documents import Document  # Updated import path
from sklearn.metrics.pairwise import cosine_similarity

class CDTEmbedder:
    def __init__(self, cdt_path="New_CDT.xlsx", model_name="nomic-embed-text:latest", index_path="faiss_cdt.index"):
        self.cdt_path = cdt_path
        self.model_name = model_name
        self.index_path = index_path

        # Load and preprocess CDT data
        self.df = pd.read_excel(cdt_path)
        self.df = self.df.rename(columns={
            "Procedure Code": "Code",
            "Description of Service": "Description",
            "Keywords": "Keywords"
        })
        self.df = self.df.dropna(subset=['Code', 'Description'])
        self.df['Keywords'] = self.df['Keywords'].fillna("").astype(str).str.lower().str.strip()

        # Format text better before embedding
        def format_text(row):
            return (
                f"CDT Code: {row['Code']}\n"
                f"Service: {row['Description']}\n"
                f"Keywords: {row['Keywords']}"
            )

        self.texts = self.df.apply(format_text, axis=1).tolist()

        # Initialize Ollama embedding model with error handling
        try:
            print(f"Initializing Ollama embeddings with model: {model_name}")
            self.embedding = OllamaEmbeddings(model=model_name)
            # Test embedding
            _ = self.embedding.embed_query("test")
            print("Ollama embeddings initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Ollama embeddings: {str(e)}")

        # Load or create FAISS index with improved error handling
        try:
            if os.path.exists(self.index_path):
                print(f"Loading FAISS index from {self.index_path}...")
                start_time = time.time()
                self.vectorstore = FAISS.load_local(
                    self.index_path, 
                    self.embedding, 
                    allow_dangerous_deserialization=True
                )
                print(f"Index loaded in {time.time() - start_time:.2f} seconds")
            else:
                print("Creating FAISS index from documents...")
                start_time = time.time()
                docs = [Document(page_content=text) for text in self.texts]
                self.vectorstore = FAISS.from_documents(docs, self.embedding)
                print(f"Index created in {time.time() - start_time:.2f} seconds")
                print(f"Saving FAISS index to {self.index_path}...")
                self.vectorstore.save_local(self.index_path)
        except Exception as e:
            print(f"Error handling FAISS index: {str(e)}")
            # Attempt to rebuild if corrupted
            print("Attempting to rebuild FAISS index...")
            docs = [Document(page_content=text) for text in self.texts]
            self.vectorstore = FAISS.from_documents(docs, self.embedding)
            self.vectorstore.save_local(self.index_path)

        # Precompute embeddings for re-ranking with progress feedback
        print("Precomputing embeddings matrix...")
        start_time = time.time()
        batch_size = 32  # Process in batches to avoid memory issues
        self.embeddings_matrix = []
        for i in range(0, len(self.texts), batch_size):
            batch = self.texts[i:i + batch_size]
            self.embeddings_matrix.extend(self.embedding.embed_documents(batch))
            print(f"Processed {min(i + batch_size, len(self.texts))}/{len(self.texts)} embeddings")
        self.embeddings_matrix = np.array(self.embeddings_matrix)
        print(f"Embeddings matrix computed in {time.time() - start_time:.2f} seconds")

    def _jaccard_similarity(self, set1, set2):
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union else 0.0

    def retrieve_best_match(self, query, k=10):
        """Retrieve best matching CDT codes with enhanced error handling"""
        try:
            # Step 1: FAISS similarity search for initial candidates
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            # Embed query for exact cosine similarity
            query_embedding = np.array(self.embedding.embed_query(query)).reshape(1, -1)

            re_ranked = []
            query_keywords = set(query.lower().split())

            for doc, score in results:
                try:
                    lines = doc.page_content.split('\n')
                    if len(lines) >= 2:
                        # Extract code from first line "CDT Code: 1234"
                        code = lines[0].replace("CDT Code:", "").strip()
                        # Extract description from second line "Service: ... "
                        desc = lines[1].replace("Service:", "").strip()
                        # Extract keywords from third line if present
                        keywords_line = lines[2] if len(lines) > 2 else ""
                        candidate_keywords = set(keywords_line.replace("Keywords:", "").strip().split())

                        # Find index in original dataframe
                        idx = self.df[self.df['Code'].astype(str) == code].index[0]

                        # Calculate exact cosine similarity with precomputed embedding
                        candidate_emb = np.array(self.embeddings_matrix[idx]).reshape(1, -1)
                        cos_sim = cosine_similarity(query_embedding, candidate_emb)[0][0]

                        # Jaccard similarity on keywords
                        jac_sim = self._jaccard_similarity(query_keywords, candidate_keywords)

                        # Combine scores (weights can be tuned)
                        combined_score = 0.7 * cos_sim + 0.3 * jac_sim

                        re_ranked.append({
                            "Code": code,
                            "Description": desc,
                            "Cosine": cos_sim,
                            "Jaccard": jac_sim,
                            "Score": combined_score
                        })
                except Exception as e:
                    print(f"Error processing document: {str(e)}")
                    continue

            # Sort by combined score descending
            re_ranked_sorted = sorted(re_ranked, key=lambda x: x['Score'], reverse=True)[:k]

            # Convert to DataFrame
            return pd.DataFrame(re_ranked_sorted), re_ranked_sorted

        except Exception as e:
            print(f"Error in retrieve_best_match: {str(e)}")
            # Return empty results if error occurs
            return pd.DataFrame(), []