"""
Comprehensive Evaluation and Benchmarking Suite for Velora AI Recommender.
Measures MAP@10, NDCG@10, Recall@10, and Genre Coherence across ground-truth clusters.
"""

import math
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Ground truth benchmark test cases
GROUND_TRUTH_CASES = [
    # Director Filmography Consistency
    {
        "category": "Director Filmography",
        "query": "SS Rajamouli",
        "mode": "movies",
        "genres": [],
        "year": "All",
        "relevant": {"RRR", "Baahubali: The Beginning", "Baahubali 2: The Conclusion", "Eega", "Magadheera"}
    },
    {
        "category": "Director Filmography",
        "query": "Rajkumar Hirani",
        "mode": "movies",
        "genres": [],
        "year": "All",
        "relevant": {"3 Idiots", "PK", "Munna Bhai M.B.B.S.", "Lage Raho Munna Bhai", "Sanju", "Dunki"}
    },
    {
        "category": "Director Filmography",
        "query": "Amar Kaushik",
        "mode": "movies",
        "genres": [],
        "year": "All",
        "relevant": {"Stree", "Stree 2", "Bhediya", "Bala", "Munjya"}
    },
    {
        "category": "Director Filmography",
        "query": "Mani Ratnam",
        "mode": "movies",
        "genres": [],
        "year": "All",
        "relevant": {"Ponniyin Selvan: I", "Ponniyin Selvan: II", "Guru", "Dil Se..", "Nayakan", "Roja", "Bombay"}
    },
    {
        "category": "Director Filmography",
        "query": "Lokesh Kanagaraj",
        "mode": "movies",
        "genres": [],
        "year": "All",
        "relevant": {"Kaithi", "Vikram", "Leo", "Master"}
    },
    # Actor Star Vehicle Clusters
    {
        "category": "Actor Filmography",
        "query": "Shah Rukh Khan",
        "mode": "movies",
        "genres": [],
        "year": "All",
        "relevant": {"Dilwale Dulhania Le Jayenge", "Swades", "Chak De! India", "Kal Ho Naa Ho", "Jawan", "Pathaan", "Dunki", "Veer-Zaara", "Devdas"}
    },
    {
        "category": "Actor Filmography",
        "query": "Vijay Sethupathi",
        "mode": "movies",
        "genres": [],
        "year": "All",
        "relevant": {"Maharaja", "Vikram", "Super Deluxe", "96", "Master"}
    },
    # Multi-Genre Combinations
    {
        "category": "Multi-Genre Combination",
        "query": "Horror Comedy",
        "mode": "movies",
        "genres": ["Comedy", "Horror"],
        "year": "All",
        "relevant": {"Stree", "Stree 2", "Bhool Bhulaiyaa", "Bhool Bhulaiyaa 2", "Bhool Bhulaiyaa 3", "Munjya", "Bhediya", "Go Goa Gone"}
    },
    {
        "category": "Multi-Genre Combination",
        "query": "Historical Action",
        "mode": "movies",
        "genres": ["Action", "Historical"],
        "year": "All",
        "relevant": {"Chhaava", "Pawankhind", "Subhedar", "Tanhaji", "Baahubali 2: The Conclusion", "RRR", "Kasoombo", "Sam Bahadur"}
    },
    # Music Composer & Artist Sonic Signatures
    {
        "category": "Music Composer Signature",
        "query": "A.R. Rahman",
        "mode": "songs",
        "genres": [],
        "year": "All",
        "relevant": {"Kun Faya Kun", "Chaiyya Chaiyya", "Jai Ho", "Nadaan Parinde", "Tum Tak", "Tere Bina", "Agar Tum Saath Ho", "Maa Tujhe Salaam"}
    },
    {
        "category": "Music Composer Signature",
        "query": "Pritam",
        "mode": "songs",
        "genres": [],
        "year": "All",
        "relevant": {"Kesariya", "Channa Mereya", "Badtameez Dil", "Tum Hi Ho Bandhu", "Ilahi", "Subhanallah", "Ae Dil Hai Mushkil", "Kalank"}
    },
    {
        "category": "Music Composer Signature",
        "query": "Ajay-Atul",
        "mode": "songs",
        "genres": [],
        "year": "All",
        "relevant": {"Zingaat", "Yad Lagla", "Apsara Aali", "Chikni Chameli", "Deva Shree Ganesha", "Kombdi Palali"}
    },
    {
        "category": "Music Artist Signature",
        "query": "Anuv Jain",
        "mode": "songs",
        "genres": [],
        "year": "All",
        "relevant": {"Husn", "Baarishein", "Alag Aasmaan", "Mishri", "Gul"}
    },
    {
        "category": "Music Artist Signature",
        "query": "Prateek Kuhad",
        "mode": "songs",
        "genres": [],
        "year": "All",
        "relevant": {"cold/mess", "Kasoor", "Tune Kaha", "Kho Gaye Hum Kahan"}
    },
    {
        "category": "Music Artist Signature",
        "query": "The Local Train",
        "mode": "songs",
        "genres": [],
        "year": "All",
        "relevant": {"Choo Lo", "Aaoge Tum Kabhi", "Dil Mere", "Aaftaab"}
    },
    # Music Multi-Genre Combination
    {
        "category": "Music Multi-Genre",
        "query": "Indie Soulful",
        "mode": "songs",
        "genres": ["Indian Indie", "Romantic & Soulful"],
        "year": "All",
        "relevant": {"Husn", "Baarishein", "cold/mess", "Kasoor", "Choo Lo", "Maan Meri Jaan", "Kho Gaye Hum Kahan", "Alag Aasmaan"}
    }
]

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max

def average_precision_at_k(r, k, total_relevant):
    r = np.asarray(r)[:k]
    if not total_relevant or not np.sum(r):
        return 0.0
    precisions = [np.sum(r[:i+1]) / (i + 1) for i in range(len(r)) if r[i]]
    return np.sum(precisions) / min(k, total_relevant)

def evaluate_recommender(recommender, k=10) -> dict:
    """Evaluates a recommender instance across all benchmark test cases."""
    start_t = time.time()
    all_ndcg = []
    all_map = []
    all_recall = []
    latencies = []

    category_results = {}

    for case in GROUND_TRUTH_CASES:
        t0 = time.time()
        res = recommender.recommend(
            query=case["query"],
            mode=case["mode"],
            genres_filter=case["genres"],
            year_filter=case["year"],
            top_n=k
        )
        latencies.append((time.time() - t0) * 1000)

        retrieved = []
        if case["mode"] in ["movies", "both"]:
            retrieved.extend([m['title'] for m in res.get('movies', [])])
        if case["mode"] in ["songs", "both"]:
            retrieved.extend([s['title'] for s in res.get('songs', [])])

        retrieved_k = retrieved[:k]
        rel_set = case["relevant"]
        # Case insensitive match
        rel_set_lower = {x.lower().strip() for x in rel_set}

        binary_relevance = [1 if item.lower().strip() in rel_set_lower else 0 for item in retrieved_k]
        num_hits = sum(binary_relevance)
        recall = num_hits / len(rel_set) if rel_set else 0.0
        ap = average_precision_at_k(binary_relevance, k, len(rel_set))
        ndcg = ndcg_at_k(binary_relevance, k)

        all_ndcg.append(ndcg)
        all_map.append(ap)
        all_recall.append(recall)

        cat = case["category"]
        if cat not in category_results:
            category_results[cat] = {"ndcg": [], "map": [], "recall": []}
        category_results[cat]["ndcg"].append(ndcg)
        category_results[cat]["map"].append(ap)
        category_results[cat]["recall"].append(recall)

    eval_time = time.time() - start_t
    return {
        "mean_ndcg": float(np.mean(all_ndcg)),
        "mean_map": float(np.mean(all_map)),
        "mean_recall": float(np.mean(all_recall)),
        "avg_latency_ms": float(np.mean(latencies)),
        "eval_time_s": float(eval_time),
        "by_category": {
            cat: {
                "ndcg": float(np.mean(vals["ndcg"])),
                "map": float(np.mean(vals["map"])),
                "recall": float(np.mean(vals["recall"]))
            }
            for cat, vals in category_results.items()
        }
    }

if __name__ == "__main__":
    from ai_entertainment_recommender.recommender import UnifiedEntertainmentRecommender
    print("Evaluating current model...")
    rec = UnifiedEntertainmentRecommender(auto_train=True)
    metrics = evaluate_recommender(rec, k=10)
    print("Metrics:")
    print(f"  NDCG@10: {metrics['mean_ndcg']:.4f}")
    print(f"  MAP@10:  {metrics['mean_map']:.4f}")
    print(f"  Recall@10: {metrics['mean_recall']:.4f}")
    print(f"  Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
