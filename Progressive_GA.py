"""
Citation for this work: 
Deepika Verma, Daison Darlan, Rammohan Mallipeddi. ''Progressive Vocabulary Learning via Pareto-Optimal Clustering''.
International Conference on ICT Convergence, South Korea (2025).

@author: Deepika Verma
"""


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from readability import Readability
from OptiMDS_word2vec import obtain_text, text  # as original_text  
from jmetal.util.solution import read_solutions
from math import sqrt

# ---------- 1. Load CSV + Regenerate Pareto Solutions ----------
pareto_csv = "nsga2_word2vec_results.csv"
solutions_file = "VAR.tsv"  # generated using jMetal 'print_variables_to_file'

df = pd.read_csv(pareto_csv)
solution_vectors = read_solutions(filename=solutions_file)

texts = []
fkgls = []
word_replacements = []

print("\nGenerating text + FKGL from Pareto solutions...")


text_words = original_text.split()

for i, sol in enumerate(solution_vectors):
    sol_values = sol.variables if hasattr(sol, 'variables') else sol

    if len(sol_values) != len(text_words):
        print(f"⚠️ Skipping solution {i}: length mismatch ({len(sol_values)} vs {len(text_words)})")
        continue

    try:
        txt = obtain_text(sol, original_text)
        r = Readability(txt)
        fkgl = r.fkgl().score
        replaced = sum(1 for x in sol_values if x >= 1)

        texts.append(txt)
        fkgls.append(fkgl)
        word_replacements.append(replaced)

    except Exception as e:
        print(f"⚠️ Error processing solution {i}: {e}")

'''
for sol in solution_vectors:
    txt = obtain_text(sol, original_text)
    texts.append(txt)
    
    try:
        r = Readability(txt)
        fkgl = r.fkgl().score
    except:
        fkgl = 12.0  # fallback if FKGL fails
    
    word_count = sum([1 for x in sol.variables if x >= 1])
    fkgls.append(fkgl)
    word_replacements.append(word_count)
'''


data = pd.DataFrame({
    "Text": texts,
    "FKGL": fkgls,
    "Words_Replaced": word_replacements
})

# ---------- 2. Normalize and Cluster into 3 Levels ----------
print("\nClustering into Beginner, Intermediate, Advanced...")

features = data[["FKGL", "Words_Replaced"]]
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
data["Cluster"] = kmeans.fit_predict(features_scaled)

# Label clusters by ascending FKGL
cluster_order = data.groupby("Cluster")["FKGL"].mean().sort_values().index.tolist()
cluster_names = {cluster_order[0]: "Beginner", cluster_order[1]: "Intermediate", cluster_order[2]: "Advanced"}
data["Level"] = data["Cluster"].map(cluster_names)

# ---------- 3. TOPSIS: Pick Best Solution from Each Cluster ----------
def topsis_cluster(df_sub):
    # Normalize
    sub_scaled = scaler.fit_transform(df_sub[["FKGL", "Words_Replaced"]])
    ideal_best = np.min(sub_scaled, axis=0)
    ideal_worst = np.max(sub_scaled, axis=0)

    distances_best = np.linalg.norm(sub_scaled - ideal_best, axis=1)
    distances_worst = np.linalg.norm(sub_scaled - ideal_worst, axis=1)
    
    scores = distances_worst / (distances_best + distances_worst + 1e-6)
    return df_sub.iloc[np.argmax(scores)]  # Highest score is closest to ideal

print("\nSelecting representative for each level using TOPSIS...\n")

final_texts = []
for level in ["Beginner", "Intermediate", "Advanced"]:
    sub_df = data[data["Level"] == level].copy()
    best_row = topsis_cluster(sub_df)
    final_texts.append((level, best_row["Text"], best_row["FKGL"], best_row["Words_Replaced"]))

# ---------- 4. Display the Results ----------
print("="*60)
print("📘 READABILITY-ADAPTIVE ENGLISH LEARNING OUTPUT")
print("="*60)

for level, text, fkgl, replaced in final_texts:
    print(f"\n📗 {level} Version:")
    print(f"FKGL Score: {fkgl:.2f} | Words Replaced: {replaced}")
    print("-" * 60)
    print(text.strip())
    print("-" * 60)

# Show original too (unmodified)
from OptiMDS_word2vec import text as original_text
r_orig = Readability(original_text).fkgl().score
print(f"\n📕 Original Text (FKGL = {r_orig:.2f}):\n")
print(original_text.strip())
print("="*60)
