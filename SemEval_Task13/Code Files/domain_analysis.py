import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy


#for all plots, change the path to plots folder on your system

print("\n=DOMAIN ANALYSIS =\n")


# Load Dataset
df = pd.read_parquet("data/train.parquet") #path to be changed to the dataset location on your system

print("Loaded samples:", len(df))
print(df.head())


#  1. LANGUAGE DOMAIN (Seen vs Unseen)


SEEN_LANGS = {"python", "java", "cpp", "c++"}
UNSEEN_LANGS = {"go", "php", "csharp", "c#", "c", "js", "javascript"}

def normalize_lang(lang):
    lang = lang.lower().replace(" ", "")
    if lang == "c++": return "cpp"
    if lang == "c#": return "csharp"
    return lang

df["language_norm"] = df["language"].apply(normalize_lang)
df["lang_domain"] = df["language_norm"].apply(
    lambda x: "seen" if x in SEEN_LANGS else "unseen"
)


# 2. CONTEXT DOMAIN DETECTION (Algorithmic / Research / Prod)


def domain_detection(code):
    code_lower = code.lower()
    
    # Algorithmic markers
    algorithmic_terms = [
        'sort', 'search', 'algorithm', 'complexity', 'graph', 'tree',
        'array', 'linked list', 'binary', 'recursion', 'dynamic programming',
        'dijkstra', 'bfs', 'dfs', 'leetcode', 'hackerrank'
    ]
    
    # Research markers
    research_terms = [
        'experiment', 'evaluation', 'metric', 'baseline', 'dataset',
        'precision', 'recall', 'f1', 'accuracy', 'result', 'methodology',
        'hypothesis', 'validation', 'paper', 'citation'
    ]
    
    # Production markers
    production_terms = [
        'error', 'exception', 'log', 'config', 'database', 'api', 'endpoint',
        'server', 'client', 'request', 'response', 'deploy', 'docker',
        'kubernetes', 'microservice', 'authentication'
    ]
    
    scores = {
        'algorithmic': sum(t in code_lower for t in algorithmic_terms),
        'research': sum(t in code_lower for t in research_terms),
        'production': sum(t in code_lower for t in production_terms)
    }
    
    if any(scores.values()):
        return max(scores, key=scores.get)
    return 'algorithmic'

df["context_domain"] = df["code"].apply(domain_detection)

print("\n--- Domain Detection Summary ---")
print(df["context_domain"].value_counts())


# 3. Domain Distribution Plots

# Language distribution
plt.figure(figsize=(7,5))
df["language_norm"].value_counts().plot(kind="bar")
plt.title("Programming Language Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/language_distribution.pdf")
plt.close()

# Context domain distribution
plt.figure(figsize=(7,5))
df["context_domain"].value_counts().plot(kind="bar", color="orange")
plt.title("Context Domain Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/context_domain_distribution.pdf")
plt.close()


# 4. Feature Extraction (Length, Tokens, Entropy)

df["code_length"] = df["code"].str.len()
df["token_count"] = df["code"].apply(lambda x: len(x.split()))

def compute_entropy(text):
    counts = Counter(text)
    total = sum(counts.values())
    probs = [c/total for c in counts.values()] if total > 0 else [1]
    return entropy(probs)

df["char_entropy"] = df["code"].apply(compute_entropy)

# Length by domain
plt.figure(figsize=(7,5))
df.boxplot(column="code_length", by="context_domain")
plt.title("Code Length by Domain")
plt.suptitle("")
plt.savefig("plots/code_length_by_domain.pdf")
plt.close()

# Token count by domain
plt.figure(figsize=(7,5))
df.boxplot(column="token_count", by="context_domain")
plt.title("Token Count by Domain")
plt.suptitle("")
plt.savefig("plots/token_count_by_domain.pdf")
plt.close()

# Entropy by domain
plt.figure(figsize=(7,5))
df.boxplot(column="char_entropy", by="context_domain")
plt.title("Entropy by Domain")
plt.suptitle("")
plt.savefig("plots/entropy_by_domain.pdf")
plt.close()

# 5. Label Distribution by Domain

plt.figure(figsize=(7,5))
df.groupby("context_domain")["label"].value_counts().unstack().plot(kind="bar")
plt.title("Label Distribution (Human vs Machine) by Domain")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/label_distribution_by_domain.pdf")
plt.close()

# 6. Correlation Heatmap

feature_cols = ["code_length", "token_count", "char_entropy"]
corr_matrix = df[feature_cols + ["label"]].corr()

plt.figure(figsize=(7,5))
plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.pdf")
plt.close()

print("\nCorrelation Matrix:")
print(corr_matrix)

# 7. OOD (Out-of-Domain) Scenario Classification

print("\n--- OOD Generalization Scenarios ---")

scenario_data = []

for _, row in df.iterrows():
    lang = row['lang_domain']
    domain = row['context_domain']
    
    if lang == "seen" and domain == "algorithmic":
        scenario = "Seen Lang + Seen Domain"
    elif lang == "unseen" and domain == "algorithmic":
        scenario = "Unseen Lang + Seen Domain"
    elif lang == "seen" and domain != "algorithmic":
        scenario = "Seen Lang + Unseen Domain"
    else:
        scenario = "Unseen Lang + Unseen Domain"
    
    scenario_data.append(scenario)

df["ood_scenario"] = scenario_data

print(df["ood_scenario"].value_counts())

plt.figure(figsize=(10,6))
df["ood_scenario"].value_counts().plot(kind="bar")
plt.title("OOD Generalization Scenario Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/ood_scenarios_distribution.pdf") 
plt.close()

# 8. Domain Shift Metrics

def calculate_domain_shift(feature):
    domain_means = df.groupby("context_domain")[feature].mean()
    return (domain_means.std() / domain_means.mean())

shift_metrics = {
    feature: calculate_domain_shift(feature)
    for feature in ['code_length', 'token_count', 'char_entropy']
}

print("\n--- Domain Shift Metrics ---")
for k, v in shift_metrics.items():
    print(f"{k}: {v:.3f}")


# 9. Cross-Domain Feature Differences

print("\n--- Cross-Domain Feature Differences ---")

pairs = [
    ("algorithmic", "research"),
    ("algorithmic", "production"),
    ("research", "production")
]

for d1, d2 in pairs:
    a = df[df["context_domain"] == d1]
    b = df[df["context_domain"] == d2]
    
    if len(a) > 0 and len(b) > 0:
        print(f"\n{d1.upper()} vs {d2.upper()}:")
        print(f"- Length diff: {abs(a['code_length'].mean() - b['code_length'].mean()):.2f}")
        print(f"- Entropy diff: {abs(a['char_entropy'].mean() - b['char_entropy'].mean()):.3f}")

# 10. Vocabulary Overlap

print("\n--- Vocabulary Overlap ---")

def get_domain_vocabulary(domain, max_features=1000):
    texts = df[df["context_domain"] == domain]["code"].tolist()
    if not texts:
        return set()

    try:
        vectorizer = CountVectorizer(max_features=max_features, token_pattern=r"\b\w+\b")
        vectorizer.fit(texts)
        return set(vectorizer.get_feature_names_out())
    except:
        return set()

domains = ["algorithmic", "research", "production"]
vocabs = {d: get_domain_vocabulary(d) for d in domains}

for i, d1 in enumerate(domains):
    for d2 in domains[i+1:]:
        overlap = len(vocabs[d1] & vocabs[d2])
        union = len(vocabs[d1] | vocabs[d2])
        overlap_ratio = overlap / union if union > 0 else 0
        print(f"{d1} ↔ {d2}: {overlap_ratio:.2%}")

# 11. Final Insights

print("\n===== CRITICAL OOD INSIGHTS =====")

print("→ Rarest scenario:", df["ood_scenario"].value_counts().idxmin())
most_imbalanced = df.groupby("context_domain")["label"].apply(
    lambda x: abs(x.value_counts(normalize=True).get("human", 0) - 0.5)
).idxmax()
print("→ Most label-imbalanced domain:", most_imbalanced)

most_sensitive = max(shift_metrics.items(), key=lambda x: x[1])[0]
print("→ Feature most affected by domain shift:", most_sensitive)

print("\nALL PLOTS SAVED IN /plots/")
print("EDA PART 8 COMPLETE.")
