import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_parquet("D:/HP/Projects/Assignment-1-Ai-Project/Data/train.parquet")

human = data[data['label']==0]['code']
machine = data[data['label']==1]['code']

# Count unigrams + bigrams
vectorizer = CountVectorizer(ngram_range=(1,2), max_features=1000)
human_matrix = vectorizer.fit_transform(human)
machine_matrix = vectorizer.transform(machine)

vocab = vectorizer.get_feature_names_out()
human_counts = human_matrix.sum(axis=0).A1
machine_counts = machine_matrix.sum(axis=0).A1

df = pd.DataFrame({
    'ngram': vocab,
    'human': human_counts,
    'machine': machine_counts
})

df['diff'] = df['machine'] - df['human']

# top 10 n-grams more frequent in machine code
top_machine = df.sort_values('diff', ascending=False).head(10)

# Plot 
plt.figure(figsize=(12,6))
bar_width = 0.4
x = range(len(top_machine))

plt.bar([i - bar_width/2 for i in x], top_machine['human'], width=bar_width, label='Human')
plt.bar([i + bar_width/2 for i in x], top_machine['machine'], width=bar_width, label='Machine')

plt.xticks(x, top_machine['ngram'], rotation=45, ha='right')
plt.ylabel("Count")
plt.title("Top 10 n-grams More Frequent in Machine-Generated Code")
plt.legend()
plt.tight_layout()

plt.savefig("D:/HP/Projects/Assignment-1-Ai-Project/Plots/N-gram-analysis.pdf")
plt.show()
plt.close()
