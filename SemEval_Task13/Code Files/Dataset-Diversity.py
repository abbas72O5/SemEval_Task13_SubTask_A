import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# data
data = pd.read_parquet("D:/HP/Projects/Assignment-1-Ai-Project/Data/train.parquet")

human = data[data['label']==0]['code']
machine = data[data['label']==1]['code']

# Vocabulary Size
vectorizer_human = CountVectorizer(token_pattern=r'\b\w+\b')
vectorizer_machine = CountVectorizer(token_pattern=r'\b\w+\b')

human_matrix = vectorizer_human.fit_transform(human)
machine_matrix = vectorizer_machine.fit_transform(machine)

human_vocab = vectorizer_human.get_feature_names_out()
machine_vocab = vectorizer_machine.get_feature_names_out()

human_vocab_size = len(human_vocab)
machine_vocab_size = len(machine_vocab)

print("Human vocabulary size:", human_vocab_size)
print("Machine vocabulary size:", machine_vocab_size)

# Type-Token Ratio
human_total_tokens = human_matrix.sum()
machine_total_tokens = machine_matrix.sum()

human_ttr = human_vocab_size / human_total_tokens
machine_ttr = machine_vocab_size / machine_total_tokens

print("Human TTR:", human_ttr)
print("Machine TTR:", machine_ttr)

# plot 
classes = ['Human', 'Machine']
vocab_sizes = [human_vocab_size, machine_vocab_size]

x = [0, 0.45]
bar_width = 0.35

plt.figure(figsize=(7,5))

plt.bar(x, vocab_sizes, width=bar_width, color=['skyblue'], label='Vocabulary Size')

plt.xticks(x, classes)
plt.ylabel("Vocabulary Size")
plt.title("Vocabulary Size Comparison: Human vs Machine Code")
plt.legend()
plt.tight_layout()

plt.savefig("D:/HP/Projects/Assignment-1-Ai-Project/Plots/diversity.pdf")
plt.show()
plt.close()
