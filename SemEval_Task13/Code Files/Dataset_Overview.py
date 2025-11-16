

#prints basic data details and plots label distribution and language distribution

import pandas as pd
import matplotlib.pyplot as plt

# Load data
train = pd.read_parquet("C:/Users/ASUS/PycharmProjects/PythonProject/semeval/sem-eval-2026-task-13-subtask-a/Task_A/train.parquet")
val   = pd.read_parquet("C:/Users/ASUS/PycharmProjects/PythonProject/semeval/sem-eval-2026-task-13-subtask-a/Task_A/validation.parquet")
test  = pd.read_parquet("C:/Users/ASUS/PycharmProjects/PythonProject/semeval/sem-eval-2026-task-13-subtask-a/Task_A/test.parquet")

# Print basic info
print("Train columns:", train.columns)
print("Validation columns:", val.columns)
print("Test columns:", test.columns)

print("Train samples:", len(train))
print("Validation samples:", len(val))
print("Test samples:", len(test))

# ---------------------
# Label distribution plots
# ---------------------
labels_train = train['label'].value_counts().sort_index()
labels_val = val['label'].value_counts().sort_index()

labels = ['Human', 'Machine']
x = range(len(labels))
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(6,5))
ax.bar([i - width/2 for i in x], labels_train, width, label='Train', color='steelblue')
ax.bar([i + width/2 for i in x], labels_val, width, label='Validation', color='orange')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Count')
ax.set_title('Human vs Machine Code: Train vs Validation')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig("C:/Users/ASUS/PycharmProjects/PythonProject/semeval/Plots/Human_vs_Machine_Distribution.pdf")
plt.show()
plt.close()

# ---------------------
# Language distribution plots
# ---------------------
languages_train = train['language'].value_counts()
languages_val = val['language'].value_counts()

# Align languages so plots are comparable
all_languages = sorted(set(languages_train.index) | set(languages_val.index))
train_counts = [languages_train.get(lang, 0) for lang in all_languages]
val_counts = [languages_val.get(lang, 0) for lang in all_languages]

x = range(len(all_languages))
width = 0.35  # width of bars

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar([i - width/2 for i in x], train_counts, width, label='Train', color='steelblue')
ax.bar([i + width/2 for i in x], val_counts, width, label='Validation', color='orange')

ax.set_xticks(x)
ax.set_xticklabels(all_languages)
ax.set_ylabel('Count')
ax.set_title('Language Distribution: Train vs Validation')
ax.legend()

plt.tight_layout()
plt.savefig("C:/Users/ASUS/PycharmProjects/PythonProject/semeval/Plots/Language_Distribution.pdf")
plt.show()
plt.close()
