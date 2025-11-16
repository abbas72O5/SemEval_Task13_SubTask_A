import pandas as pd
import re
from hashlib import md5


# Load Dataset
df = pd.read_parquet("data/train.parquet") #path to be changed to the dataset location on your system

print("=== EDA 7: Noise & Modality Issues ===")
print("Total samples:", len(df))


# EXACT DUPLICATES CHECK

print("\n[A] Checking Exact Duplicate Code Snippets...")

df["hash"] = df["code"].apply(lambda x: md5(x.strip().encode()).hexdigest())
duplicate_count = df["hash"].duplicated().sum()

# Check if duplicates cross human/machine boundaries
dup_info = df[df["hash"].duplicated(keep=False)].groupby("hash")["label"].nunique()
cross_label_duplicates = (dup_info > 1).sum()

print("Exact duplicates found:", duplicate_count)
print("Duplicates with conflicting labels:", cross_label_duplicates)


# TRUNCATED / INCOMPLETE CODE CHECK

print("\n[B] Checking for Incomplete / Truncated Code Snippets...")

truncation_patterns = [
    r"(def$|class$|public$|void$|if$|for$|while$|try$)",  # Mid-line cuts
    r"\.\.\.\s*$",  # Explicit ellipsis
    r"#\s*TODO|#\s*FIXME",  # Incomplete markers
    r"//\s*TODO|//\s*FIXME",
    r"\{?\s*$"  # Ends with open brace
]

df["truncation_flags"] = df["code"].apply(
    lambda x: sum(1 for pattern in truncation_patterns if re.search(pattern, str(x), re.MULTILINE))
)
truncated_count = (df["truncation_flags"] > 0).sum()

print("Truncated/incomplete code snippets:", truncated_count)
print("Truncation rate by label:")
print(df.groupby("label")["truncation_flags"].mean())


# MIXED LANGUAGE CHECK

print("\n[C] Checking for Mixed-Language Artifacts...")

language_patterns = {
    "python": r"def |import |print\(",
    "cpp": r"#include|std::|cout|cin",
    "java": r"class |public |System\.out\.println"
}

def detect_languages(text):
    detected = []
    for lang, pat in language_patterns.items():
        if re.search(pat, text):
            detected.append(lang)
    return ",".join(detected)

df["detected_langs"] = df["code"].apply(detect_languages)

# Mixed = more than 1 language detected
mixed_langs = df[df["detected_langs"].str.contains(",")]
mixed_count = len(mixed_langs)

print("Mixed-language snippets:", mixed_count)


# MACHINE TEMPLATE REPETITION

print("\n[D] Checking for Common Machine-Generated Template Patterns...")

template_patterns = {
    "Solution class": r"class Solution\s*\{",
    "Generic function names": r"def (solve|solution|answer|main)\(",
    "Standard main methods": r"(public static void main|int main\()",
    "Excessive comments": r"#.*(generated|auto|AI|chat)",
    "Overly generic vars": r"(var|val|temp|result|data)\s*=",
}

for name, pattern in template_patterns.items():
    count = df["code"].str.contains(pattern, re.IGNORECASE).sum()
    print(f"'{name}': {count} samples")


# EMPTY/NEAR-EMPTY CODE CHECK

print("\n[E] Checking Empty/Short Code...")

df["code_length"] = df["code"].str.len()
empty_count = (df["code_length"] <= 10).sum()
short_count = ((df["code_length"] > 10) & (df["code_length"] <= 50)).sum()

print(f"Empty code (<=10 chars): {empty_count}")
print(f"Very short code (10-50 chars): {short_count}")


# REPETITIVE PATTERNS CHECK

print("\n[F] Checking Repetitive Structures...")

def repetitive_score(text):
    lines = str(text).split('\n')
    if len(lines) <= 3: return 0
    
    # Check for many similar lines (machine repetition)
    line_hashes = [md5(line.strip().encode()).hexdigest() for line in lines]
    unique_lines = len(set(line_hashes))
    return 1 - (unique_lines / len(lines))

df["repetition_score"] = df["code"].apply(repetitive_score)
high_repetition = (df["repetition_score"] > 0.5).sum()

print(f"High repetition samples: {high_repetition}")


# ----------------------------------------
# Summary
# ----------------------------------------
print("\n===== SUMMARY =====")
print("Total duplicates:", duplicate_count)
print("Duplicates with conflicting labels:", cross_label_duplicates)
print("Incomplete code:", truncated_count)
print("Mixed-language files:", mixed_count)
print("Empty/very short code:", empty_count + short_count)
print("High repetition samples:", high_repetition)
print("====================")

if cross_label_duplicates > 0:
    print(" CRITICAL: Duplicate code with different labels found!")
    print("   This indicates potential labeling errors in the dataset.")