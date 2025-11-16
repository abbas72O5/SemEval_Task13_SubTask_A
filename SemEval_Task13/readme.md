###### **README — SemEval Task 13 (Subtask A) Dataset Analysis**



This project provides a complete exploratory analysis of the SemEval-2026 Task 13, Subtask A dataset.



**It includes:**

Dataset overview (basic stats \& distributions)

Domain analysis (language domain, context domain, OOD scenarios, entropy, correlations, etc.)

Noise \& modality analysis (duplicates, truncation, mixed languages, template repetition)



All analyses produce high-quality PDF plots saved in the Plots/ folder.





###### **Project Structure:**



Project/

│

├── Code Files/

│     ├── Dataset\_Overview.py

│     ├── domain\_analysis.py

│     ├── noise\_modality.py

│     ├── N-gram-Analysis.py

│     ├── Dataset-Diversity.py

│

├── Plots/

│     └── (all generated graphs saved here)

│

└── README.md

**Requirements:**
---

###### 

You must download the SemEval Task 13, Subtask A dataset manually.

Install: pandas, matplotlib, scikit-learn, scipy, numpy, pyarrow, fastparquet





###### **How to Run:**



**1. Update all file paths as shown below.**



The scripts contain hard-coded paths that must be updated for your system.

Below is the complete list of paths you need to modify.

1) Dataset\_Overview.py:



-> Replace the folder path with the directory where your dataset is located.



train = pd.read\_parquet("C:/Users/ASUS/PycharmProjects/PythonProject/semeval/sem-eval-2026-task-13-subtask-a/Task\_A/train.parquet")

val   = pd.read\_parquet("C:/Users/ASUS/PycharmProjects/PythonProject/semeval/sem-eval-2026-task-13-subtask-a/Task\_A/validation.parquet")

test  = pd.read\_parquet("C:/Users/ASUS/PycharmProjects/PythonProject/semeval/sem-eval-2026-task-13-subtask-a/Task\_A/test.parquet")



-> Replace the path to point to your Plots/ directory.



plt.savefig("C:/Users/ASUS/PycharmProjects/PythonProject/semeval/Plots/Human\_vs\_Machine\_Distribution.pdf")

plt.savefig("C:/Users/ASUS/PycharmProjects/PythonProject/semeval/Plots/Language\_Distribution.pdf")





2\) domain\_analysis.py:



-> Replace "data/train.parquet" with the full path to your training dataset.



df = pd.read\_parquet("data/train.parquet")





-> Change "plots/" to the full path of your Plots directory



Every line beginning with: plt.savefig("plots/...





3\) noise\_modality.py:



-> Replace with full path to your train.parquet.



df = pd.read\_parquet("data/train.parquet")





4\) N-gram-Analysis.py:



-> Replace with full path to your train.parquet.



data = pd.read\_parquet("D:/HP/Projects/Assignment-1-Ai-Project/Data/train.parquet")



-> Change "plots/" to the full path of your Plots directory



plt.savefig("D:/HP/Projects/Assignment-1-Ai-Project/Plots/N-gram-analysis.pdf")





5\) Dataset-Diversity.py:



-> Replace with full path to your train.parquet.



data = pd.read\_parquet("D:/HP/Projects/Assignment-1-Ai-Project/Data/train.parquet")



-> Change "plots/" to the full path of your Plots directory



plt.savefig("D:/HP/Projects/Assignment-1-Ai-Project/Plots/diversity.pdf")





**2. Run each analysis independently:**

&nbsp;  python Dataset\_Overview.py

&nbsp;  python domain\_analysis.py

&nbsp;  python noise\_modality.py

&nbsp;  python N-gram-Analysis.py

&nbsp;  python Dataset-Diversity.py



**3. All plots will be saved in the Plots/ folder.**









