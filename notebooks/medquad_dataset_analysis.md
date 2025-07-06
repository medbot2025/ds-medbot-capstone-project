# 🩺 MedQuAD Dataset Analysis

This is an analysis of the `medquad.csv` dataset, which contains medical questions and answers.

---

## 📊 Summary of Steps Performed

### 1. 📥 Data Loading and Initial Exploration

- The `medquad.csv` dataset is loaded into a Pandas DataFrame named `df_med`.
- Basic information is displayed using:
  - `df_med.head()`
  - `df_med.info()`
  - `df_med.describe()`
- The shape of the DataFrame is checked to show number of rows and columns.
- Duplicated rows are counted.
- Unique values in the `focus_area` column are identified.

---

### 2. 🔍 Missing Data Handling

- Missing values for each column are calculated.
- A copy of the DataFrame is created (`df`).
- Rows with missing values in the `answer` and `focus_area` columns are dropped to create `df_cleaned`.
- Value counts of `focus_area` are printed to confirm cleanup.

---

### 3. 🛠 Feature Engineering

- Two new columns are added:
  - `question_len`: Word count in the `question` column.
  - `answer_len`: Word count in the `answer` column.
- Descriptive statistics for both new columns are printed.

---

### 4. 📈 Exploratory Data Analysis (EDA)

- Histograms are created for:
  - `question_len`
  - `answer_len`
- A bar plot shows the **Top 10 most frequent** `focus_area` categories.
- `source` column value counts are displayed with a corresponding bar plot.

---

### 5. 🧹 Data Quality Checks and Further Cleaning

- Very short entries (questions/answers with < 3 words) are removed.
- Missing values are re-checked after cleanup.
- A custom function `alpha_ratio` is defined to calculate the ratio of alphabetic characters in a string:

```python
def alpha_ratio(text):
    return sum(c.isalpha() for c in text) / len(text)
```

- Entries where `alpha_ratio` in `answer` is below 0.5 are considered noisy and removed.
- A boxplot is generated to visualize the distribution of `answer_len` after cleaning.
