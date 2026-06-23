"""
EDA - Log Data Exploration
Run with: python notebooks/01_exploratory_analysis.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_generator import LogDataGenerator
from src.preprocessing import LogPreprocessor

sns.set_palette("muted")
plt.rcParams.update({'figure.figsize': (10, 5), 'font.size': 11})

sep = "=" * 78
print(sep)
print("EXPLORATORY DATA ANALYSIS — SYNTHETIC LOGS")
print(sep)

# ----- A. Generate raw data -----
print("\n[A] GENERATING DATASET...")
gen = LogDataGenerator()
raw = gen.generate_dataset(n_samples=1000)
print(f"  Rows: {len(raw)} | Columns: {list(raw.columns)}")
print(f"  Preview:\n{raw.head()}")

# ----- B. Preprocessing impact -----
print(f"\n[B] CLEANING & TRANSFORMATION...")
proc = LogPreprocessor()
raw['cleaned'] = raw['log'].apply(proc.clean_log)
raw['len_raw'] = raw['log'].str.len()
raw['len_clean'] = raw['cleaned'].str.len()

print(f"  Avg length before: {raw['len_raw'].mean():.0f} chars")
print(f"  Avg length after:  {raw['len_clean'].mean():.0f} chars")
shrink = (1 - raw['len_clean'].mean() / raw['len_raw'].mean()) * 100
print(f"  Reduction: {shrink:.1f}%")
print("  Samples:")
for i in range(3):
    print(f"  --- #{i} ---")
    print(f"  Before: {raw['log'].iloc[i]}")
    print(f"  After:  {raw['cleaned'].iloc[i]}")

# ----- C. Label distribution -----
print(f"\n[C] ROOT CAUSE ANALYSIS...")
dist = raw['label'].value_counts()
print(dist.to_string())

fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(13, 5))
colors = sns.color_palette("Set2", len(dist))

ax_bar.bar(dist.index, dist.values, color=colors, edgecolor='gray')
ax_bar.set_title("Root cause frequency", weight='bold')
ax_bar.set_ylabel("Count")
ax_bar.tick_params(axis='x', rotation=30)

wedges, texts, autotexts = ax_pie.pie(
    dist.values, labels=dist.index, autopct='%1.1f%%',
    colors=colors, startangle=120, wedgeprops={'edgecolor': 'white'}
)
ax_pie.set_title("Root cause proportion", weight='bold')

plt.tight_layout()
plt.savefig('results/01_label_distribution.png', dpi=150, bbox_inches='tight')
print("  Saved → results/01_label_distribution.png")

# ----- D. Length & token statistics -----
print(f"\n[D] TEXT STATISTICS...")
char_lens = raw['log'].str.len()
tok_lens = raw['log'].str.split().str.len()
print(f"  Avg chars: {char_lens.mean():.0f} | Max chars: {char_lens.max():.0f}")
print(f"  Avg tokens: {tok_lens.mean():.0f}")

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
ax_l.hist(char_lens, bins=35, color='steelblue', edgecolor='white')
ax_l.set_title("Log character length", weight='bold')
ax_l.set_xlabel("Characters"); ax_l.set_ylabel("Frequency")

ax_r.hist(tok_lens, bins=35, color='seagreen', edgecolor='white')
ax_r.set_title("Log token length", weight='bold')
ax_r.set_xlabel("Tokens"); ax_r.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig('results/02_log_statistics.png', dpi=150, bbox_inches='tight')
print("  Saved → results/02_log_statistics.png")

# ----- E. Persist -----
print(f"\n[E] SAVING DATASET...")
raw.to_csv('data/raw/synthetic_logs.csv', index=False)
print("  Saved → data/raw/synthetic_logs.csv")

# ----- F. Summary -----
print(f"\n{sep}")
print("SUMMARY")
print(sep)
print(f"  Total logs:     {len(raw)}")
print(f"  Root causes:    {raw['label'].nunique()}")
print(f"  Avg length:     {char_lens.mean():.0f} chars")
print(f"  Balanced data:  {dist.std() < 50}")
print(sep)
