import pandas as pd
import numpy as np

# Load the cleaned data
print("Loading cleaned data...")
df = pd.read_csv('data/IMDb_movies_cleaned.csv')
print(f"Dataset shape: {df.shape}")
print(f"Total rows: {len(df):,}\n")

print("="*80)
print("UNIQUE VALUES ANALYSIS FOR EACH COLUMN")
print("="*80)

# Separate columns by type
id_cols = ['imdb_title_id']
target_col = ['avg_vote']
numeric_cols = ['year', 'duration', 'votes']
text_cols = ['title', 'original_title', 'description']
categorical_cols = ['genre', 'country', 'language', 'director', 'writer',
                   'production_company', 'actors']
date_cols = ['date_published']

all_cols = df.columns.tolist()

# Function to analyze column
def analyze_column(col):
    unique_count = df[col].nunique()
    total_count = len(df)
    uniqueness_ratio = unique_count / total_count * 100

    print(f"\n{col}")
    print(f"  {'─' * 70}")
    print(f"  Unique values: {unique_count:,}")
    print(f"  Uniqueness ratio: {uniqueness_ratio:.2f}%")
    print(f"  Data type: {df[col].dtype}")

    # Show sample values
    if unique_count <= 10:
        print(f"  All unique values:")
        for val in sorted(df[col].unique()):
            count = (df[col] == val).sum()
            print(f"    - {val}: {count:,} ({count/total_count*100:.2f}%)")
    else:
        print(f"  Top 10 most common values:")
        top_values = df[col].value_counts().head(10)
        for val, count in top_values.items():
            val_str = str(val)[:50] + "..." if len(str(val)) > 50 else str(val)
            print(f"    - {val_str}: {count:,} ({count/total_count*100:.2f}%)")

    # Additional stats for numeric columns
    if df[col].dtype in ['int64', 'float64'] and col not in ['year']:
        print(f"  Range: [{df[col].min()}, {df[col].max()}]")
        print(f"  Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")

# Analyze by category
print("\n" + "="*80)
print("📊 NUMERIC COLUMNS")
print("="*80)
for col in numeric_cols:
    if col in df.columns:
        analyze_column(col)

print("\n\n" + "="*80)
print("🏷️  CATEGORICAL COLUMNS (Simple)")
print("="*80)
simple_cats = ['country', 'language', 'production_company']
for col in simple_cats:
    if col in df.columns:
        analyze_column(col)

print("\n\n" + "="*80)
print("🎬 PEOPLE COLUMNS (Complex - Multiple values per row)")
print("="*80)
people_cols = ['director', 'writer', 'actors']
for col in people_cols:
    if col in df.columns:
        analyze_column(col)
        # Count individual people (split by comma)
        all_people = []
        for val in df[col].dropna():
            all_people.extend([p.strip() for p in str(val).split(',')])
        unique_people = len(set(all_people))
        print(f"  → Total unique individual {col}: {unique_people:,}")

print("\n\n" + "="*80)
print("🎭 GENRE COLUMN (Multi-label)")
print("="*80)
if 'genre' in df.columns:
    analyze_column('genre')
    # Count individual genres
    all_genres = []
    for val in df['genre'].dropna():
        all_genres.extend([g.strip() for g in str(val).split(',')])
    unique_genres = sorted(set(all_genres))
    print(f"  → Total unique individual genres: {len(unique_genres)}")
    print(f"  → Individual genres: {', '.join(unique_genres[:20])}")
    if len(unique_genres) > 20:
        print(f"    ... and {len(unique_genres) - 20} more")

print("\n\n" + "="*80)
print("📝 TEXT COLUMNS (High cardinality)")
print("="*80)
for col in text_cols:
    if col in df.columns:
        analyze_column(col)
        # Show average length
        avg_len = df[col].astype(str).str.len().mean()
        print(f"  Average text length: {avg_len:.1f} characters")

print("\n\n" + "="*80)
print("🎯 TARGET COLUMN")
print("="*80)
if 'avg_vote' in df.columns:
    analyze_column('avg_vote')

print("\n\n" + "="*80)
print("📋 SUMMARY")
print("="*80)
print("\nComplexity Classification:")
print("  LOW complexity (< 100 unique):    ", end="")
low_complex = [col for col in df.columns if df[col].nunique() < 100]
print(f"{len(low_complex)} columns")

print("  MEDIUM complexity (100-1000):     ", end="")
med_complex = [col for col in df.columns if 100 <= df[col].nunique() < 1000]
print(f"{len(med_complex)} columns")

print("  HIGH complexity (1000-10000):     ", end="")
high_complex = [col for col in df.columns if 1000 <= df[col].nunique() < 10000]
print(f"{len(high_complex)} columns")

print("  VERY HIGH complexity (>10000):    ", end="")
very_high_complex = [col for col in df.columns if df[col].nunique() >= 10000]
print(f"{len(very_high_complex)} columns")

print("\n" + "="*80)
