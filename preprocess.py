import pandas as pd
import numpy as np

# Load the data
print("Loading data...")
df = pd.read_csv('data/IMDb movies.csv')
print(f"Original shape: {df.shape}")
print(f"Original rows: {len(df):,}")

# Step 1: Drop high-missing columns (>60%)
print("\n" + "="*60)
print("STEP 1: Dropping high-missing columns (>60%)")
print("="*60)
cols_to_drop = ['worlwide_gross_income', 'budget', 'usa_gross_income', 'metascore']
print(f"Dropping columns: {cols_to_drop}")
df = df.drop(columns=cols_to_drop)
print(f"Shape after dropping columns: {df.shape}")

# Step 2: Drop rows with missing critical features
print("\n" + "="*60)
print("STEP 2: Dropping rows with missing critical features")
print("="*60)
critical_cols = ['director', 'actors', 'writer', 'description']
print(f"Critical columns: {critical_cols}")

for col in critical_cols:
    before = len(df)
    df = df.dropna(subset=[col])
    after = len(df)
    print(f"  {col:20s}: Dropped {before - after:,} rows")

print(f"\nShape after dropping critical missing rows: {df.shape}")
print(f"Remaining rows: {len(df):,}")

# Step 3: Simple imputation for remaining columns
print("\n" + "="*60)
print("STEP 3: Imputing remaining missing values")
print("="*60)

# Impute categorical columns with 'Unknown'
categorical_cols = ['country', 'language', 'production_company']
for col in categorical_cols:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        df[col] = df[col].fillna('Unknown')
        print(f"  {col:20s}: Imputed {missing_count:,} values with 'Unknown'")

# Impute numeric review columns with median
numeric_cols = ['reviews_from_users', 'reviews_from_critics']
for col in numeric_cols:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  {col:20s}: Imputed {missing_count:,} values with median ({median_val:.1f})")

# Final validation
print("\n" + "="*60)
print("FINAL DATA QUALITY CHECK")
print("="*60)
missing_after = df.isnull().sum().sum()
print(f"Total missing values remaining: {missing_after}")
print(f"Final shape: {df.shape}")
print(f"Final rows: {len(df):,}")
print(f"Rows retained: {len(df) / 85855 * 100:.2f}%")

# Display missing values summary
print("\nMissing values per column:")
missing_summary = df.isnull().sum()
if missing_summary.sum() == 0:
    print("  ✓ No missing values!")
else:
    for col, count in missing_summary[missing_summary > 0].items():
        print(f"  {col}: {count}")

# Save cleaned data
output_file = 'data/IMDb_movies_cleaned.csv'
df.to_csv(output_file, index=False)
print(f"\n✓ Cleaned data saved to: {output_file}")

# Display sample statistics
print("\n" + "="*60)
print("TARGET VARIABLE STATISTICS (avg_vote)")
print("="*60)
print(df['avg_vote'].describe())
print(f"\nColumns in cleaned dataset: {list(df.columns)}")
