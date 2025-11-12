import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎬 IMDb FEATURE ENGINEERING PIPELINE")
print("="*80)

# Load cleaned data
print("\n[1/9] Loading cleaned data...")
df = pd.read_csv('data/IMDb_movies_cleaned.csv')
print(f"   Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# ============================================================================
# STEP 1: Drop unnecessary columns
# ============================================================================
print("\n[2/9] Dropping unnecessary columns...")
cols_to_drop = ['imdb_title_id', 'date_published', 'reviews_from_users',
                'reviews_from_critics', 'original_title']
df = df.drop(columns=cols_to_drop)
print(f"   Dropped: {cols_to_drop}")
print(f"   Remaining columns: {df.shape[1]}")

# ============================================================================
# STEP 2: Train/Test Split (CRITICAL: Prevent data leakage)
# ============================================================================
print("\n[3/9] Splitting data (80/20) to prevent data leakage...")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"   Train set: {len(train_df):,} rows")
print(f"   Test set: {len(test_df):,} rows")

# ============================================================================
# STEP 3: Log-transform votes
# ============================================================================
print("\n[4/9] Log-transforming 'votes' column...")
train_df['votes_log'] = np.log1p(train_df['votes'])
test_df['votes_log'] = np.log1p(test_df['votes'])
print(f"   Original votes - Mean: {train_df['votes'].mean():.2f}, Median: {train_df['votes'].median():.2f}")
print(f"   Log votes - Mean: {train_df['votes_log'].mean():.2f}, Median: {train_df['votes_log'].median():.2f}")

# ============================================================================
# STEP 4: Genre Multi-hot Encoding
# ============================================================================
print("\n[5/9] Multi-hot encoding genres...")
# Split genres and create multi-hot encoding
train_genres = train_df['genre'].str.split(', ').tolist()
test_genres = test_df['genre'].str.split(', ').tolist()

mlb = MultiLabelBinarizer()
train_genre_encoded = pd.DataFrame(
    mlb.fit_transform(train_genres),
    columns=['genre_' + g for g in mlb.classes_],
    index=train_df.index
)
test_genre_encoded = pd.DataFrame(
    mlb.transform(test_genres),
    columns=['genre_' + g for g in mlb.classes_],
    index=test_df.index
)
print(f"   Created {len(mlb.classes_)} genre features: {list(mlb.classes_)[:10]}...")

# ============================================================================
# STEP 5: Target Encoding for Categorical Features
# ============================================================================
print("\n[6/9] Target encoding categorical features (country, language, production_company)...")

def target_encode(train_df, test_df, column, target='avg_vote', min_samples=10):
    """Target encode with smoothing to avoid overfitting"""
    # Calculate global mean
    global_mean = train_df[target].mean()

    # Calculate mean and count per category
    agg = train_df.groupby(column)[target].agg(['mean', 'count'])

    # Smooth with global mean (Bayesian averaging)
    smoothing = 1.0 / (1.0 + np.exp(-(agg['count'] - min_samples) / min_samples))
    agg['smoothed'] = global_mean * (1 - smoothing) + agg['mean'] * smoothing

    # Map to train and test
    train_encoded = train_df[column].map(agg['smoothed']).fillna(global_mean)
    test_encoded = test_df[column].map(agg['smoothed']).fillna(global_mean)

    return train_encoded, test_encoded, agg

# Target encode country
train_df['country_encoded'], test_df['country_encoded'], country_map = \
    target_encode(train_df, test_df, 'country')
print(f"   ✓ Country encoded (top: {train_df['country'].value_counts().head(3).index.tolist()})")

# Target encode language
train_df['language_encoded'], test_df['language_encoded'], lang_map = \
    target_encode(train_df, test_df, 'language')
print(f"   ✓ Language encoded (top: {train_df['language'].value_counts().head(3).index.tolist()})")

# Target encode production company
train_df['production_company_encoded'], test_df['production_company_encoded'], prod_map = \
    target_encode(train_df, test_df, 'production_company')
print(f"   ✓ Production company encoded")

# ============================================================================
# STEP 6: Historical Performance Ratings (Director, Writer, Actors)
# ============================================================================
print("\n[7/9] Calculating historical performance ratings...")

def calculate_person_rating(train_df, test_df, column, target='avg_vote'):
    """Calculate mean rating per person from training data only"""
    global_mean = train_df[target].mean()

    # Build person -> rating mapping from training data
    person_ratings = {}

    for idx, row in train_df.iterrows():
        people = [p.strip() for p in str(row[column]).split(',')]
        rating = row[target]

        for person in people:
            if person not in person_ratings:
                person_ratings[person] = []
            person_ratings[person].append(rating)

    # Calculate mean rating per person
    person_mean = {person: np.mean(ratings) for person, ratings in person_ratings.items()}
    person_counts = {person: len(ratings) for person, ratings in person_ratings.items()}

    # Encode train and test
    def encode_row(row):
        people = [p.strip() for p in str(row[column]).split(',')]
        ratings = [person_mean.get(p, global_mean) for p in people]
        counts = [person_counts.get(p, 0) for p in people]
        return np.mean(ratings) if ratings else global_mean, np.mean(counts) if counts else 0

    train_ratings, train_counts = zip(*train_df.apply(encode_row, axis=1))
    test_ratings, test_counts = zip(*test_df.apply(encode_row, axis=1))

    return list(train_ratings), list(test_ratings), list(train_counts), list(test_counts)

# Director ratings
print("   Processing directors...")
train_df['director_rating'], test_df['director_rating'], \
train_df['director_count'], test_df['director_count'] = \
    calculate_person_rating(train_df, test_df, 'director')

# Writer ratings
print("   Processing writers...")
train_df['writer_rating'], test_df['writer_rating'], \
train_df['writer_count'], test_df['writer_count'] = \
    calculate_person_rating(train_df, test_df, 'writer')

# Actor ratings (top 3 actors)
print("   Processing actors...")
train_df['actor_rating'], test_df['actor_rating'], \
train_df['actor_count'], test_df['actor_count'] = \
    calculate_person_rating(train_df, test_df, 'actors')

print(f"   ✓ Director rating - Mean: {train_df['director_rating'].mean():.3f}")
print(f"   ✓ Writer rating - Mean: {train_df['writer_rating'].mean():.3f}")
print(f"   ✓ Actor rating - Mean: {train_df['actor_rating'].mean():.3f}")

# ============================================================================
# STEP 7: Generate Description Embeddings (GPU-accelerated)
# ============================================================================
print("\n[8/9] Generating description embeddings (using GPU)...")
print("   Loading sentence-transformer model: all-mpnet-base-v2 (768-dim)...")

# Load model with GPU support
model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
print("   ✓ Model loaded on GPU")

# Generate embeddings
print("   Encoding training descriptions...")
train_descriptions = train_df['description'].fillna('').tolist()
train_embeddings = model.encode(train_descriptions, show_progress_bar=True, batch_size=64)

print("   Encoding test descriptions...")
test_descriptions = test_df['description'].fillna('').tolist()
test_embeddings = model.encode(test_descriptions, show_progress_bar=True, batch_size=64)

# Convert to DataFrame
train_embed_df = pd.DataFrame(
    train_embeddings,
    columns=[f'desc_emb_{i}' for i in range(train_embeddings.shape[1])],
    index=train_df.index
)
test_embed_df = pd.DataFrame(
    test_embeddings,
    columns=[f'desc_emb_{i}' for i in range(test_embeddings.shape[1])],
    index=test_df.index
)
print(f"   ✓ Generated {train_embeddings.shape[1]}-dimensional embeddings")

# ============================================================================
# STEP 8: Combine All Features
# ============================================================================
print("\n[9/9] Combining all features...")

# Select final features
feature_cols = ['year', 'duration', 'votes_log',
                'country_encoded', 'language_encoded', 'production_company_encoded',
                'director_rating', 'director_count',
                'writer_rating', 'writer_count',
                'actor_rating', 'actor_count']

train_features = pd.concat([
    train_df[feature_cols].reset_index(drop=True),
    train_genre_encoded.reset_index(drop=True),
    train_embed_df.reset_index(drop=True)
], axis=1)

test_features = pd.concat([
    test_df[feature_cols].reset_index(drop=True),
    test_genre_encoded.reset_index(drop=True),
    test_embed_df.reset_index(drop=True)
], axis=1)

train_target = train_df['avg_vote'].reset_index(drop=True)
test_target = test_df['avg_vote'].reset_index(drop=True)

print(f"   Final feature matrix shape: {train_features.shape}")
print(f"   Total features: {train_features.shape[1]}")
print(f"     - Basic features: {len(feature_cols)}")
print(f"     - Genre features: {train_genre_encoded.shape[1]}")
print(f"     - Embedding features: {train_embed_df.shape[1]}")

# Add target column to features
train_features['avg_vote'] = train_target.values
test_features['avg_vote'] = test_target.values

# Save processed data
print("\n💾 Saving processed data...")
train_features.to_csv('data/train_features.csv', index=False)
test_features.to_csv('data/test_features.csv', index=False)

print("   ✓ Saved: data/train_features.csv (with target column)")
print("   ✓ Saved: data/test_features.csv (with target column)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("="*80)
print(f"\nDataset Summary:")
print(f"  Training samples: {len(train_features):,}")
print(f"  Test samples: {len(test_features):,}")
print(f"  Total features: {train_features.shape[1]:,}")
print(f"  Target range: [{train_target.min():.1f}, {train_target.max():.1f}]")
print(f"  Target mean: {train_target.mean():.3f}")
print(f"\nFeature Types:")
print(f"  • Numeric: year, duration, votes_log")
print(f"  • Target-encoded: country, language, production_company")
print(f"  • Historical ratings: director, writer, actor (with counts)")
print(f"  • Multi-hot: {train_genre_encoded.shape[1]} genres")
print(f"  • Embeddings: {train_embed_df.shape[1]}-dim description vectors")
print("\n🎯 Ready for modeling!")
print("="*80)
