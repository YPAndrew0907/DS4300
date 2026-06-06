import pandas as pd

RANDOM_SEED = 42
TARGET_SAMPLE_SIZE = 3000

INPUT_PATH = "spotify.csv"
OUTPUT_PATH = "spotify_stratified_sample_3000.csv"
VALIDATION_PATH = "spotify_sample_validation.csv"

df = pd.read_csv(INPUT_PATH)

# Force include Professor Rachlin's liked artists
liked_artist_pattern = r"(?:^|;)The Strokes(?:;|$)|(?:^|;)Regina Spektor(?:;|$)"

liked_songs = df[df["artists"].str.contains(
    liked_artist_pattern,
    case=False,
    na=False,
    regex=True
)].copy()

# Add popularity bins so the sample keeps a good mix of popular and less popular songs
df["popularity_bin"] = pd.cut(
    df["popularity"],
    bins=[-1, 24, 49, 74, 100],
    labels=["low", "medium", "high", "very_high"]
)

liked_songs["popularity_bin"] = pd.cut(
    liked_songs["popularity"],
    bins=[-1, 24, 49, 74, 100],
    labels=["low", "medium", "high", "very_high"]
)

remaining_n = TARGET_SAMPLE_SIZE - len(liked_songs)

pool = df[~df.index.isin(liked_songs.index)].copy()

sampled = (
    pool
    .groupby(["track_genre", "popularity_bin"], group_keys=False, observed=True)
    .sample(frac=remaining_n / len(pool), random_state=RANDOM_SEED)
)

if len(sampled) < remaining_n:
    top_up = pool[~pool.index.isin(sampled.index)].sample(
        n=remaining_n - len(sampled),
        random_state=RANDOM_SEED
    )
    sampled = pd.concat([sampled, top_up], ignore_index=False)

elif len(sampled) > remaining_n:
    sampled = sampled.sample(n=remaining_n, random_state=RANDOM_SEED)

sample_df = pd.concat([liked_songs, sampled], ignore_index=True)
sample_df = sample_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

sample_df.to_csv(OUTPUT_PATH, index=False)

validation = sample_df["track_genre"].value_counts().reset_index()
validation.columns = ["track_genre", "sample_count"]
validation.to_csv(VALIDATION_PATH, index=False)

print("Full dataset rows:", len(df))
print("Sample rows:", len(sample_df))
print("Genres in full dataset:", df["track_genre"].nunique())
print("Genres in sample:", sample_df["track_genre"].nunique())

print("The Strokes rows:", sample_df["artists"].str.contains(
    r"(?:^|;)The Strokes(?:;|$)", case=False, na=False, regex=True
).sum())

print("Regina Spektor rows:", sample_df["artists"].str.contains(
    r"(?:^|;)Regina Spektor(?:;|$)", case=False, na=False, regex=True
).sum())

print("Sample saved to:", OUTPUT_PATH)
print("Validation saved to:", VALIDATION_PATH)
