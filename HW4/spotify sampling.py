import pandas as pd

RANDOM_SEED = 42
TARGET_SAMPLE_SIZE = 60000

INPUT_PATH = "/Users/yipengandrewwang/DS4300/HW4/spotify.csv"
OUTPUT_PATH = "/Users/yipengandrewwang/DS4300/HW4/spotify_stratified_sample_60000.csv"
VALIDATION_PATH = "/Users/yipengandrewwang/DS4300/HW4/spotify_sample_validation.csv"

df = pd.read_csv(INPUT_PATH)

# Force include Professor Rachlin's liked artists
liked_artist_pattern = r"(?:^|;)The Strokes(?:;|$)|(?:^|;)Regina Spektor(?:;|$)"


def add_stratification_columns(data):
    data = data.copy()

    data["popularity_bin"] = pd.cut(
        data["popularity"],
        bins=[-1, 24, 49, 74, 100],
        labels=["low", "medium", "high", "very_high"]
    )

    data["duration_bin"] = pd.cut(
        data["duration_ms"] / 60000,
        bins=[0, 2.5, 4, 6, float("inf")],
        labels=["short", "medium", "long", "very_long"],
        include_lowest=True
    )

    data["energy_bin"] = pd.cut(
        data["energy"],
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["low_energy", "medium_energy", "high_energy"]
    )

    data["danceability_bin"] = pd.cut(
        data["danceability"],
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["low_dance", "medium_dance", "high_dance"]
    )

    # I normalize explicit through text because pandas sometimes reads this as
    # bools and sometimes as strings depending on the CSV.
    explicit_text = data["explicit"].astype(str).str.lower()
    data["explicit_bin"] = explicit_text.map({
        "true": "explicit",
        "false": "clean"
    })

    return data


df = add_stratification_columns(df)

# I keep the full row-level dataset for validation.
# Then I make the sampling frame one row per track_id, because Neo4j should have
# one Song node per real song.
full_df = df.copy()

# I preserve every genre attached to a track_id because the Kaggle file repeats
# some songs under multiple genres. Neo4j can use this as real song metadata.
genre_lookup = (
    df
    .groupby("track_id")["track_genre"]
    .apply(lambda values: ";".join(sorted(set(str(value) for value in values))))
    .rename("genres")
    .reset_index()
)

# I still keep one primary genre so the stratified sample stays close to your
# original sampling code.
primary_genre_lookup = (
    df
    .groupby("track_id")["track_genre"]
    .first()
    .rename("track_genre")
    .reset_index()
)

df = (
    df
    .sort_values(["track_id", "popularity"], ascending=[True, False])
    .drop_duplicates("track_id")
    .drop(columns=["track_genre"])
    .merge(genre_lookup, on="track_id", how="left")
    .merge(primary_genre_lookup, on="track_id", how="left")
    .reset_index(drop=True)
)

liked_songs = df[df["artists"].str.contains(
    liked_artist_pattern,
    case=False,
    na=False,
    regex=True
)].copy()

remaining_n = TARGET_SAMPLE_SIZE - len(liked_songs)

if remaining_n < 0:
    raise ValueError("Liked songs are more than the target sample size.")

pool = df[~df.index.isin(liked_songs.index)].copy()

stratify_columns = [
    "track_genre",
    "popularity_bin",
    "explicit_bin",
    "duration_bin",
    "energy_bin",
    "danceability_bin"
]

sampled = (
    pool
    .groupby(stratify_columns, group_keys=False, observed=True)
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


validation_parts = []

for column in stratify_columns:
    full_counts = (
        full_df[column]
        .astype(str)
        .value_counts()
        .rename_axis("value")
        .reset_index(name="full_count")
    )

    sample_counts = (
        sample_df[column]
        .astype(str)
        .value_counts()
        .rename_axis("value")
        .reset_index(name="sample_count")
    )

    validation = full_counts.merge(sample_counts, on="value", how="left").fillna(0)
    validation["sample_count"] = validation["sample_count"].astype(int)
    validation["full_percent"] = validation["full_count"] / len(full_df)
    validation["sample_percent"] = validation["sample_count"] / len(sample_df)
    validation.insert(0, "stratified_column", column)

    validation_parts.append(validation)

validation_df = pd.concat(validation_parts, ignore_index=True)
validation_df.to_csv(VALIDATION_PATH, index=False)

print("Full dataset rows:", len(full_df))
print("Unique track rows available after de-duplication:", len(df))
print("Sample rows:", len(sample_df))
print("Unique track IDs in sample:", sample_df["track_id"].nunique())
print("Genres in full dataset:", full_df["track_genre"].nunique())
print("Genres in sample:", sample_df["track_genure" if False else "track_genre"].nunique())

print("The Strokes rows:", sample_df["artists"].str.contains(
    r"(?:^|;)The Strokes(?:;|$)", case=False, na=False, regex=True
).sum())

print("Regina Spektor rows:", sample_df["artists"].str.contains(
    r"(?:^|;)Regina Spektor(?:;|$)", case=False, na=False, regex=True
).sum())

print("Sample saved to:", OUTPUT_PATH)
print("Validation saved to:", VALIDATION_PATH)