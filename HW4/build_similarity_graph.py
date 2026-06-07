import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SAMPLE_CSV_PATH = "/Users/yipengandrewwang/DS4300/HW4/spotify_stratified_sample_60000.csv"
NEO4J_OUTPUT_DIR = "/Users/yipengandrewwang/DS4300/HW4/neo4j_spotify_import"

LIKED_ARTISTS = ["The Strokes", "Regina Spektor"]

NUM_RECOMMENDATIONS = 5


# we saw multiple artists are stored as "Artist A;Artist B".
# one track could have multiple artists for example "Ingrid Michaelson;ZAYN"
# in this case we need a function to separate them;
def split_artists(text):
    '''
    A helper function to separate multiple artists string into an array
    '''
    artists = []

    for artist in str(text).split(";"):
        artist = artist.strip()

        if artist:
            artists.append(artist)

    return artists


# whether a track is created by one of the liked artists is very important in track rec
# we use exact artist matching here so a track only counts as a seed when one of
# the semicolon-separated artist names is exactly The Strokes or Regina Spektor.
# we keep the seed artist name because later we balance the two taste lanes.
# Otherwise Regina Spektor gets drowned by the larger number of The Strokes tracks.
def seed_artist_name(artist_text):
    artists = split_artists(artist_text)

    for artist in LIKED_ARTISTS:
        if artist in artists:
            return artist

    # this will basically return "The Strokes", "Regina Spektor", or just ""
    return ""


# we normalize booleans through text because pandas can read True/False columns
# in slightly different ways depending on the CSV file.
# do not remove this. At least in the current sample, without this function it causes an error
def as_bool(value):
    return str(value).strip().lower() == "true"


# The graph should have one track node per track_id.
def prepare_songs(sample):
    songs = (
        sample
        .copy()
        .sort_values(["track_id", "popularity"], ascending=[True, False])
        .drop_duplicates("track_id")
        .reset_index(drop=True)
    )

    songs["primary_genre"] = songs["track_genre"].astype(str)
    songs["explicit_bool"] = songs["explicit"].apply(as_bool)

    return songs


# This is the main semantic feature builder.
def build_feature_matrix(songs):
    feature_frame = songs.copy()
    feature_frame["explicit_bool"] = feature_frame["explicit_bool"].astype(float)

    numeric_columns = []
    numeric_parts = []

    # automatically take every useful numeric track property.
    for column in feature_frame.columns:
        if column.lower().startswith("unnamed"):
            continue
        if column == "explicit":
            continue

        if pd.api.types.is_numeric_dtype(feature_frame[column]):
            values = pd.to_numeric(feature_frame[column], errors="coerce")

            if values.nunique(dropna=True) > 1:
                numeric_columns.append(column)
                numeric_parts.append(values.fillna(values.median()).to_numpy(float))

    numeric_features = np.column_stack(numeric_parts)


    # Each genre becomes a normal feature column, the same scaler/PCA process
    # decides how it interacts with audio properties.
    genre_features = songs["genres"].astype(str).str.get_dummies(sep=";")
    genres = sorted(genre_features.columns)
    genre_features = genre_features[genres].to_numpy(float)

    raw_features = np.column_stack([numeric_features, genre_features]).astype(np.float64)
    feature_columns = numeric_columns + ["genre:" + genre for genre in genres]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(raw_features)


    # we think PCA needed to be used here
    # the problem here is that if we directly did
    # sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2 + ...)
    #on the raw Spotify columns, the result would be messy because the columns live in different units.
    # duration_ms is in hundreds of thousands,
    # danceability is between 0 and 1, loudness is negative decibels,
    # and genres are category flags. In this case we have to normalize and PCA them
    pca = PCA(whiten=True, svd_solver="full")
    features = pca.fit_transform(scaled_features).astype(np.float32)

    return features, feature_columns


# This is where we build the actual track graph.
# we start with the sample-size k. If an unusual sample is disconnected, we let k
# grow until the graph is connected. That way we do not need to hand-pick a
# similarity threshold.
def build_similarity_graph(features):
    number_of_songs = features.shape[0]
    k = max(1, round(2.0 * math.sqrt(number_of_songs)))
    tree = cKDTree(features)

    while True:
        distances, neighbors = tree.query(features, k=k + 1, workers=-1)

        rows = []
        cols = []
        dists = []
        local_scales = []

        # we first store every directed nearest-neighbor edge. Later we symmetrize the
        # graph so Neo4j can treat SIMILAR_TO as an undirected relationship.
        for i in range(number_of_songs):
            song_neighbors = []
            song_distances = []

            for distance, neighbor_index in zip(distances[i], neighbors[i]):
                neighbor_index = int(neighbor_index)

                if neighbor_index == i:
                    continue

                song_neighbors.append(neighbor_index)
                song_distances.append(float(distance))

                if len(song_neighbors) == k:
                    break

            # we use the kth-neighbor distance as this track's local scale.
            # That scale comes from the data around the track, not from me guessing a
            # global threshold.
            local_scale = max(song_distances[-1], np.finfo(np.float32).tiny)
            local_scales.append(local_scale)

            for neighbor_index, distance in zip(song_neighbors, song_distances):
                rows.append(i)
                cols.append(neighbor_index)
                dists.append(distance)

        rows = np.array(rows, dtype=np.int32)
        cols = np.array(cols, dtype=np.int32)
        dists = np.array(dists, dtype=np.float32)
        local_scales = np.array(local_scales, dtype=np.float32)

        # This is the core similarity formula:
        #   exp(-distance^2 / (sigma_i * sigma_j))
        # where sigma_i is track i's local kth-neighbor distance.
        #
        # No hand-crafted feature weights. No hand-crafted genre boost. No manual
        # similarity threshold.
        weights = np.exp(
            -(dists ** 2) / (
                local_scales[rows] * local_scales[cols]
                + np.finfo(np.float32).tiny
            )
        ).astype(np.float32)

        graph = sparse.coo_matrix(
            (weights, (rows, cols)),
            shape=(number_of_songs, number_of_songs),
            dtype=np.float32,
        )

        # If i chose j or j chose i, we keep the stronger edge. That gives me a clean
        # undirected kNN graph without duplicate relationships in the CSV export.
        graph = graph.maximum(graph.T).tocsr()
        graph.eliminate_zeros()

        component_count, _ = connected_components(graph, directed=False)

        if component_count == 1 or k == number_of_songs - 1:
            return graph, k

        k = min(number_of_songs - 1, k * 2)


# we run personalized weighted PageRank in Python so the graph score is
# reproducible before Neo4j import. we still export this score for the poster/demo.
def personalized_pagerank(graph, seed_indices, seed_lanes):
    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    inverse_row_sums = np.divide(
        1.0,
        row_sums,
        out=np.zeros_like(row_sums),
        where=row_sums > 0,
    )

    transition = sparse.diags(inverse_row_sums).dot(graph).tocsr()

    restart = np.zeros(graph.shape[0], dtype=np.float64)
    lane_counts = Counter(seed_lanes)
    lane_count = len(lane_counts)

    # we give The Strokes and Regina Spektor equal total restart mass. Since
    # Regina has fewer tracks, each Regina seed gets a larger individual restart
    # value. This is a fairness step across liked-artist lanes, not a similarity
    # feature weight.
    for index, lane in zip(seed_indices, seed_lanes):
        restart[int(index)] += 1.0 / (lane_count * lane_counts[lane])

    restart = restart / restart.sum()
    scores = restart.copy()
    transition_t = transition.T.tocsr()

    # we read the damping value from the graph's own average edge strength.
    # Stronger average similarity means we trust graph walks more; weaker average
    # similarity means we restart back to the liked tracks more often.
    damping = float(graph.data.mean())
    damping = min(
        max(damping, np.finfo(np.float32).eps),
        1.0 - np.finfo(np.float32).eps,
    )

    # we use machine precision scaled by graph size as the stopping rule.
    # This keeps the run deterministic without a hand-picked iteration count.
    tolerance = np.finfo(np.float32).eps * graph.shape[0]

    delta = math.inf
    iterations = 0

    while delta > tolerance:
        next_scores = damping * (transition_t @ scores) + (1.0 - damping) * restart
        delta = float(np.abs(next_scores - scores).sum())
        scores = next_scores
        iterations += 1

    return scores.astype(np.float32), damping, iterations


# we compute a direct semantic score from the same learned feature space.
# For each taste lane, we measure every track against that lane's liked tracks, then
# use the lane's own median distance as its bandwidth. Again, the scale comes
# from the data instead of a hand-written value.
def seed_semantic_scores(features, seed_indices, seed_lanes):
    lane_to_indices = defaultdict(list)

    for index, lane in zip(seed_indices, seed_lanes):
        lane_to_indices[lane].append(int(index))

    lane_scores = {}

    for lane, lane_seed_indices in lane_to_indices.items():
        seed_features = features[lane_seed_indices]
        squared_distances = (
            (features[:, None, :] - seed_features[None, :, :]) ** 2
        ).sum(axis=2)

        bandwidth = float(np.median(squared_distances[squared_distances > 0]))

        lane_scores[lane] = np.exp(
            -squared_distances / (bandwidth + np.finfo(np.float32).tiny)
        ).max(axis=1).astype(np.float32)

    final_score = np.max(np.vstack(list(lane_scores.values())), axis=0)

    return final_score.astype(np.float32), lane_scores


# we finish with a simple lane-balanced recommendation pass.
# First we take one strong recommendation from each liked-artist lane, then we fill
# the rest by the overall semantic score. we also avoid repeating the same lead
# artist in the final five.
def choose_recommendations(songs, final_score, lane_scores, seed_indices):
    seed_set = set(int(index) for index in seed_indices)
    chosen = []
    used_artists = set()

    for lane in LIKED_ARTISTS:
        if lane not in lane_scores:
            continue

        score = lane_scores[lane]

        for index in np.argsort(-score):
            index = int(index)

            if index in seed_set:
                continue

            if seed_artist_name(songs.loc[index, "artists"]):
                continue

            if index in chosen:
                continue

            artist_list = split_artists(songs.loc[index, "artists"])
            lead_artist = artist_list[0] if artist_list else ""

            if lead_artist in used_artists:
                continue

            chosen.append(index)
            used_artists.add(lead_artist)
            break

    for index in np.argsort(-final_score):
        index = int(index)

        if len(chosen) == NUM_RECOMMENDATIONS:
            break

        if index in seed_set:
            continue

        if seed_artist_name(songs.loc[index, "artists"]):
            continue

        if index in chosen:
            continue

        artist_list = split_artists(songs.loc[index, "artists"])
        lead_artist = artist_list[0] if artist_list else ""

        if lead_artist in used_artists:
            continue

        chosen.append(index)
        used_artists.add(lead_artist)

    return chosen[:NUM_RECOMMENDATIONS]


# we use csv.DictWriter for exports because Neo4j LOAD CSV likes plain, predictable
# CSV files with simple headers.
def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# This function writes every CSV my Cypher import expects:
# track rows, artists, genres, artist edges, genre edges, similarity edges, final
# recommendation rows, and a tiny graph_stats file for the poster.
def export_for_neo4j(
    output_dir,
    songs,
    graph,
    final_score,
    ppr_score,
    recommendations,
    k_neighbors,
    damping,
    pagerank_iterations,
    feature_columns,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recommendation_rank = {}

    for rank, index in enumerate(recommendations, start=1):
        recommendation_rank[int(index)] = rank

    song_rows = []

    for i, row in songs.iterrows():
        artist_list = split_artists(row["artists"])
        primary_artist = ""

        if artist_list:
            primary_artist = artist_list[0]

        seed_artist = seed_artist_name(row["artists"])
        rec_rank = recommendation_rank.get(i, "")

        song_rows.append(
            {
                "track_id": row["track_id"],
                "name": str(row["track_name"]) + " — " + primary_artist,
                "track_name": row["track_name"],
                "album_name": row["album_name"],
                "artists": row["artists"],
                "primary_artist": primary_artist,
                "genres": row["genres"],
                "primary_genre": row["primary_genre"],
                "popularity": int(row["popularity"]),
                "duration_ms": int(row["duration_ms"]),
                "explicit": str(bool(row["explicit_bool"])).lower(),
                "danceability": float(row["danceability"]),
                "energy": float(row["energy"]),
                "key": int(row["key"]),
                "loudness": float(row["loudness"]),
                "mode": int(row["mode"]),
                "speechiness": float(row["speechiness"]),
                "acousticness": float(row["acousticness"]),
                "instrumentalness": float(row["instrumentalness"]),
                "liveness": float(row["liveness"]),
                "valence": float(row["valence"]),
                "tempo": float(row["tempo"]),
                "time_signature": int(row["time_signature"]),
                "is_seed": str(seed_artist != "").lower(),
                "seed_artist": seed_artist,
                "is_recommendation": str(i in recommendation_rank).lower(),
                "recommendation_rank": rec_rank,
                "ppr_score": float(ppr_score[i]),
                "rec_score": float(final_score[i]),
            }
        )

    write_csv(
        output_dir / "songs.csv",
        song_rows,
        [
            "track_id",
            "name",
            "track_name",
            "album_name",
            "artists",
            "primary_artist",
            "genres",
            "primary_genre",
            "popularity",
            "duration_ms",
            "explicit",
            "danceability",
            "energy",
            "key",
            "loudness",
            "mode",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "time_signature",
            "is_seed",
            "seed_artist",
            "is_recommendation",
            "recommendation_rank",
            "ppr_score",
            "rec_score",
        ],
    )

    artist_names = set()

    for text in songs["artists"]:
        for artist in split_artists(text):
            artist_names.add(artist)

    artist_rows = []

    for artist in sorted(artist_names):
        artist_rows.append({"name": artist})

    write_csv(output_dir / "artists.csv", artist_rows, ["name"])

    genre_names = set()

    for text in songs["genres"]:
        for genre in str(text).split(";"):
            genre = genre.strip()

            if genre:
                genre_names.add(genre)

    genre_rows = []

    for genre in sorted(genre_names):
        genre_rows.append({"name": genre})

    write_csv(output_dir / "genres.csv", genre_rows, ["name"])

    song_artist_rows = []

    for _, row in songs.iterrows():
        artists_for_song = split_artists(row["artists"])

        for order, artist in enumerate(artists_for_song, start=1):
            song_artist_rows.append(
                {
                    "track_id": row["track_id"],
                    "artist": artist,
                    "artist_order": order,
                }
            )

    write_csv(
        output_dir / "song_artist_edges.csv",
        song_artist_rows,
        ["track_id", "artist", "artist_order"],
    )

    song_genre_rows = []

    for _, row in songs.iterrows():
        for genre in str(row["genres"]).split(";"):
            genre = genre.strip()

            if genre:
                song_genre_rows.append(
                    {
                        "track_id": row["track_id"],
                        "genre": genre,
                    }
                )

    write_csv(
        output_dir / "song_genre_edges.csv",
        song_genre_rows,
        ["track_id", "genre"],
    )

    # we export only the upper triangle because Neo4j only needs one stored
    # relationship for each undirected track pair.
    upper_graph = sparse.triu(graph, k=1).tocoo()
    track_ids = songs["track_id"].tolist()
    similarity_rows = []

    for source, target, weight in zip(upper_graph.row, upper_graph.col, upper_graph.data):
        similarity_rows.append(
            {
                "source": track_ids[int(source)],
                "target": track_ids[int(target)],
                "weight": float(weight),
                "metric": "standardized_numeric_plus_genre_pca_whitened_self_tuning_knn",
            }
        )

    write_csv(
        output_dir / "similar_edges.csv",
        similarity_rows,
        ["source", "target", "weight", "metric"],
    )

    recommendation_rows = []

    for rank, index in enumerate(recommendations, start=1):
        row = songs.loc[int(index)]

        recommendation_rows.append(
            {
                "rank": rank,
                "track_id": row["track_id"],
                "artist": row["artists"],
                "album": row["album_name"],
                "title": row["track_name"],
                "score": float(final_score[int(index)]),
            }
        )

    write_csv(
        output_dir / "recommendations.csv",
        recommendation_rows,
        ["rank", "track_id", "artist", "album", "title", "score"],
    )

    graph_density = (2.0 * len(similarity_rows)) / (len(songs) * (len(songs) - 1))

    graph_stats_rows = [
        {
            "song_nodes": len(songs),
            "similar_to_edges": len(similarity_rows),
            "undirected_graph_density": graph_density,
            "k_neighbors": k_neighbors,
            "pagerank_damping_from_graph": damping,
            "pagerank_iterations_to_converge": pagerank_iterations,
            "semantic_feature_columns": ";".join(feature_columns),
        }
    ]

    write_csv(
        output_dir / "graph_stats.csv",
        graph_stats_rows,
        [
            "song_nodes",
            "similar_to_edges",
            "undirected_graph_density",
            "k_neighbors",
            "pagerank_damping_from_graph",
            "pagerank_iterations_to_converge",
            "semantic_feature_columns",
        ],
    )

    return len(similarity_rows), graph_density



def main():
    sample = pd.read_csv(SAMPLE_CSV_PATH)
    songs = prepare_songs(sample)
    features, feature_columns = build_feature_matrix(songs)

    seed_indices = []
    seed_lanes = []

    for i, row in songs.iterrows():
        lane = seed_artist_name(row["artists"])

        if lane:
            seed_indices.append(i)
            seed_lanes.append(lane)

    seed_indices = np.array(seed_indices, dtype=np.int32)

    if len(seed_indices) == 0:
        raise RuntimeError("No seed songs found. Check the sample and LIKED_ARTISTS.")

    graph, k_neighbors = build_similarity_graph(features)
    ppr_score, damping, pagerank_iterations = personalized_pagerank(
        graph,
        seed_indices,
        seed_lanes,
    )

    # we use the learned semantic space for the final recommendation score.
    final_score, lane_scores = seed_semantic_scores(
        features,
        seed_indices,
        seed_lanes,
    )

    recommendations = choose_recommendations(
        songs,
        final_score,
        lane_scores,
        seed_indices,
    )

    edge_count, density = export_for_neo4j(
        NEO4J_OUTPUT_DIR,
        songs,
        graph,
        final_score,
        ppr_score,
        recommendations,
        k_neighbors,
        damping,
        pagerank_iterations,
        feature_columns,
    )

    print("Sample rows / Song nodes:", len(songs))
    print("Seed songs:", len(seed_indices), Counter(seed_lanes))
    print("Semantic feature count:", len(feature_columns))
    print("Auto-selected k_neighbors:", k_neighbors)
    print("PageRank damping learned from graph:", round(damping, 6))
    print("PageRank iterations to converge:", pagerank_iterations)
    print("SIMILAR_TO edges:", edge_count)
    print("Undirected graph density:", round(density, 8))
    print("Neo4j CSV output directory:", NEO4J_OUTPUT_DIR)
    print()
    print("Final five recommendations:")

    for rank, index in enumerate(recommendations, start=1):
        row = songs.loc[int(index)]
        print(f"{rank}. {row['artists']} | {row['album_name']} | {row['track_name']}")


if __name__ == "__main__":
    main()