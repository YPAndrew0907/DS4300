MATCH (n)
DETACH DELETE n;

CREATE CONSTRAINT song_track_id IF NOT EXISTS
FOR (s:Song)
REQUIRE s.track_id IS UNIQUE;

CREATE CONSTRAINT artist_name IF NOT EXISTS
FOR (a:Artist)
REQUIRE a.name IS UNIQUE;

CREATE CONSTRAINT genre_name IF NOT EXISTS
FOR (g:Genre)
REQUIRE g.name IS UNIQUE;

CREATE INDEX song_name IF NOT EXISTS
FOR (s:Song)
ON (s.name);

CREATE INDEX song_primary_artist IF NOT EXISTS
FOR (s:Song)
ON (s.primaryArtist);

CREATE INDEX song_primary_genre IF NOT EXISTS
FOR (s:Song)
ON (s.primaryGenre);

LOAD CSV WITH HEADERS FROM $csvRoot + "songs.csv" AS row
CREATE (:Song {
    track_id: row.track_id,
    name: row.name,
    trackName: row.track_name,
    albumName: row.album_name,
    artists: row.artists,
    primaryArtist: row.primary_artist,
    genres: row.genres,
    primaryGenre: row.primary_genre,

    popularity: toInteger(row.popularity),
    durationMs: toInteger(row.duration_ms),
    explicit: toBoolean(row.explicit),

    danceability: toFloat(row.danceability),
    energy: toFloat(row.energy),
    key: toInteger(row.key),
    loudness: toFloat(row.loudness),
    mode: toInteger(row.mode),
    speechiness: toFloat(row.speechiness),
    acousticness: toFloat(row.acousticness),
    instrumentalness: toFloat(row.instrumentalness),
    liveness: toFloat(row.liveness),
    valence: toFloat(row.valence),
    tempo: toFloat(row.tempo),
    timeSignature: toInteger(row.time_signature),

    isSeed: toBoolean(row.is_seed),
    seedArtist: row.seed_artist,
    isRecommendation: toBoolean(row.is_recommendation),

    recommendationRank:
        CASE row.recommendation_rank
            WHEN "" THEN null
            ELSE toInteger(row.recommendation_rank)
        END,

    pprScore: toFloat(row.ppr_score),
    recScore: toFloat(row.rec_score)
});

MATCH (s:Song)
WHERE s.isSeed = true
SET s:SeedSong;

MATCH (s:Song)
WHERE s.isRecommendation = true
SET s:RecommendedSong;

LOAD CSV WITH HEADERS FROM $csvRoot + "artists.csv" AS row
CREATE (:Artist {
    name: row.name
});

LOAD CSV WITH HEADERS FROM $csvRoot + "genres.csv" AS row
CREATE (:Genre {
    name: row.name
});

LOAD CSV WITH HEADERS FROM $csvRoot + "song_artist_edges.csv" AS row
MATCH (s:Song {track_id: row.track_id})
MATCH (a:Artist {name: row.artist})
CREATE (a)-[:PERFORMED {
    artistOrder: toInteger(row.artist_order)
}]->(s);

LOAD CSV WITH HEADERS FROM $csvRoot + "song_genre_edges.csv" AS row
MATCH (s:Song {track_id: row.track_id})
MATCH (g:Genre {name: row.genre})
CREATE (s)-[:IN_GENRE]->(g);

LOAD CSV WITH HEADERS FROM $csvRoot + "similar_edges.csv" AS row
CALL (row) {
    MATCH (source:Song {track_id: row.source})
    MATCH (target:Song {track_id: row.target})
    CREATE (source)-[:SIMILAR_TO {
        weight: toFloat(row.weight),
        metric: row.metric
    }]->(target)
} IN TRANSACTIONS OF 10000 ROWS;

MERGE (u:UserProfile {
    userId: "prof_rachlin",
    name: "Professor Rachlin"
});

MATCH (u:UserProfile {userId: "prof_rachlin"})
MATCH (s:SeedSong)
MERGE (u)-[:LIKES]->(s);

MATCH (u:UserProfile {userId: "prof_rachlin"})
MATCH (s:RecommendedSong)
MERGE (u)-[:RECOMMENDED {
    rank: s.recommendationRank,
    score: s.recScore
}]->(s);

MATCH (a:Artist)
WHERE a.name IN ["The Strokes", "Regina Spektor"]
SET a:LikedArtist;

MATCH (a:Artist)-[:PERFORMED]->(:RecommendedSong)
SET a:RecommendedArtist;

MATCH (s:Song)
WITH count(s) AS songNodes
MATCH ()-[r:SIMILAR_TO]->()
WITH songNodes, count(r) AS similarityEdges
RETURN
    songNodes,
    similarityEdges,
    round(100000000.0 * (2.0 * similarityEdges) / (songNodes * (songNodes - 1))) / 100000000.0
        AS undirectedGraphDensity;

MATCH (s:RecommendedSong)
RETURN
    s.recommendationRank AS rank,
    s.primaryArtist AS artist,
    s.albumName AS album,
    s.trackName AS title,
    round(1000000.0 * s.recScore) / 1000000.0 AS score
ORDER BY rank;
