CALL gds.graph.drop("songGraph", false)
YIELD graphName
RETURN graphName;

CALL gds.graph.project(
    "songGraph",
    "Song",
    {
        SIMILAR_TO: {
            orientation: "UNDIRECTED",
            properties: "weight"
        }
    }
)
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount;

MATCH (seed:SeedSong)
WITH seed.seedArtist AS lane, collect(seed) AS seeds
WITH collect({
    lane: lane,
    seeds: seeds,
    laneSize: size(seeds)
}) AS lanes
WITH lanes, size(lanes) AS laneCount
UNWIND lanes AS laneInfo
UNWIND laneInfo.seeds AS seed
WITH collect([
    seed,
    1.0 / (laneCount * laneInfo.laneSize)
]) AS sourceNodes
CALL gds.pageRank.stream("songGraph", {
    maxIterations: 80,
    dampingFactor: 0.72,
    relationshipWeightProperty: "weight",
    sourceNodes: sourceNodes
})
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS song, score
SET song.neo4jPprScore = score
RETURN
    song.primaryArtist AS artist,
    song.trackName AS title,
    score
ORDER BY score DESC
LIMIT 20;

MATCH (candidate:Song)
WHERE candidate.isSeed = false
  AND none(artist IN split(candidate.artists, ";")
           WHERE trim(artist) IN ["The Strokes", "Regina Spektor"])
  AND EXISTS {
      MATCH (candidate)-[:IN_GENRE]->(:Genre)<-[:IN_GENRE]-(:SeedSong)
  }
OPTIONAL MATCH (candidate)-[r:SIMILAR_TO]-(seed:SeedSong)
WITH
    candidate,
    coalesce(max(r.weight), 0.0) AS directSeedWeight
WITH
    candidate,
    (
        100000.0 * coalesce(candidate.neo4jPprScore, candidate.pprScore, 0.0)
        + 0.40 * directSeedWeight
        + 0.03 * log(10 + candidate.popularity)
    ) AS neo4jRecommendationScore
WITH
    candidate.primaryArtist AS artist,
    collect({
        song: candidate,
        score: neo4jRecommendationScore
    }) AS songsByArtist
WITH songsByArtist[0] AS bestPerArtist
WITH bestPerArtist.song AS song, bestPerArtist.score AS score
ORDER BY score DESC
RETURN
    song.primaryArtist AS artist,
    song.albumName AS album,
    song.trackName AS title,
    round(1000000.0 * score) / 1000000.0 AS score
LIMIT 10;

CALL gds.louvain.write("songGraph", {
    relationshipWeightProperty: "weight",
    writeProperty: "communityId"
})
YIELD communityCount, modularity
RETURN communityCount, modularity;

MATCH (c:Community)
DETACH DELETE c;

MATCH (s:Song)
WITH
    s.communityId AS communityId,
    count(s) AS size,
    avg(coalesce(s.pprScore, 0.0)) AS avgPpr,
    collect(s.primaryGenre)[0] AS sampleGenre
CREATE (:Community {
    communityId: communityId,
    name: "Community " + toString(communityId),
    size: size,
    avgPpr: avgPpr,
    sampleGenre: sampleGenre
});

MATCH (s:Song)
MATCH (c:Community {communityId: s.communityId})
CREATE (s)-[:IN_COMMUNITY]->(c);

MATCH ()-[r:COMMUNITY_SIMILAR]->()
DELETE r;

MATCH (a:Song)-[r:SIMILAR_TO]->(b:Song)
WHERE a.communityId IS NOT NULL
  AND b.communityId IS NOT NULL
  AND a.communityId < b.communityId
WITH
    a.communityId AS sourceCommunity,
    b.communityId AS targetCommunity,
    count(r) AS edgeCount,
    avg(r.weight) AS avgWeight
MATCH (source:Community {communityId: sourceCommunity})
MATCH (target:Community {communityId: targetCommunity})
CREATE (source)-[:COMMUNITY_SIMILAR {
    edgeCount: edgeCount,
    weight: avgWeight
}]->(target);

MATCH (c:Community)
RETURN count(c) AS communities;

MATCH ()-[r:COMMUNITY_SIMILAR]->()
RETURN count(r) AS communitySimilarEdges;
