# A5: scale benchmark (synthetic graph, scripted decider)

git c554bd4b279df2f649b9c7eddd432dbfdace03d5; args {"nodes": 200000, "out_degree": 6, "hubs": 10, "relations": 20, "samples": 200, "walks": 50, "seed": 0}

Import: 1,399,994 triples -> 200,010 nodes, 1,399,994 edges in 155 s (9,056 triples/s); 1141 MB on disk. Peak RSS 911 MB.

| operation | n | p50 ms | p95 ms | max ms | notes |
|---|---|---|---|---|---|
| neighbors, regular nodes | 200 | 0.4 | 0.6 | 1.3 | mean degree 13 |
| adjacency, regular nodes | 200 | 0.0 | 0.1 | 0.1 | mean degree 13 |
| degree, regular nodes | 200 | 0.0 | 0.1 | 0.1 | mean degree 13 |
| neighbors, hub nodes | 10 | 1036.9 | 1202.8 | 1202.8 | mean degree 20,000 |
| adjacency, hub nodes | 10 | 55.8 | 185.8 | 185.8 | mean degree 20,000 |
| degree, hub nodes | 10 | 35.6 | 39.4 | 39.4 | mean degree 20,000 |
| name linking | 50 | 0.0 | 0.0 | 0.0 | index build 5.9 s; right node linked 100% |
| walk, entity/truncate/regular | 50 | 7.2 | 192.8 | 329.4 | 3.0 decisions |
| walk, entity/truncate/hub | 10 | 191.1 | 207.3 | 207.3 | 3.0 decisions |
| walk, entity/prefilter/regular | 50 | 7.7 | 579.3 | 1018.7 | 3.0 decisions |
| walk, entity/prefilter/hub | 10 | 538.7 | 689.8 | 689.8 | 3.0 decisions |
| walk, relation/truncate/regular | 50 | 7.7 | 176.2 | 179.4 | 3.0 decisions |
| walk, relation/truncate/hub | 10 | 113.2 | 140.9 | 140.9 | 3.0 decisions |
| walk, relation/prefilter/regular | 50 | 7.9 | 743.7 | 844.6 | 3.0 decisions |
| walk, relation/prefilter/hub | 10 | 418.0 | 505.7 | 505.7 | 3.0 decisions |
