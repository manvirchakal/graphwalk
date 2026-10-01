# A5: scale benchmark (synthetic graph, scripted decider)

git 416b485c15ba249122faf6e031d3cb7740b94c8f; args {"nodes": 200000, "out_degree": 6, "hubs": 10, "relations": 20, "samples": 200, "walks": 50, "seed": 0}

Import: 1,399,994 triples -> 200,010 nodes, 1,399,994 edges in 176 s (7,940 triples/s); 1141 MB on disk. Peak RSS 911 MB.

| operation | n | p50 ms | p95 ms | max ms | notes |
|---|---|---|---|---|---|
| neighbors, regular nodes | 200 | 0.4 | 0.6 | 1.3 | mean degree 13 |
| degree, regular nodes | 200 | 0.2 | 0.2 | 0.3 | mean degree 13 |
| neighbors, hub nodes | 10 | 1157.1 | 1419.5 | 1419.5 | mean degree 20,000 |
| degree, hub nodes | 10 | 492.9 | 724.3 | 724.3 | mean degree 20,000 |
| name linking | 50 | 0.0 | 0.0 | 0.2 | index build 6.6 s; right node linked 100% |
| walk, entity/truncate/regular | 50 | 9.4 | 1319.6 | 2673.1 | 3.0 decisions |
| walk, entity/truncate/hub | 10 | 1307.2 | 1339.1 | 1339.1 | 3.0 decisions |
| walk, entity/prefilter/regular | 50 | 9.7 | 1589.8 | 3104.2 | 3.0 decisions |
| walk, entity/prefilter/hub | 10 | 1623.7 | 1877.1 | 1877.1 | 3.0 decisions |
| walk, relation/truncate/regular | 50 | 10.4 | 6047.8 | 6194.2 | 3.0 decisions |
| walk, relation/truncate/hub | 10 | 3069.2 | 3258.7 | 3258.7 | 3.0 decisions |
| walk, relation/prefilter/regular | 50 | 9.3 | 6302.0 | 6524.7 | 3.0 decisions |
| walk, relation/prefilter/hub | 10 | 3192.2 | 3309.1 | 3309.1 | 3.0 decisions |
