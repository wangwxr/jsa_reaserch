DEFAULT RESULTS (all maps evaluated with the unchanged JSA protocol)

| Default τ=0.1, α=0.5 | VGG10 | VGG144 | Flickr10 | Flickr144 |
| --- | --- | --- | --- | --- |
| AUD_L4 | 0.4015 / 0.4074 | 0.4002 / 0.4127 | 0.7640 / 0.5916 | 0.8040 / 0.6228 |
| L3_AFFINITY | 0.0000 / 0.0410 | 0.0000 / 0.0399 | 0.0000 / 0.0532 | 0.0000 / 0.0446 |
| L3_NATIVE_REFINE | 0.0000 / 0.0444 | 0.0000 / 0.0417 | 0.0000 / 0.0612 | 0.0000 / 0.0476 |
| L3_POOLED_REFINE | 0.0019 / 0.0848 | 0.0004 / 0.0791 | 0.0000 / 0.1326 | 0.0040 / 0.1132 |

POST-HOC DIAGNOSTIC BEST VS OGL NUMERIC REFERENCES

| Dataset | Best native no-OGL | Best any refinement | JSA OGL ref | L3+L4 OGL ref |
| --- | --- | --- | --- | --- |
| VGG10 | τ=0.2 α=0.75: 0.0161 / 0.1605 | L3_POOLED7_REFINED τ=0.2 α=0.75: 0.1107 / 0.2425 | 0.4190 / 0.4176 | 0.4432 / 0.4292 |
| VGG144 | τ=0.2 α=0.75: 0.0056 / 0.1282 | L3_POOLED7_REFINED τ=0.2 α=0.75: 0.0686 / 0.2029 | 0.4294 / 0.4258 | 0.4343 / 0.4307 |
| Flickr10 | τ=0.2 α=0.75: 0.1120 / 0.2786 | L3_POOLED7_REFINED τ=0.2 α=0.75: 0.3320 / 0.3976 | 0.8160 / 0.6242 | 0.8400 / 0.6154 |
| Flickr144 | τ=0.2 α=0.75: 0.0400 / 0.1940 | L3_POOLED7_REFINED τ=0.2 α=0.75: 0.2240 / 0.3268 | 0.8440 / 0.6250 | 0.8440 / 0.6392 |

OGL/object-prior maps were not loaded or used; the table contains reference numbers only.
