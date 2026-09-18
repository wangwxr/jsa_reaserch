# Frozen alignment protocol

| Readout | Native source | Formal comparison representation |
|---|---:|---|
| AUD | Stage-2 `H_matched`, 14×14 | bicubic to 224×224, independently min-max normalized |
| IMG_QUERY | `image_native`, 7×7 | bicubic to 224×224, independently min-max normalized |
| IQR | AUD and IMG_QUERY | `minmax(0.6*AUD + 0.4*IMG_QUERY)` |
| OGL | AUD and object prior | `minmax(0.6*AUD + 0.4*OBJ_PRIOR)` |

`PositiveExcess=max(AUD-IMG_QUERY,0)` and `QueryExcess=max(IMG_QUERY-AUD,0)` remain continuous. The existing threshold 0.6 is used only to reconstruct the pre-existing AUD→OGL transition masks (RemoveFP/RemoveTP); no threshold is tuned for Group-A. Earlier native-grid replay was descriptive; this audit uses the formal cIoU grid to avoid resize-induced disagreement.
