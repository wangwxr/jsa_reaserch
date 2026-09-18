# Frozen protocol

AUD and IMG_QUERY use the preceding positive-excess alignment: native maps are independently bicubic resized to 224×224 and min-max normalized, and `PositiveExcess=max(AUD-IMG_QUERY,0)`. The existing excess threshold 0.10 defines the ambiguous set. The internal target core is `AUD>=0.6 AND IMG_QUERY>=0.6`; no GT/OGL/group signal participates in any candidate definition.

GT/OGL labels are used only offline. TP-excess is evaluated both as GT-excess and as retained-TP-excess. FP-excess is evaluated both as outside-GT excess and as OGL-RemoveFP excess. VGG Group-A remains the frozen 428 IDs.
