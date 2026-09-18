# Positive-excess audit (read-only)

This audit tests whether the matched-audio response absent from `IMG_QUERY` is a safe, label-free proxy for the false-positive regions that OGL removes. It uses frozen Mean-CRAM Stage-2 outputs only; no training, loss, checkpoint, or data split was changed.

The 144k VGG-SS and Flickr replays are complete. The optional 10k replay was deliberately not run: full formal-grid pixel reconstruction was not low-cost and does not alter the prespecified 144k main conclusion.

Result: **S2**. In VGG Group-A, excess is FP/background-enriched and contains many OGL-removed FPs, but it does not separate RemoveFP from RemoveTP. In AUD-better samples its mass is substantially more GT-aligned and overlaps OGL RemoveTP. A global suppression of `AUD-IMG_QUERY` would therefore damage useful audio-added target coverage.
