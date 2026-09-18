# 8.7 VGG residual OGL-gap audit

Read-only audit of frozen C3 Stage-1 plus vanilla Stage-2 maps.  It reproduces
the established AUD and OGL per-sample IoU before analysing why OGL changes a
thresholded AUD map.  No model is trained and no prior result is overwritten.

Run from the repository root:

```bash
/home/wxr/miniconda3/envs/wwww/bin/python chuagnxindian/8.7_counterfactual_relative_attention_matching/residual_ogl_gap_audit/analyze.py
```
