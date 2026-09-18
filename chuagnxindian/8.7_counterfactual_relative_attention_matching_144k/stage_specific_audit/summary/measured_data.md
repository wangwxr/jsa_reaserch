# Measured data appendix

Gradient cosines: undefined zero-gradient groups excluded from fractions.

| dataset   | state        | group               |   cosine_mean |   cosine_median |   cosine_p25 |   cosine_p75 |   negative_fraction |   below_minus025 |   below_minus05 |   ratio_mean |
|:----------|:-------------|:--------------------|--------------:|----------------:|-------------:|-------------:|--------------------:|-----------------:|----------------:|-------------:|
| vggss     | old25_epoch1 | F34_activation      |        0.0924 |          0.0965 |       0.0669 |       0.1221 |              0.0000 |           0.0000 |          0.0000 |       0.6823 |
| vggss     | old25_epoch1 | K34_activation      |        0.0752 |          0.0741 |       0.0522 |       0.0972 |              0.0000 |           0.0000 |          0.0000 |       0.6794 |
| vggss     | old25_epoch1 | adapter             |       -0.8276 |         -0.8223 |      -0.8737 |      -0.7762 |              1.0000 |           1.0000 |          1.0000 |       0.9887 |
| vggss     | old25_epoch1 | adapter_output_conv |       -0.7555 |         -0.7338 |      -0.8014 |      -0.6880 |              1.0000 |           1.0000 |          1.0000 |       0.9246 |
| vggss     | old25_epoch1 | all_student         |       -0.8256 |         -0.8237 |      -0.8807 |      -0.7686 |              1.0000 |           1.0000 |          1.0000 |       1.0264 |
| vggss     | old25_epoch1 | proj3_spatial       |       -0.8265 |         -0.8222 |      -0.8675 |      -0.7812 |              1.0000 |           1.0000 |          1.0000 |       1.1707 |
| vggss     | old25_latest | F34_activation      |        0.0715 |          0.0598 |       0.0490 |       0.0823 |              0.0000 |           0.0000 |          0.0000 |       1.4295 |
| vggss     | old25_latest | K34_activation      |        0.0140 |          0.0175 |       0.0057 |       0.0257 |              0.2500 |           0.0000 |          0.0000 |       1.0084 |
| vggss     | old25_latest | adapter             |       -0.9892 |         -0.9909 |      -0.9913 |      -0.9888 |              1.0000 |           1.0000 |          1.0000 |       0.9700 |
| vggss     | old25_latest | adapter_output_conv |       -0.9880 |         -0.9893 |      -0.9907 |      -0.9866 |              1.0000 |           1.0000 |          1.0000 |       0.9562 |
| vggss     | old25_latest | all_student         |       -0.9885 |         -0.9901 |      -0.9910 |      -0.9876 |              1.0000 |           1.0000 |          1.0000 |       0.9530 |
| vggss     | old25_latest | proj3_spatial       |       -0.9884 |         -0.9898 |      -0.9916 |      -0.9865 |              1.0000 |           1.0000 |          1.0000 |       0.9343 |
| flickr    | old25_epoch1 | F34_activation      |        0.1011 |          0.1026 |       0.0907 |       0.1129 |              0.0000 |           0.0000 |          0.0000 |       0.5513 |
| flickr    | old25_epoch1 | K34_activation      |        0.0899 |          0.0918 |       0.0768 |       0.1048 |              0.0000 |           0.0000 |          0.0000 |       0.5456 |
| flickr    | old25_epoch1 | adapter             |       -0.9535 |         -0.9602 |      -0.9670 |      -0.9467 |              1.0000 |           1.0000 |          1.0000 |       1.2610 |
| flickr    | old25_epoch1 | adapter_output_conv |       -0.9636 |         -0.9681 |      -0.9760 |      -0.9557 |              1.0000 |           1.0000 |          1.0000 |       1.2943 |
| flickr    | old25_epoch1 | all_student         |       -0.9491 |         -0.9575 |      -0.9640 |      -0.9426 |              1.0000 |           1.0000 |          1.0000 |       1.2430 |
| flickr    | old25_epoch1 | proj3_spatial       |       -0.9427 |         -0.9582 |      -0.9623 |      -0.9386 |              1.0000 |           1.0000 |          1.0000 |       1.1896 |
| flickr    | old25_latest | F34_activation      |        0.0218 |          0.0200 |       0.0119 |       0.0300 |              0.0000 |           0.0000 |          0.0000 |       0.7687 |
| flickr    | old25_latest | K34_activation      |        0.0528 |          0.0483 |       0.0475 |       0.0537 |              0.0000 |           0.0000 |          0.0000 |       0.8811 |
| flickr    | old25_latest | adapter             |       -0.9736 |         -0.9770 |      -0.9805 |      -0.9701 |              1.0000 |           1.0000 |          1.0000 |       1.1070 |
| flickr    | old25_latest | adapter_output_conv |       -0.9641 |         -0.9763 |      -0.9805 |      -0.9599 |              1.0000 |           1.0000 |          1.0000 |       1.1447 |
| flickr    | old25_latest | all_student         |       -0.9774 |         -0.9792 |      -0.9823 |      -0.9743 |              1.0000 |           1.0000 |          1.0000 |       1.1038 |
| flickr    | old25_latest | proj3_spatial       |       -0.9855 |         -0.9839 |      -0.9862 |      -0.9832 |              1.0000 |           1.0000 |          1.0000 |       1.0974 |

## vggss

| model               |   gt_response |   nearfp_response |   gt_nearfp_gap |   gt_vs_nearfp_auroc |   coverage |   precision |   pred_area_ratio |   RemoveFP |    AddTP |   RemoveTP |     AddFP |   normalized_entropy |   cIoU |    AUC |
|:--------------------|--------------:|------------------:|----------------:|---------------------:|-----------:|------------:|------------------:|-----------:|---------:|-----------:|----------:|---------------------:|-------:|-------:|
| original_1.3g_final |        0.7979 |            0.7862 |          0.0150 |               0.5920 |     0.8677 |      0.4964 |            0.5078 |     0.0000 |   0.0000 |     0.0000 |    0.0000 |               0.9996 | 0.4269 | 0.4230 |
| cram_c3_stage1      |        0.7309 |            0.7034 |          0.0304 |               0.5901 |     0.7577 |      0.5442 |            0.3914 |  4845.6578 | 351.7003 |  2300.5252 |  953.1764 |               0.9989 | 0.3912 | 0.4132 |
| cram_c3_stage2      |        0.7631 |            0.7203 |          0.0455 |               0.6256 |     0.8110 |      0.5376 |            0.4303 |  3887.7410 | 441.6115 |  1497.0045 | 1053.7047 |               0.9993 | 0.4343 | 0.4296 |
| short_step000       |        0.7306 |            0.7023 |          0.0312 |               0.5936 |     0.7520 |      0.5456 |            0.3867 |  4957.2598 | 338.6357 |  2387.7635 |  927.8732 |               0.9992 | 0.3881 | 0.4126 |
| short_step016       |        0.7377 |            0.7088 |          0.0317 |               0.5962 |     0.7642 |      0.5390 |            0.3997 |  4667.5613 | 385.5283 |  2206.9855 | 1067.0698 |               0.9993 | 0.3932 | 0.4134 |
| short_step064       |        0.7508 |            0.7177 |          0.0359 |               0.6066 |     0.7862 |      0.5309 |            0.4194 |  4252.3581 | 460.0041 |  1911.1561 | 1267.5888 |               0.9994 | 0.3990 | 0.4163 |
| lambda025k_best     |        0.7377 |            0.7036 |          0.0366 |               0.5979 |     0.7763 |      0.5276 |            0.4194 |  4458.7703 | 523.3315 |  2156.9145 | 1655.7685 |               0.9994 | 0.3790 | 0.4081 |
| lambda025k_latest   |        0.4310 |            0.4190 |          0.0129 |               0.5809 |     0.2134 |      0.2241 |            0.1501 | 10521.3311 | 307.9463 |  9858.2439 | 2122.1634 |               0.9981 | 0.0727 | 0.1271 |
| lambda050k_best     |        0.6594 |            0.6384 |          0.0229 |               0.5658 |     0.6336 |      0.4362 |            0.4337 |  5373.8042 | 882.8145 |  4159.2295 | 4930.8319 |               0.9994 | 0.2076 | 0.3009 |
| lambda050k_latest   |        0.3501 |            0.3411 |          0.0094 |               0.5956 |     0.1719 |      0.1586 |            0.1377 | 10778.6179 | 291.5535 | 10547.2767 | 2461.7251 |               0.9954 | 0.0520 | 0.0950 |

| dataset   | model               | readout   |   cIoU |    AUC |
|:----------|:--------------------|:----------|-------:|-------:|
| vggss     | original_1.3g_final | AUD       | 0.4269 | 0.4230 |
| vggss     | original_1.3g_final | OGL       | 0.4570 | 0.4401 |
| vggss     | cram_c3_stage1      | AUD       | 0.3912 | 0.4132 |
| vggss     | cram_c3_stage1      | OGL       | 0.4242 | 0.4277 |
| vggss     | cram_c3_stage2      | AUD       | 0.4343 | 0.4296 |
| vggss     | cram_c3_stage2      | OGL       | 0.4523 | 0.4401 |
| vggss     | short_step000       | AUD       | 0.3881 | 0.4126 |
| vggss     | short_step000       | OGL       | 0.4205 | 0.4272 |
| vggss     | short_step016       | AUD       | 0.3932 | 0.4134 |
| vggss     | short_step016       | OGL       | 0.4257 | 0.4292 |
| vggss     | short_step064       | AUD       | 0.3990 | 0.4163 |
| vggss     | short_step064       | OGL       | 0.4374 | 0.4337 |
| vggss     | lambda025k_best     | AUD       | 0.3790 | 0.4081 |
| vggss     | lambda025k_best     | OGL       | 0.4329 | 0.4312 |
| vggss     | lambda025k_latest   | AUD       | 0.0727 | 0.1271 |
| vggss     | lambda025k_latest   | OGL       | 0.2819 | 0.3622 |
| vggss     | lambda050k_best     | AUD       | 0.2076 | 0.3009 |
| vggss     | lambda050k_best     | OGL       | 0.3404 | 0.3900 |
| vggss     | lambda050k_latest   | AUD       | 0.0520 | 0.0950 |
| vggss     | lambda050k_latest   | OGL       | 0.1929 | 0.3093 |

## flickr

| model               |   gt_response |   nearfp_response |   gt_nearfp_gap |   gt_vs_nearfp_auroc |   coverage |   precision |   pred_area_ratio |   RemoveFP |    AddTP |   RemoveTP |    AddFP |   normalized_entropy |   cIoU |    AUC |
|:--------------------|--------------:|------------------:|----------------:|---------------------:|-----------:|------------:|------------------:|-----------:|---------:|-----------:|---------:|---------------------:|-------:|-------:|
| original_1.3g_final |        0.8060 |            0.7640 |          0.0427 |               0.6600 |     0.9114 |      0.6449 |            0.6475 |     0.0000 |   0.0000 |     0.0000 |   0.0000 |               0.9995 | 0.8120 | 0.6356 |
| cram_c3_stage1      |        0.7132 |            0.6254 |          0.0889 |               0.6732 |     0.7655 |      0.7492 |            0.4631 |  5175.3760 | 222.3640 |  4656.1440 | 355.0600 |               0.9982 | 0.8640 | 0.6184 |
| cram_c3_stage2      |        0.7563 |            0.6590 |          0.0981 |               0.7078 |     0.8369 |      0.7256 |            0.5275 |  4183.6720 | 366.9720 |  2734.8440 | 529.0240 |               0.9988 | 0.8960 | 0.6532 |
| short_step000       |        0.7213 |            0.6292 |          0.0933 |               0.6782 |     0.7732 |      0.7479 |            0.4687 |  5110.9040 | 238.4640 |  4453.9040 | 352.2160 |               0.9988 | 0.8720 | 0.6236 |
| short_step016       |        0.7239 |            0.6340 |          0.0910 |               0.6764 |     0.7777 |      0.7435 |            0.4743 |  4991.8040 | 250.0600 |  4331.5000 | 382.3720 |               0.9989 | 0.8680 | 0.6240 |
| short_step064       |        0.7306 |            0.6383 |          0.0934 |               0.6844 |     0.7902 |      0.7393 |            0.4855 |  4809.0680 | 273.8240 |  4015.0000 | 417.3880 |               0.9989 | 0.8720 | 0.6296 |
| lambda025k_best     |        0.6706 |            0.5544 |          0.1179 |               0.7120 |     0.6836 |      0.8065 |            0.3877 |  6491.2560 | 134.7360 |  6810.6880 | 129.6200 |               0.9988 | 0.7560 | 0.5826 |
| lambda025k_latest   |        0.5604 |            0.4291 |          0.1329 |               0.7150 |     0.4709 |      0.8411 |            0.2626 |  7997.6840 | 220.3040 | 11890.0560 | 351.2200 |               0.9974 | 0.3880 | 0.4216 |
| lambda050k_best     |        0.5275 |            0.4197 |          0.1092 |               0.7035 |     0.4072 |      0.8476 |            0.2365 |  7656.0560 | 216.3040 | 13471.7520 | 286.3240 |               0.9976 | 0.2480 | 0.3440 |
| lambda050k_latest   |        0.4849 |            0.3695 |          0.1175 |               0.6810 |     0.3261 |      0.8443 |            0.1765 |  8512.7920 | 144.9600 | 15426.1520 | 157.2720 |               0.9956 | 0.1120 | 0.2964 |

| dataset   | model               | readout   |   cIoU |    AUC |
|:----------|:--------------------|:----------|-------:|-------:|
| flickr    | original_1.3g_final | AUD       | 0.8120 | 0.6356 |
| flickr    | original_1.3g_final | OGL       | 0.8680 | 0.6592 |
| flickr    | cram_c3_stage1      | AUD       | 0.8640 | 0.6184 |
| flickr    | cram_c3_stage1      | OGL       | 0.8480 | 0.6160 |
| flickr    | cram_c3_stage2      | AUD       | 0.8960 | 0.6532 |
| flickr    | cram_c3_stage2      | OGL       | 0.8880 | 0.6430 |
| flickr    | short_step000       | AUD       | 0.8720 | 0.6236 |
| flickr    | short_step000       | OGL       | 0.8520 | 0.6214 |
| flickr    | short_step016       | AUD       | 0.8680 | 0.6240 |
| flickr    | short_step016       | OGL       | 0.8520 | 0.6230 |
| flickr    | short_step064       | AUD       | 0.8720 | 0.6296 |
| flickr    | short_step064       | OGL       | 0.8640 | 0.6294 |
| flickr    | lambda025k_best     | AUD       | 0.7560 | 0.5826 |
| flickr    | lambda025k_best     | OGL       | 0.7800 | 0.5972 |
| flickr    | lambda025k_latest   | AUD       | 0.3880 | 0.4216 |
| flickr    | lambda025k_latest   | OGL       | 0.5720 | 0.5184 |
| flickr    | lambda050k_best     | AUD       | 0.2480 | 0.3440 |
| flickr    | lambda050k_best     | OGL       | 0.3880 | 0.4424 |
| flickr    | lambda050k_latest   | AUD       | 0.1120 | 0.2964 |
| flickr    | lambda050k_latest   | OGL       | 0.3960 | 0.4472 |

## Paired per-image performance intervals

| dataset   | candidate         | reference           | readout   | metric   |    mean |   ci_low |   ci_high |
|:----------|:------------------|:--------------------|:----------|:---------|--------:|---------:|----------:|
| vggss     | cram_c3_stage2    | original_1.3g_final | AUD       | cIoU     |  0.0074 |  -0.0017 |    0.0169 |
| vggss     | cram_c3_stage2    | original_1.3g_final | AUD       | AUC      |  0.0066 |   0.0043 |    0.0092 |
| vggss     | cram_c3_stage2    | original_1.3g_final | OGL       | cIoU     | -0.0047 |  -0.0130 |    0.0035 |
| vggss     | cram_c3_stage2    | original_1.3g_final | OGL       | AUC      | -0.0000 |  -0.0017 |    0.0016 |
| vggss     | lambda025k_best   | cram_c3_stage2      | AUD       | cIoU     | -0.0553 |  -0.0642 |   -0.0467 |
| vggss     | lambda025k_best   | cram_c3_stage2      | AUD       | AUC      | -0.0215 |  -0.0235 |   -0.0196 |
| vggss     | lambda025k_best   | cram_c3_stage2      | OGL       | cIoU     | -0.0194 |  -0.0266 |   -0.0128 |
| vggss     | lambda025k_best   | cram_c3_stage2      | OGL       | AUC      | -0.0089 |  -0.0102 |   -0.0075 |
| vggss     | lambda025k_latest | cram_c3_stage2      | AUD       | cIoU     | -0.3616 |  -0.3759 |   -0.3484 |
| vggss     | lambda025k_latest | cram_c3_stage2      | AUD       | AUC      | -0.3025 |  -0.3094 |   -0.2955 |
| vggss     | lambda025k_latest | cram_c3_stage2      | OGL       | cIoU     | -0.1704 |  -0.1819 |   -0.1590 |
| vggss     | lambda025k_latest | cram_c3_stage2      | OGL       | AUC      | -0.0778 |  -0.0815 |   -0.0741 |
| flickr    | cram_c3_stage2    | original_1.3g_final | AUD       | cIoU     |  0.0840 |   0.0440 |    0.1240 |
| flickr    | cram_c3_stage2    | original_1.3g_final | AUD       | AUC      |  0.0176 |   0.0058 |    0.0286 |
| flickr    | cram_c3_stage2    | original_1.3g_final | OGL       | cIoU     |  0.0200 |  -0.0120 |    0.0520 |
| flickr    | cram_c3_stage2    | original_1.3g_final | OGL       | AUC      | -0.0162 |  -0.0258 |   -0.0066 |
| flickr    | lambda025k_best   | cram_c3_stage2      | AUD       | cIoU     | -0.1400 |  -0.1960 |   -0.0880 |
| flickr    | lambda025k_best   | cram_c3_stage2      | AUD       | AUC      | -0.0706 |  -0.0866 |   -0.0544 |
| flickr    | lambda025k_best   | cram_c3_stage2      | OGL       | cIoU     | -0.1080 |  -0.1521 |   -0.0680 |
| flickr    | lambda025k_best   | cram_c3_stage2      | OGL       | AUC      | -0.0458 |  -0.0560 |   -0.0360 |
| flickr    | lambda025k_latest | cram_c3_stage2      | AUD       | cIoU     | -0.5080 |  -0.5720 |   -0.4439 |
| flickr    | lambda025k_latest | cram_c3_stage2      | AUD       | AUC      | -0.2316 |  -0.2584 |   -0.2066 |
| flickr    | lambda025k_latest | cram_c3_stage2      | OGL       | cIoU     | -0.3160 |  -0.3840 |   -0.2560 |
| flickr    | lambda025k_latest | cram_c3_stage2      | OGL       | AUC      | -0.1246 |  -0.1428 |   -0.1058 |
