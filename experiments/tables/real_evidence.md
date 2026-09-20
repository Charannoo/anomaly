# Real-defect evidence analysis (Phase 5, todo 10)

Evidence on CALIBRATED branch maps inside the real GT region (256-res GT
stored with E4 predictions). Relative anomaly evidence, not causal.

Images analysed: 1197 across 10 categories.

## Per-category summary

| category    |   n |   mean_p_rgb |   frac_rgb_dom |   frac_depth_dom |   frac_joint |   frac_uncertain |   s_full_mean |   s_full_min |   large_gt_frac |
|:------------|----:|-------------:|---------------:|-----------------:|-------------:|-----------------:|--------------:|-------------:|----------------:|
| bagel       | 110 |       0.5335 |         0.2818 |           0      |       0.5182 |           0.2    |        8.1307 |       5.2956 |          0.2182 |
| cable_gland | 108 |       0.6624 |         0.6667 |           0      |       0.1389 |           0.1944 |        9.5635 |       6.4796 |          0      |
| carrot      | 159 |       0.5888 |         0.434  |           0      |       0.3962 |           0.1698 |       11.0855 |       5.7978 |          0      |
| cookie      | 131 |       0.4192 |         0.0534 |           0.0076 |       0.7252 |           0.2137 |        5.3989 |       2.9703 |          0.4427 |
| dowel       | 130 |       0.5011 |         0.2231 |           0.0077 |       0.5692 |           0.2    |        8.6057 |       6.1501 |          0.0077 |
| foam        | 100 |       0.6326 |         0.56   |           0.01   |       0.23   |           0.2    |        5.2624 |       3.2041 |          0.01   |
| peach       | 132 |       0.6384 |         0.7121 |           0      |       0.0909 |           0.197  |        9.0833 |       5.6653 |          0.0076 |
| potato      | 114 |       0.6227 |         0.5    |           0      |       0.307  |           0.193  |        7.942  |       5.0605 |          0      |
| rope        | 101 |       0.1554 |         0      |           0.5545 |       0.1287 |           0.3168 |       15.4106 |       9.5575 |          0.1881 |
| tire        | 112 |       0.4325 |         0.0804 |           0      |       0.6964 |           0.2232 |        6.6319 |       5.2481 |          0      |

## Pooled

|    n |   mean_p_rgb |   frac_rgb_dom |   frac_depth_dom |   frac_joint |   frac_uncertain |   s_full_mean |
|-----:|-------------:|---------------:|-----------------:|-------------:|-----------------:|--------------:|
| 1197 |       0.5234 |         0.3542 |           0.0493 |       0.3885 |            0.208 |        8.7265 |

## Failure/discovery candidates (low E4 score, large GT)

| category   | defect        | sample_id                     |   gt_area |   label |   s_full |   p_rgb | evidence_class   |   e_rgb_topk |   e_depth_topk |   e_rgb_mean |   e_depth_mean |   rank_score |
|:-----------|:--------------|:------------------------------|----------:|--------:|---------:|--------:|:-----------------|-------------:|---------------:|-------------:|---------------:|-------------:|
| bagel      | combined      | bagel/test/combined/007       |    0.0168 |       1 |   6.9392 |  0.5646 | JOINT            |       7.468  |         5.7597 |       5.3268 |         1.8295 |            1 |
| bagel      | combined      | bagel/test/combined/021       |    0.0111 |       1 |   7.5212 |  0.5668 | JOINT            |       7.5975 |         5.8059 |       5.0214 |         2.1067 |            2 |
| bagel      | crack         | bagel/test/crack/007          |    0.0113 |       1 |   7.6893 |  0.5516 | JOINT            |       7.5623 |         6.1467 |       5.5537 |         2.328  |            3 |
| cookie     | crack         | cookie/test/crack/000         |    0.0158 |       1 |   3.4218 |  0.5598 | JOINT            |       3.6984 |         2.908  |       1.6815 |         0.8654 |            1 |
| cookie     | combined      | cookie/test/combined/021      |    0.0113 |       1 |   3.5928 |  0.5265 | JOINT            |       3.6824 |         3.3122 |       1.5209 |         1.6262 |            2 |
| cookie     | contamination | cookie/test/contamination/008 |    0.0189 |       1 |   4.1618 |  0.5711 | JOINT            |       4.6169 |         3.4677 |       2.3126 |         1.9957 |            3 |
| dowel      | combined      | dowel/test/combined/002       |    0.0114 |       1 |   8.9492 |  0.556  | JOINT            |       8.4289 |         6.7302 |       2.6361 |         3.2214 |            1 |
| foam       | color         | foam/test/color/000           |    0.0142 |       1 |   6.9144 |  0.866  | RGB-DOMINANT     |       8.4826 |         1.3128 |       4.7806 |        -0.4073 |            1 |
| peach      | combined      | peach/test/combined/024       |    0.0102 |       1 |  11.1891 |  0.7733 | RGB-DOMINANT     |      11.2987 |         3.3124 |       6.9355 |         1.2626 |            1 |
| rope       | open          | rope/test/open/003            |    0.0149 |       1 |  13.5645 |  0.2447 | DEPTH-DOMINANT   |       7.6244 |        23.5399 |       2.5846 |        14.7637 |            1 |
| rope       | open          | rope/test/open/013            |    0.0147 |       1 |  13.6657 |  0.0963 | DEPTH-DOMINANT   |       2.5081 |        23.5381 |       1.1509 |        15.5814 |            2 |
| rope       | open          | rope/test/open/002            |    0.0126 |       1 |  13.9839 |  0.0851 | DEPTH-DOMINANT   |       2.4184 |        25.9917 |       0.5832 |        13.1074 |            3 |

## Strongest real RGB-dominant / depth-dominant examples (per category)

### RGB-dominant

| category    | defect        | sample_id                     |   gt_area |   label |   s_full |   p_rgb | evidence_class   |   e_rgb_topk |   e_depth_topk |   e_rgb_mean |   e_depth_mean |
|:------------|:--------------|:------------------------------|----------:|--------:|---------:|--------:|:-----------------|-------------:|---------------:|-------------:|---------------:|
| bagel       | crack         | bagel/test/crack/003          |    0.0081 |       1 |   7.7349 |  0.817  | RGB-DOMINANT     |       9.1116 |         2.0409 |       5.8684 |         0.8529 |
| cable_gland | thread        | cable_gland/test/thread/017   |    0.0053 |       1 |   9.4113 |  1      | RGB-DOMINANT     |      11.3223 |        -0.1983 |       7.0241 |        -0.664  |
| carrot      | contamination | carrot/test/contamination/009 |    0.0066 |       1 |   7.8389 |  0.911  | RGB-DOMINANT     |      12.9405 |         1.264  |       5.9757 |         0.5022 |
| cookie      | contamination | cookie/test/contamination/022 |    0.0076 |       1 |   3.7694 |  0.9443 | RGB-DOMINANT     |       4.2257 |         0.2493 |       2.6068 |        -0.1287 |
| dowel       | contamination | dowel/test/contamination/009  |    0.0069 |       1 |  10.2069 |  0.8053 | RGB-DOMINANT     |      15.5504 |         3.7586 |       8.3132 |         1.9619 |
| foam        | color         | foam/test/color/000           |    0.0142 |       1 |   6.9144 |  0.866  | RGB-DOMINANT     |       8.4826 |         1.3128 |       4.7806 |        -0.4073 |
| peach       | combined      | peach/test/combined/009       |    0.006  |       1 |   8.4284 |  0.9633 | RGB-DOMINANT     |       8.4698 |         0.3222 |       5.7764 |        -0.3982 |
| potato      | combined      | potato/test/combined/010      |    0.0059 |       1 |  11.3656 |  0.7929 | RGB-DOMINANT     |      11.6245 |         3.0354 |       7.131  |         1.1228 |
| rope        | cut           | rope/test/cut/013             |    0.008  |       1 |  10.0625 |  0.3666 | JOINT            |       9.3224 |        16.1058 |       4.3254 |        11.3154 |
| tire        | hole          | tire/test/hole/024            |    0.0068 |       1 |   8.571  |  0.6692 | JOINT            |      10.4597 |         5.1711 |       5.5374 |         2.4875 |

### Depth-dominant

| category    | defect        | sample_id                     |   gt_area |   label |   s_full |   p_rgb | evidence_class   |   e_rgb_topk |   e_depth_topk |   e_rgb_mean |   e_depth_mean |
|:------------|:--------------|:------------------------------|----------:|--------:|---------:|--------:|:-----------------|-------------:|---------------:|-------------:|---------------:|
| bagel       | crack         | bagel/test/crack/008          |    0.0087 |       1 |   8.5502 |  0.4908 | JOINT            |       8.4946 |         8.8129 |       5.6815 |         3.833  |
| cable_gland | bent          | cable_gland/test/bent/003     |    0.0051 |       1 |   9.8931 |  0.6284 | JOINT            |       8.1496 |         4.8186 |       5.7594 |        -0.0854 |
| carrot      | crack         | carrot/test/crack/003         |    0.0066 |       1 |  11.9419 |  0.5222 | JOINT            |      13.7726 |        12.603  |      10.2326 |         7.4849 |
| cookie      | contamination | cookie/test/contamination/012 |    0.0053 |       1 |   5.0827 |  0.3358 | JOINT            |       1.819  |         3.5984 |       1.2464 |         1.8781 |
| dowel       | combined      | dowel/test/combined/013       |    0.0055 |       1 |   7.8851 |  0.4881 | JOINT            |      10.3807 |        10.8854 |       5.8117 |         4.2157 |
| foam        | combined      | foam/test/combined/003        |    0.006  |       1 |   5.6219 |  0.5514 | JOINT            |       6.5248 |         5.3078 |       3.2794 |         1.9018 |
| peach       | cut           | peach/test/cut/010            |    0.0095 |       1 |   9.9642 |  0.6246 | JOINT            |       9.5096 |         5.7144 |       6.4298 |         1.5924 |
| potato      | combined      | potato/test/combined/020      |    0.006  |       1 |   8.9855 |  0.6324 | JOINT            |      10.1863 |         5.922  |       8.5929 |         2.4649 |
| rope        | open          | rope/test/open/001            |    0.0132 |       1 |  18.0531 |  0.0673 | DEPTH-DOMINANT   |       1.9302 |        26.7621 |       0.1606 |        14.6348 |
| tire        | contamination | tire/test/contamination/010   |    0.0065 |       1 |   7.1695 |  0.5122 | JOINT            |       7.2356 |         6.8908 |       3.5781 |         2.948  |