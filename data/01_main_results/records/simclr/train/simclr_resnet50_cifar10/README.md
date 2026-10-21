# SimCLR ResNet-50 / CIFAR-10

SimCLR: train-mode extraction.

`p` is a p-value except for MMDAgg, where it records a 0/1 rejection decision.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## MMD, ED and C2ST · Multiple budgets in each CSV

Tests: C2ST, ED, ED (raw features), MMD, MMD (raw features).

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [ssl_cifar10_bntrain_base_c481-580.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c481-580.csv) | 505–580 (16) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c581-680.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c581-680.csv) | 585–680 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c681-780.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c681-780.csv) | 685–780 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c686-689.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c686-689.csv) | 686–689 (4) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c688-712.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c688-712.csv) | 691–712 (18) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c713-737.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c713-737.csv) | 713–737 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c738-762.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c738-762.csv) | 738–762 (16) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c756-759.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c756-759.csv) | 756–759 (4) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c763-787.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c763-787.csv) | 763–787 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c781-880.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c781-880.csv) | 785–880 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c788-812.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c788-812.csv) | 788–812 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c813-837.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c813-837.csv) | 813–837 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c838-862.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c838-862.csv) | 838–862 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c863-887.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c863-887.csv) | 863–887 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c881-980.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c881-980.csv) | 885–980 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c888-912.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c888-912.csv) | 888–912 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c913-937.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c913-937.csv) | 913–937 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c938-962.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c938-962.csv) | 938–962 (12) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c946-949.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c946-949.csv) | 946–949 (4) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c951-954.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c951-954.csv) | 951–954 (4) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c963-987.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c963-987.csv) | 963–987 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c981-1000.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c981-1000.csv) | 985–1000 (4) | 42–91 (50) |
| [ssl_cifar10_bntrain_base_c988-1000.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_base_c988-1000.csv) | 988–999 (10) | 42–91 (50) |

## MMD and ED on raw features · 10,000 points per split

Tests: ED (raw features), MMD (raw features).

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [ssl_cifar10_bntrain_raw_c536-539.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_bntrain_raw_c536-539.csv) | 536–539 (4) | 42–91 (50) |
| [ssl_cifar10_bntrain_raw_c541-565.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_bntrain_raw_c541-565.csv) | 541–564 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_raw_c566-590.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_bntrain_raw_c566-590.csv) | 566–589 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_raw_c591-615.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_bntrain_raw_c591-615.csv) | 591–614 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_raw_c616-640.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_bntrain_raw_c616-640.csv) | 616–639 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_raw_c641-665.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_bntrain_raw_c641-665.csv) | 641–664 (16) | 42–91 (50) |
| [ssl_cifar10_bntrain_raw_c646-649.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_bntrain_raw_c646-649.csv) | 646–649 (4) | 42–91 (50) |
| [ssl_cifar10_bntrain_raw_c666-685.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_bntrain_raw_c666-685.csv) | 666–684 (16) | 42–91 (50) |

## MMDAgg · 1,500 points per split

Tests: MMDAgg.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [ssl_cifar10_bntrain_slow_c1-125.csv](mmdagg/points_per_split_1500/ssl_cifar10_bntrain_slow_c1-125.csv) | 5–125 (25) | 42–91 (50) |
| [ssl_cifar10_bntrain_slow_c126-250.csv](mmdagg/points_per_split_1500/ssl_cifar10_bntrain_slow_c126-250.csv) | 130–250 (25) | 42–91 (50) |
| [ssl_cifar10_bntrain_slow_c251-375.csv](mmdagg/points_per_split_1500/ssl_cifar10_bntrain_slow_c251-375.csv) | 255–375 (25) | 42–91 (50) |
| [ssl_cifar10_bntrain_slow_c376-500.csv](mmdagg/points_per_split_1500/ssl_cifar10_bntrain_slow_c376-500.csv) | 380–500 (25) | 42–91 (50) |
| [ssl_cifar10_bntrain_slow_c501-625.csv](mmdagg/points_per_split_1500/ssl_cifar10_bntrain_slow_c501-625.csv) | 505–625 (25) | 42–91 (50) |
| [ssl_cifar10_bntrain_slow_c626-750.csv](mmdagg/points_per_split_1500/ssl_cifar10_bntrain_slow_c626-750.csv) | 630–750 (25) | 42–91 (50) |
| [ssl_cifar10_bntrain_slow_c751-875.csv](mmdagg/points_per_split_1500/ssl_cifar10_bntrain_slow_c751-875.csv) | 755–875 (25) | 42–91 (50) |
| [ssl_cifar10_bntrain_slow_c801-804.csv](mmdagg/points_per_split_1500/ssl_cifar10_bntrain_slow_c801-804.csv) | 801–804 (4) | 42–91 (50) |
| [ssl_cifar10_bntrain_slow_c806-835.csv](mmdagg/points_per_split_1500/ssl_cifar10_bntrain_slow_c806-835.csv) | 806–834 (24) | 42–91 (50) |
| [ssl_cifar10_bntrain_slow_c836-865.csv](mmdagg/points_per_split_1500/ssl_cifar10_bntrain_slow_c836-865.csv) | 836–864 (24) | 42–91 (50) |
| [ssl_cifar10_bntrain_slow_c866-895.csv](mmdagg/points_per_split_1500/ssl_cifar10_bntrain_slow_c866-895.csv) | 866–894 (24) | 42–91 (50) |
| [ssl_cifar10_bntrain_slow_c876-1000.csv](mmdagg/points_per_split_1500/ssl_cifar10_bntrain_slow_c876-1000.csv) | 880–1000 (25) | 42–91 (50) |
| [ssl_cifar10_bntrain_slow_c896-925.csv](mmdagg/points_per_split_1500/ssl_cifar10_bntrain_slow_c896-925.csv) | 896–924 (24) | 42–91 (50) |
| [ssl_cifar10_bntrain_slow_c926-955.csv](mmdagg/points_per_split_1500/ssl_cifar10_bntrain_slow_c926-955.csv) | 926–954 (24) | 42–91 (50) |
| [ssl_cifar10_bntrain_slow_c956-985.csv](mmdagg/points_per_split_1500/ssl_cifar10_bntrain_slow_c956-985.csv) | 956–984 (24) | 42–91 (50) |
| [ssl_cifar10_bntrain_slow_c986-1000.csv](mmdagg/points_per_split_1500/ssl_cifar10_bntrain_slow_c986-1000.csv) | 986–999 (12) | 42–91 (50) |

## MMDAgg · 5,000 points per split

[Open 218 CSV files with checkpoint and seed ranges](mmdagg/points_per_split_5000).

## Schilling and Henze · Multiple budgets in each CSV

Tests: Henze, Schilling.

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [ssl_cifar10_bntrain_knn_c1-100.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c1-100.csv) | 5–100 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c101-200.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c101-200.csv) | 105–200 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c201-300.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c201-300.csv) | 205–300 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c301-400.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c301-400.csv) | 305–400 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c401-500.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c401-500.csv) | 405–500 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c501-600.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c501-600.csv) | 505–600 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c601-700.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c601-700.csv) | 605–700 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c641-644.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c641-644.csv) | 641–644 (4) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c642-666.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c642-666.csv) | 646–666 (17) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c667-691.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c667-691.csv) | 667–691 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c692-716.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c692-716.csv) | 692–716 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c701-800.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c701-800.csv) | 705–800 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c717-741.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c717-741.csv) | 717–741 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c742-766.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c742-766.csv) | 742–766 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c767-791.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c767-791.csv) | 767–791 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c792-816.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c792-816.csv) | 792–816 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c801-900.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c801-900.csv) | 805–900 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c817-841.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c817-841.csv) | 817–841 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c842-866.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c842-866.csv) | 842–866 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c867-891.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c867-891.csv) | 867–891 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c892-916.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c892-916.csv) | 892–916 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c901-1000.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c901-1000.csv) | 905–1000 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c917-941.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c917-941.csv) | 917–939 (19) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c941-944.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c941-944.csv) | 941–944 (4) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c942-966.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c942-966.csv) | 946–966 (13) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c951-954.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c951-954.csv) | 951–954 (4) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c967-991.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c967-991.csv) | 967–991 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_knn_c992-1000.csv](schilling_henze/multiple_budgets/ssl_cifar10_bntrain_knn_c992-1000.csv) | 992–999 (7) | 42–91 (50) |

## VRT · Multiple budgets in each CSV

Tests: VRT, VRT (low-reference variant).

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [ssl_cifar10_bntrain_vrt_c481-500.csv (empty job shard)](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c481-500.csv) | — | — |
| [ssl_cifar10_bntrain_vrt_c501-520.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c501-520.csv) | 501–520 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c521-540.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c521-540.csv) | 521–540 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c541-560.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c541-560.csv) | 541–560 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c561-580.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c561-580.csv) | 561–580 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c581-600.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c581-600.csv) | 581–600 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c601-620.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c601-620.csv) | 601–620 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c621-640.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c621-640.csv) | 621–640 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c641-660.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c641-660.csv) | 641–660 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c661-680.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c661-680.csv) | 661–680 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c681-700.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c681-700.csv) | 681–700 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c701-720.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c701-720.csv) | 701–720 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c721-740.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c721-740.csv) | 721–740 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c741-760.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c741-760.csv) | 741–760 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c761-780.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c761-780.csv) | 761–780 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c781-800.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c781-800.csv) | 781–800 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c801-820.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c801-820.csv) | 801–820 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c821-840.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c821-840.csv) | 821–840 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c841-860.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c841-860.csv) | 841–860 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c861-880.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c861-880.csv) | 861–880 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c881-900.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c881-900.csv) | 881–900 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c901-920.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c901-920.csv) | 901–920 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c921-940.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c921-940.csv) | 921–940 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c941-960.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c941-960.csv) | 941–960 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c961-980.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c961-980.csv) | 961–980 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_vrt_c981-1000.csv](vrt/multiple_budgets/ssl_cifar10_bntrain_vrt_c981-1000.csv) | 981–1000 (20) | 42–91 (50) |

## VRT, MMD, ED and C2ST · Multiple budgets in each CSV

Tests: C2ST, ED, ED (raw features), MMD, MMD (raw features), VRT, VRT (low-reference variant).

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [ssl_cifar10_bntrain_fast_c1-20.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c1-20.csv) | 1–20 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c21-40.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c21-40.csv) | 21–40 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c41-60.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c41-60.csv) | 41–60 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c61-80.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c61-80.csv) | 61–80 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c81-100.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c81-100.csv) | 81–100 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c101-120.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c101-120.csv) | 101–120 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c121-140.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c121-140.csv) | 121–140 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c141-160.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c141-160.csv) | 141–160 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c161-180.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c161-180.csv) | 161–180 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c181-200.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c181-200.csv) | 181–200 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c201-220.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c201-220.csv) | 201–220 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c221-240.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c221-240.csv) | 221–240 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c241-260.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c241-260.csv) | 241–260 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c261-280.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c261-280.csv) | 261–280 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c281-300.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c281-300.csv) | 281–300 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c301-320.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c301-320.csv) | 301–320 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c321-340.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c321-340.csv) | 321–340 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c341-360.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c341-360.csv) | 341–360 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c361-380.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c361-380.csv) | 361–380 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c381-400.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c381-400.csv) | 381–400 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c401-420.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c401-420.csv) | 401–420 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c421-440.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c421-440.csv) | 421–440 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c441-460.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c441-460.csv) | 441–460 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c461-480.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c461-480.csv) | 461–480 (20) | 42–91 (50) |
| [ssl_cifar10_bntrain_fast_c481-500.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_bntrain_fast_c481-500.csv) | 481–500 (20) | 42–91 (50) |
