# SimCLR ResNet-50 / CIFAR-100

SimCLR: train-mode extraction.

`p` is a p-value except for MMDAgg, where it records a 0/1 rejection decision.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## MMD, ED and C2ST · Multiple budgets in each CSV

Tests: C2ST, ED, ED (raw features), MMD, MMD (raw features).

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [ssl_cifar100_bntrain_base_c1-100.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c1-100.csv) | 5–100 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c101-200.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c101-200.csv) | 105–200 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c201-300.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c201-300.csv) | 205–300 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c301-400.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c301-400.csv) | 305–400 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c401-500.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c401-500.csv) | 405–500 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c501-600.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c501-600.csv) | 505–600 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c576-579.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c576-579.csv) | 576–579 (4) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c580-604.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c580-604.csv) | 581–604 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c601-700.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c601-700.csv) | 605–700 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c605-629.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c605-629.csv) | 606–629 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c630-654.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c630-654.csv) | 631–654 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c655-679.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c655-679.csv) | 656–679 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c680-704.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c680-704.csv) | 681–704 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c701-800.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c701-800.csv) | 705–800 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c705-729.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c705-729.csv) | 706–729 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c730-754.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c730-754.csv) | 731–754 (16) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c741-744.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c741-744.csv) | 741–744 (4) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c755-779.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c755-779.csv) | 756–779 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c780-804.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c780-804.csv) | 781–799 (16) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c801-804.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c801-804.csv) | 801–804 (4) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c801-900.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c801-900.csv) | 805–900 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c805-829.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c805-829.csv) | 806–829 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c830-854.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c830-854.csv) | 831–854 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c855-879.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c855-879.csv) | 856–879 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c880-904.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c880-904.csv) | 881–904 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c901-1000.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c901-1000.csv) | 905–1000 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c905-929.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c905-929.csv) | 906–929 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c930-954.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c930-954.csv) | 931–954 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c955-979.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c955-979.csv) | 956–979 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_base_c980-1000.csv](mmd_ed_c2st/multiple_budgets/ssl_cifar100_bntrain_base_c980-1000.csv) | 981–999 (16) | 42–91 (50) |

## MMD and ED on raw features · 10,000 points per split

Tests: ED (raw features), MMD (raw features).

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [ssl_cifar100_bntrain_raw_c401-404.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar100_bntrain_raw_c401-404.csv) | 401–404 (4) | 42–91 (50) |
| [ssl_cifar100_bntrain_raw_c406-430.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar100_bntrain_raw_c406-430.csv) | 406–429 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_raw_c431-455.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar100_bntrain_raw_c431-455.csv) | 431–454 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_raw_c456-480.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar100_bntrain_raw_c456-480.csv) | 456–479 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_raw_c481-505.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar100_bntrain_raw_c481-505.csv) | 481–504 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_raw_c506-530.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar100_bntrain_raw_c506-530.csv) | 506–529 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_raw_c531-555.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar100_bntrain_raw_c531-555.csv) | 531–554 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_raw_c556-575.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar100_bntrain_raw_c556-575.csv) | 556–574 (16) | 42–91 (50) |

## MMDAgg · 1,500 points per split

Tests: MMDAgg.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [ssl_cifar100_bntrain_slow_c1-125.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c1-125.csv) | 5–125 (25) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c126-250.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c126-250.csv) | 130–250 (25) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c251-375.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c251-375.csv) | 255–375 (25) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c376-500.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c376-500.csv) | 380–500 (25) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c501-625.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c501-625.csv) | 505–625 (25) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c626-750.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c626-750.csv) | 630–750 (25) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c676-679.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c676-679.csv) | 676–679 (4) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c681-710.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c681-710.csv) | 681–709 (24) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c711-740.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c711-740.csv) | 711–739 (24) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c741-770.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c741-770.csv) | 741–769 (24) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c751-875.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c751-875.csv) | 755–875 (25) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c771-800.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c771-800.csv) | 771–799 (24) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c801-830.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c801-830.csv) | 801–829 (24) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c831-860.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c831-860.csv) | 831–859 (24) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c861-890.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c861-890.csv) | 861–889 (24) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c876-1000.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c876-1000.csv) | 880–1000 (25) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c891-920.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c891-920.csv) | 891–919 (24) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c921-950.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c921-950.csv) | 921–949 (24) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c951-980.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c951-980.csv) | 951–979 (24) | 42–91 (50) |
| [ssl_cifar100_bntrain_slow_c981-1000.csv](mmdagg/points_per_split_1500/ssl_cifar100_bntrain_slow_c981-1000.csv) | 981–999 (16) | 42–91 (50) |

## MMDAgg · 5,000 points per split

[Open 283 CSV files with checkpoint and seed ranges](mmdagg/points_per_split_5000).

## Schilling and Henze · Multiple budgets in each CSV

Tests: Henze, Schilling.

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [ssl_cifar100_bntrain_knn_c1-100.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c1-100.csv) | 5–100 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c101-200.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c101-200.csv) | 105–200 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c201-300.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c201-300.csv) | 205–300 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c301-400.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c301-400.csv) | 305–400 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c401-500.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c401-500.csv) | 405–500 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c501-600.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c501-600.csv) | 505–600 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c601-700.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c601-700.csv) | 605–700 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c641-644.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c641-644.csv) | 641–644 (4) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c642-666.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c642-666.csv) | 646–666 (17) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c667-691.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c667-691.csv) | 667–691 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c692-716.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c692-716.csv) | 692–716 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c701-800.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c701-800.csv) | 705–800 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c717-741.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c717-741.csv) | 717–741 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c742-766.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c742-766.csv) | 742–766 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c767-791.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c767-791.csv) | 767–791 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c792-816.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c792-816.csv) | 792–816 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c801-900.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c801-900.csv) | 805–900 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c817-841.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c817-841.csv) | 817–841 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c842-866.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c842-866.csv) | 842–866 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c867-891.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c867-891.csv) | 867–891 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c892-916.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c892-916.csv) | 892–916 (16) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c896-899.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c896-899.csv) | 896–899 (4) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c901-1000.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c901-1000.csv) | 905–1000 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c917-941.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c917-941.csv) | 917–941 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c942-966.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c942-966.csv) | 942–966 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c967-991.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c967-991.csv) | 967–991 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_knn_c992-1000.csv](schilling_henze/multiple_budgets/ssl_cifar100_bntrain_knn_c992-1000.csv) | 992–999 (7) | 42–91 (50) |

## VRT · Multiple budgets in each CSV

Tests: VRT, VRT (low-reference variant).

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [ssl_cifar100_bntrain_vrt_c1-20.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c1-20.csv) | 1–20 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c21-40.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c21-40.csv) | 21–40 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c41-60.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c41-60.csv) | 41–60 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c61-80.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c61-80.csv) | 61–80 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c81-100.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c81-100.csv) | 81–100 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c101-120.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c101-120.csv) | 101–120 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c121-140.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c121-140.csv) | 121–140 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c141-160.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c141-160.csv) | 141–160 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c161-180.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c161-180.csv) | 161–180 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c181-200.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c181-200.csv) | 181–200 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c201-220.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c201-220.csv) | 201–220 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c221-240.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c221-240.csv) | 221–240 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c241-260.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c241-260.csv) | 241–260 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c261-280.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c261-280.csv) | 261–280 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c281-300.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c281-300.csv) | 281–300 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c301-320.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c301-320.csv) | 301–320 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c321-340.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c321-340.csv) | 321–340 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c341-360.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c341-360.csv) | 341–360 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c361-380.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c361-380.csv) | 361–380 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c381-400.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c381-400.csv) | 381–400 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c401-420.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c401-420.csv) | 401–420 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c421-440.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c421-440.csv) | 421–440 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c441-460.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c441-460.csv) | 441–460 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c461-480.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c461-480.csv) | 461–480 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c481-500.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c481-500.csv) | 481–500 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c501-520.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c501-520.csv) | 501–520 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c521-540.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c521-540.csv) | 521–540 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c541-560.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c541-560.csv) | 541–560 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c561-580.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c561-580.csv) | 561–580 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c581-600.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c581-600.csv) | 581–600 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c601-620.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c601-620.csv) | 601–620 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c621-640.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c621-640.csv) | 621–640 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c641-660.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c641-660.csv) | 641–660 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c661-680.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c661-680.csv) | 661–680 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c681-700.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c681-700.csv) | 681–700 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c701-720.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c701-720.csv) | 701–720 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c721-740.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c721-740.csv) | 721–740 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c741-760.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c741-760.csv) | 741–760 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c761-780.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c761-780.csv) | 761–780 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c781-800.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c781-800.csv) | 781–800 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c801-820.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c801-820.csv) | 801–820 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c821-840.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c821-840.csv) | 821–840 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c841-860.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c841-860.csv) | 841–860 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c861-880.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c861-880.csv) | 861–880 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c881-900.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c881-900.csv) | 881–900 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c901-920.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c901-920.csv) | 901–920 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c921-940.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c921-940.csv) | 921–940 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c941-960.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c941-960.csv) | 941–960 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c961-980.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c961-980.csv) | 961–980 (20) | 42–91 (50) |
| [ssl_cifar100_bntrain_vrt_c981-1000.csv](vrt/multiple_budgets/ssl_cifar100_bntrain_vrt_c981-1000.csv) | 981–1000 (20) | 42–91 (50) |
