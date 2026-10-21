# SimCLR ResNet-50 / CIFAR-10

SimCLR: eval-mode extraction.

`p` is a p-value except for MMDAgg, where it records a 0/1 rejection decision.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## MMD and ED on raw features · 10,000 points per split

Tests: ED (raw features), MMD (raw features).

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [ssl_cifar10_raw_c1-50.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c1-50.csv) | 1–50 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c51-100.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c51-100.csv) | 51–100 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c101-150.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c101-150.csv) | 101–150 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c151-200.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c151-200.csv) | 151–200 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c201-250.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c201-250.csv) | 201–250 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c251-300.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c251-300.csv) | 251–300 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c301-350.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c301-350.csv) | 301–350 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c351-400.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c351-400.csv) | 351–400 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c401-450.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c401-450.csv) | 401–450 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c451-500.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c451-500.csv) | 451–500 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c501-550.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c501-550.csv) | 501–550 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c551-600.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c551-600.csv) | 551–600 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c601-650.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c601-650.csv) | 601–650 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c651-700.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c651-700.csv) | 651–700 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c701-750.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c701-750.csv) | 701–750 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c751-800.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c751-800.csv) | 751–800 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c801-850.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c801-850.csv) | 801–850 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c851-900.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c851-900.csv) | 851–900 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c901-950.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c901-950.csv) | 901–950 (50) | 42–91 (50) |
| [ssl_cifar10_raw_c951-1000.csv](mmd_ed_raw_features/points_per_split_10000/ssl_cifar10_raw_c951-1000.csv) | 951–1000 (50) | 42–91 (50) |

## MMDAgg · 1,500 points per split

[Open 117 CSV files with checkpoint and seed ranges](mmdagg/points_per_split_1500).

## Reversed VRT, Schilling and Henze · Multiple budgets in each CSV

Tests: Henze, Schilling, Reversed VRT.

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [ssl_cifar10_extra_c1-100.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c1-100.csv) | 5–100 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c101-200.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c101-200.csv) | 105–200 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c201-300.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c201-300.csv) | 205–300 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c256-259.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c256-259.csv) | 256–259 (4) | 42–91 (50) |
| [ssl_cifar10_extra_c257-281.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c257-281.csv) | 261–281 (17) | 42–91 (50) |
| [ssl_cifar10_extra_c282-306.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c282-306.csv) | 282–306 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c301-400.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c301-400.csv) | 305–400 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c307-331.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c307-331.csv) | 307–331 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c332-356.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c332-356.csv) | 332–356 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c357-381.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c357-381.csv) | 357–381 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c382-406.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c382-406.csv) | 382–406 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c401-500.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c401-500.csv) | 405–500 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c407-431.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c407-431.csv) | 407–431 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c432-456.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c432-456.csv) | 432–456 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c457-481.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c457-481.csv) | 457–479 (19) | 42–91 (50) |
| [ssl_cifar10_extra_c481-484.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c481-484.csv) | 481–484 (4) | 42–91 (50) |
| [ssl_cifar10_extra_c482-506.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c482-506.csv) | 486–506 (17) | 42–91 (50) |
| [ssl_cifar10_extra_c501-600.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c501-600.csv) | 505–600 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c507-531.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c507-531.csv) | 507–531 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c532-556.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c532-556.csv) | 532–556 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c557-581.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c557-581.csv) | 557–581 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c582-606.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c582-606.csv) | 582–606 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c601-700.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c601-700.csv) | 605–700 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c607-631.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c607-631.csv) | 607–631 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c632-656.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c632-656.csv) | 632–656 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c657-681.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c657-681.csv) | 657–681 (16) | 42–91 (50) |
| [ssl_cifar10_extra_c676-679.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c676-679.csv) | 676–679 (4) | 42–91 (50) |
| [ssl_cifar10_extra_c682-706.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c682-706.csv) | 682–706 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c701-800.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c701-800.csv) | 705–800 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c707-731.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c707-731.csv) | 707–729 (19) | 42–91 (50) |
| [ssl_cifar10_extra_c731-734.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c731-734.csv) | 731–734 (4) | 42–91 (50) |
| [ssl_cifar10_extra_c732-756.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c732-756.csv) | 736–756 (17) | 42–91 (50) |
| [ssl_cifar10_extra_c757-781.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c757-781.csv) | 757–781 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c782-806.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c782-806.csv) | 782–806 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c801-900.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c801-900.csv) | 805–900 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c807-831.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c807-831.csv) | 807–831 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c832-856.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c832-856.csv) | 832–856 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c857-881.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c857-881.csv) | 857–881 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c882-906.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c882-906.csv) | 882–906 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c901-1000.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c901-1000.csv) | 905–1000 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c907-931.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c907-931.csv) | 907–931 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c932-956.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c932-956.csv) | 932–956 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c957-981.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c957-981.csv) | 957–981 (20) | 42–91 (50) |
| [ssl_cifar10_extra_c982-1000.csv](reversed_vrt_schilling_henze/multiple_budgets/ssl_cifar10_extra_c982-1000.csv) | 982–999 (15) | 42–91 (50) |

## VRT, MMD, ED and C2ST · Multiple budgets in each CSV

Tests: C2ST, ED, MMD, VRT, VRT (low-reference variant).

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [ssl_cifar10_fast_c1-20.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c1-20.csv) | 1–20 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c21-40.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c21-40.csv) | 21–40 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c41-60.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c41-60.csv) | 41–60 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c61-80.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c61-80.csv) | 61–80 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c81-100.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c81-100.csv) | 81–100 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c101-120.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c101-120.csv) | 101–120 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c121-140.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c121-140.csv) | 121–140 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c141-160.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c141-160.csv) | 141–160 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c161-180.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c161-180.csv) | 161–180 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c181-200.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c181-200.csv) | 181–200 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c201-220.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c201-220.csv) | 201–220 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c221-240.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c221-240.csv) | 221–240 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c241-260.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c241-260.csv) | 241–260 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c261-280.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c261-280.csv) | 261–280 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c281-300.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c281-300.csv) | 281–300 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c301-320.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c301-320.csv) | 301–320 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c321-340.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c321-340.csv) | 321–340 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c341-360.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c341-360.csv) | 341–360 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c361-380.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c361-380.csv) | 361–380 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c381-400.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c381-400.csv) | 381–400 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c401-420.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c401-420.csv) | 401–420 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c421-440.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c421-440.csv) | 421–440 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c441-460.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c441-460.csv) | 441–460 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c461-480.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c461-480.csv) | 461–480 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c481-500.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c481-500.csv) | 481–500 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c501-520.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c501-520.csv) | 501–520 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c521-540.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c521-540.csv) | 521–540 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c541-560.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c541-560.csv) | 541–560 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c561-580.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c561-580.csv) | 561–580 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c581-600.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c581-600.csv) | 581–600 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c601-620.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c601-620.csv) | 601–620 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c621-640.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c621-640.csv) | 621–640 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c641-660.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c641-660.csv) | 641–660 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c661-680.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c661-680.csv) | 661–680 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c681-700.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c681-700.csv) | 681–700 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c701-720.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c701-720.csv) | 701–720 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c721-740.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c721-740.csv) | 721–740 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c741-760.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c741-760.csv) | 741–760 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c761-780.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c761-780.csv) | 761–780 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c781-800.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c781-800.csv) | 781–800 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c801-820.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c801-820.csv) | 801–820 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c821-840.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c821-840.csv) | 821–840 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c841-860.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c841-860.csv) | 841–860 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c861-880.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c861-880.csv) | 861–880 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c881-900.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c881-900.csv) | 881–900 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c901-920.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c901-920.csv) | 901–920 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c921-940.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c921-940.csv) | 921–940 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c941-960.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c941-960.csv) | 941–960 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c961-980.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c961-980.csv) | 961–980 (20) | 42–91 (50) |
| [ssl_cifar10_fast_c981-1000.csv](vrt_mmd_ed_c2st/multiple_budgets/ssl_cifar10_fast_c981-1000.csv) | 981–1000 (20) | 42–91 (50) |
