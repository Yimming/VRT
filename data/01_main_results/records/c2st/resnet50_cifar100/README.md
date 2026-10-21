# ResNet-50 / CIFAR-100

C2ST.

The CSV `p` column contains per-test p-values.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## Eval extraction (running BatchNorm statistics) · Multiple budgets in each CSV

Points per split: 1,500, 10,000.

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [c2st_r50_b1500-10000_c0-76.csv](eval_mode/multiple_budgets/c2st_r50_b1500-10000_c0-76.csv) | 0–76 (77) | 42–91 (50) |
| [c2st_r50_b1500-10000_c77-153.csv](eval_mode/multiple_budgets/c2st_r50_b1500-10000_c77-153.csv) | 77–153 (77) | 42–91 (50) |
| [c2st_r50_b1500-10000_c154-200.csv](eval_mode/multiple_budgets/c2st_r50_b1500-10000_c154-200.csv) | 154–200 (46) | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 10,000 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [c2st_r50_bntrain_b10000_c0-4.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c0-4.csv) | 0–4 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c5-9.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c5-9.csv) | 5–9 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c10-14.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c10-14.csv) | 10–14 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c15-19.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c15-19.csv) | 15–19 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c20-24.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c20-24.csv) | 20–24 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c25-29.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c25-29.csv) | 25–29 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c30-34.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c30-34.csv) | 30–34 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c35-39.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c35-39.csv) | 35–39 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c40-44.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c40-44.csv) | 40–44 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c45-49.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c45-49.csv) | 45–49 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c50-54.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c50-54.csv) | 50–54 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c55-59.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c55-59.csv) | 55–59 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c60-64.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c60-64.csv) | 60–64 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c65-69.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c65-69.csv) | 65–69 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c70-74.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c70-74.csv) | 70–74 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c75-79.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c75-79.csv) | 75–79 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c80-84.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c80-84.csv) | 80–84 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c85-89.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c85-89.csv) | 85–89 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c90-94.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c90-94.csv) | 90–94 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c95-99.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c95-99.csv) | 95–99 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c100-104.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c100-104.csv) | 100–104 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c105-109.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c105-109.csv) | 105–109 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c110-114.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c110-114.csv) | 110–114 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c115-119.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c115-119.csv) | 115–119 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c120-124.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c120-124.csv) | 120–124 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c125-129.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c125-129.csv) | 125–129 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c130-134.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c130-134.csv) | 130–134 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c135-139.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c135-139.csv) | 135–139 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c140-144.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c140-144.csv) | 140–144 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c145-149.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c145-149.csv) | 145–149 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c150-154.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c150-154.csv) | 150–154 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c155-159.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c155-159.csv) | 155–159 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c160-164.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c160-164.csv) | 160–164 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c165-169.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c165-169.csv) | 165–169 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c170-174.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c170-174.csv) | 170–174 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c175-179.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c175-179.csv) | 175–179 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c180-184.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c180-184.csv) | 180–184 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c185-189.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c185-189.csv) | 185–189 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c190-194.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c190-194.csv) | 190–194 (5) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c195-199.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c195-199.csv) | 195–198 (4) | 42–91 (50) |
| [c2st_r50_bntrain_b10000_c200-204.csv](train_mode/points_per_split_10000/c2st_r50_bntrain_b10000_c200-204.csv) | 200 | 42–91 (50) |

## Train extraction (current-batch BatchNorm statistics) · 1,500 points per split

| File | Checkpoints | Test seeds |
| --- | --- | --- |
| [c2st_r50_bntrain_b1500_c0-200.csv](train_mode/points_per_split_1500/c2st_r50_bntrain_b1500_c0-200.csv) | 0–200 (200) | 42–91 (50) |
