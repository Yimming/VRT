# SimCLR ResNet-50 / CIFAR-100 (earlier runs)

SimCLR: MMDAgg at 5,000 points and earlier runs.

These earlier runs, including later seed-completion files, are retained for reference and excluded from the paper's final onset summaries.

`p` is a p-value except for MMDAgg, where it records a 0/1 rejection decision.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## Files

| File | Checkpoints | Test seeds | Tests | Points per split |
| --- | --- | --- | --- | --- |
| [sslbase_cifar100_1-50.csv](sslbase_cifar100_1-50.csv) | 1–50 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslvrt_cifar100_1-125.csv](sslvrt_cifar100_1-125.csv) | 1–125 (125) | 42–51 (10) | VRT (low-reference variant) | 1,500 |
| [sslbase_cifar100_51-100.csv](sslbase_cifar100_51-100.csv) | 51–100 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslbase_cifar100_101-150.csv](sslbase_cifar100_101-150.csv) | 101–150 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslvrt_cifar100_126-250.csv](sslvrt_cifar100_126-250.csv) | 126–250 (125) | 42–51 (10) | VRT (low-reference variant) | 1,500 |
| [sslbase_cifar100_151-200.csv](sslbase_cifar100_151-200.csv) | 151–186 (36) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslseeds_cifar100_170-202.csv](sslseeds_cifar100_170-202.csv) | 170–202 (33) | 52–91 (40) | ED, MMD, VRT | 10,000 |
| [sslbase_cifar100_187-193.csv](sslbase_cifar100_187-193.csv) | 187–193 (7) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslbase_cifar100_194-200.csv](sslbase_cifar100_194-200.csv) | 194–200 (7) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslbase_cifar100_201-250.csv](sslbase_cifar100_201-250.csv) | 201–237 (37) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslseeds_cifar100_203-235.csv](sslseeds_cifar100_203-235.csv) | 203–235 (33) | 52–91 (40) | ED, MMD, VRT | 10,000 |
| [sslseeds_cifar100_236-268.csv](sslseeds_cifar100_236-268.csv) | 236–268 (33) | 52–91 (40) | ED, MMD, VRT | 10,000 |
| [sslbase_cifar100_237-243.csv](sslbase_cifar100_237-243.csv) | 237–243 (7) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslbase_cifar100_244-250.csv](sslbase_cifar100_244-250.csv) | 244–250 (7) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslbase_cifar100_251-300.csv](sslbase_cifar100_251-300.csv) | 251–300 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslvrt_cifar100_251-375.csv](sslvrt_cifar100_251-375.csv) | 251–375 (125) | 42–51 (10) | VRT (low-reference variant) | 1,500 |
| [sslseeds_cifar100_269-300.csv](sslseeds_cifar100_269-300.csv) | 269–300 (32) | 52–91 (40) | ED, MMD, VRT | 10,000 |
| [sslbase_cifar100_301-350.csv](sslbase_cifar100_301-350.csv) | 301–350 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslbase_cifar100_351-400.csv](sslbase_cifar100_351-400.csv) | 351–400 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslvrt_cifar100_376-500.csv](sslvrt_cifar100_376-500.csv) | 376–500 (125) | 42–51 (10) | VRT (low-reference variant) | 1,500 |
| [sslbase_cifar100_401-450.csv](sslbase_cifar100_401-450.csv) | 401–450 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslbase_cifar100_451-500.csv](sslbase_cifar100_451-500.csv) | 451–500 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslbase_cifar100_501-550.csv](sslbase_cifar100_501-550.csv) | 501–550 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslvrt_cifar100_501-625.csv](sslvrt_cifar100_501-625.csv) | 501–625 (125) | 42–51 (10) | VRT (low-reference variant) | 1,500 |
| [sslbase_cifar100_551-600.csv](sslbase_cifar100_551-600.csv) | 551–600 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslbase_cifar100_601-650.csv](sslbase_cifar100_601-650.csv) | 601–650 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslvrt_cifar100_626-750.csv](sslvrt_cifar100_626-750.csv) | 626–750 (125) | 42–51 (10) | VRT (low-reference variant) | 1,500 |
| [sslbase_cifar100_651-700.csv](sslbase_cifar100_651-700.csv) | 651–700 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslbase_cifar100_701-750.csv](sslbase_cifar100_701-750.csv) | 701–750 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslbase_cifar100_751-800.csv](sslbase_cifar100_751-800.csv) | 751–800 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslvrt_cifar100_751-875.csv](sslvrt_cifar100_751-875.csv) | 751–875 (125) | 42–51 (10) | VRT (low-reference variant) | 1,500 |
| [sslbase_cifar100_801-850.csv](sslbase_cifar100_801-850.csv) | 801–850 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslbase_cifar100_851-900.csv](sslbase_cifar100_851-900.csv) | 851–900 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslvrt_cifar100_876-1000.csv](sslvrt_cifar100_876-1000.csv) | 876–1000 (125) | 42–51 (10) | VRT (low-reference variant) | 1,500 |
| [sslbase_cifar100_901-950.csv](sslbase_cifar100_901-950.csv) | 901–950 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
| [sslbase_cifar100_951-1000.csv](sslbase_cifar100_951-1000.csv) | 951–1000 (50) | 42–51 (10) | C2ST, ED, MMD, MMDAgg | 1,500, 8,000, 10,000 |
