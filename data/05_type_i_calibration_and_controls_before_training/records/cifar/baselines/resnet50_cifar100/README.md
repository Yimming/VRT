# ResNet-50 / CIFAR-100

CIFAR random-network baseline controls.

`p` is a p-value except for MMDAgg, where it records a 0/1 rejection decision.

Ranges below come from the CSV rows; the number in parentheses is the number of distinct checkpoints or seeds.

## Eval extraction (running BatchNorm statistics) · Official and random splits in each CSV

Tests: ED, Henze, MMD, Schilling.

| File | Repetitions | Test seeds |
| --- | --- | --- |
| [base_null_r50_c100_eval_gpu_r0-99.csv](eval_mode/combined_splits/base_null_r50_c100_eval_gpu_r0-99.csv) | 0–99 (100) | 100000–100099 (100) |
| [base_null_r50_c100_eval_gpu_r100-199.csv](eval_mode/combined_splits/base_null_r50_c100_eval_gpu_r100-199.csv) | 100–199 (100) | 100100–100199 (100) |
| [base_null_r50_c100_eval_gpu_r200-299.csv](eval_mode/combined_splits/base_null_r50_c100_eval_gpu_r200-299.csv) | 200–299 (100) | 100200–100299 (100) |
| [base_null_r50_c100_eval_gpu_r300-399.csv](eval_mode/combined_splits/base_null_r50_c100_eval_gpu_r300-399.csv) | 300–399 (100) | 100300–100399 (100) |
| [base_null_r50_c100_eval_gpu_r400-499.csv](eval_mode/combined_splits/base_null_r50_c100_eval_gpu_r400-499.csv) | 400–499 (100) | 100400–100499 (100) |

## Eval extraction (running BatchNorm statistics) · Official training/test split

Tests: C2ST, MMDAgg.

Points per split: 5,000, 10,000.

| File | Repetitions | Test seeds |
| --- | --- | --- |
| [base_null_r50_c100_eval_cpu_r0-9_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r0-9_official.csv) | 0–9 (10) | 100000–100009 (10) |
| [base_null_r50_c100_eval_cpu_r10-19_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r10-19_official.csv) | 10–19 (10) | 100010–100019 (10) |
| [base_null_r50_c100_eval_cpu_r20-29_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r20-29_official.csv) | 20–29 (10) | 100020–100029 (10) |
| [base_null_r50_c100_eval_cpu_r30-39_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r30-39_official.csv) | 30–39 (10) | 100030–100039 (10) |
| [base_null_r50_c100_eval_cpu_r40-49_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r40-49_official.csv) | 40–49 (10) | 100040–100049 (10) |
| [base_null_r50_c100_eval_cpu_r50-59_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r50-59_official.csv) | 50–59 (10) | 100050–100059 (10) |
| [base_null_r50_c100_eval_cpu_r60-69_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r60-69_official.csv) | 60–69 (10) | 100060–100069 (10) |
| [base_null_r50_c100_eval_cpu_r70-79_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r70-79_official.csv) | 70–79 (10) | 100070–100079 (10) |
| [base_null_r50_c100_eval_cpu_r80-89_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r80-89_official.csv) | 80–89 (10) | 100080–100089 (10) |
| [base_null_r50_c100_eval_cpu_r90-99_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r90-99_official.csv) | 90–99 (10) | 100090–100099 (10) |
| [base_null_r50_c100_eval_cpu_r100-109_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r100-109_official.csv) | 100–109 (10) | 100100–100109 (10) |
| [base_null_r50_c100_eval_cpu_r110-119_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r110-119_official.csv) | 110–119 (10) | 100110–100119 (10) |
| [base_null_r50_c100_eval_cpu_r120-129_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r120-129_official.csv) | 120–129 (10) | 100120–100129 (10) |
| [base_null_r50_c100_eval_cpu_r130-139_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r130-139_official.csv) | 130–139 (10) | 100130–100139 (10) |
| [base_null_r50_c100_eval_cpu_r140-149_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r140-149_official.csv) | 140–149 (10) | 100140–100149 (10) |
| [base_null_r50_c100_eval_cpu_r150-159_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r150-159_official.csv) | 150–159 (10) | 100150–100159 (10) |
| [base_null_r50_c100_eval_cpu_r160-169_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r160-169_official.csv) | 160–169 (10) | 100160–100169 (10) |
| [base_null_r50_c100_eval_cpu_r170-179_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r170-179_official.csv) | 170–179 (10) | 100170–100179 (10) |
| [base_null_r50_c100_eval_cpu_r180-189_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r180-189_official.csv) | 180–189 (10) | 100180–100189 (10) |
| [base_null_r50_c100_eval_cpu_r190-199_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r190-199_official.csv) | 190–199 (10) | 100190–100199 (10) |
| [base_null_r50_c100_eval_cpu_r200-209_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r200-209_official.csv) | 200–209 (10) | 100200–100209 (10) |
| [base_null_r50_c100_eval_cpu_r210-219_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r210-219_official.csv) | 210–219 (10) | 100210–100219 (10) |
| [base_null_r50_c100_eval_cpu_r220-229_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r220-229_official.csv) | 220–229 (10) | 100220–100229 (10) |
| [base_null_r50_c100_eval_cpu_r230-239_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r230-239_official.csv) | 230–239 (10) | 100230–100239 (10) |
| [base_null_r50_c100_eval_cpu_r240-249_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r240-249_official.csv) | 240–249 (10) | 100240–100249 (10) |
| [base_null_r50_c100_eval_cpu_r250-259_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r250-259_official.csv) | 250–259 (10) | 100250–100259 (10) |
| [base_null_r50_c100_eval_cpu_r260-269_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r260-269_official.csv) | 260–269 (10) | 100260–100269 (10) |
| [base_null_r50_c100_eval_cpu_r270-279_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r270-279_official.csv) | 270–279 (10) | 100270–100279 (10) |
| [base_null_r50_c100_eval_cpu_r280-289_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r280-289_official.csv) | 280–289 (10) | 100280–100289 (10) |
| [base_null_r50_c100_eval_cpu_r290-299_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r290-299_official.csv) | 290–299 (10) | 100290–100299 (10) |
| [base_null_r50_c100_eval_cpu_r300-309_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r300-309_official.csv) | 300–309 (10) | 100300–100309 (10) |
| [base_null_r50_c100_eval_cpu_r310-319_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r310-319_official.csv) | 310–319 (10) | 100310–100319 (10) |
| [base_null_r50_c100_eval_cpu_r320-329_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r320-329_official.csv) | 320–329 (10) | 100320–100329 (10) |
| [base_null_r50_c100_eval_cpu_r330-339_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r330-339_official.csv) | 330–339 (10) | 100330–100339 (10) |
| [base_null_r50_c100_eval_cpu_r340-349_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r340-349_official.csv) | 340–349 (10) | 100340–100349 (10) |
| [base_null_r50_c100_eval_cpu_r350-359_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r350-359_official.csv) | 350–359 (10) | 100350–100359 (10) |
| [base_null_r50_c100_eval_cpu_r360-369_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r360-369_official.csv) | 360–369 (10) | 100360–100369 (10) |
| [base_null_r50_c100_eval_cpu_r370-379_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r370-379_official.csv) | 370–379 (10) | 100370–100379 (10) |
| [base_null_r50_c100_eval_cpu_r380-389_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r380-389_official.csv) | 380–389 (10) | 100380–100389 (10) |
| [base_null_r50_c100_eval_cpu_r390-399_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r390-399_official.csv) | 390–399 (10) | 100390–100399 (10) |
| [base_null_r50_c100_eval_cpu_r400-409_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r400-409_official.csv) | 400–409 (10) | 100400–100409 (10) |
| [base_null_r50_c100_eval_cpu_r410-419_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r410-419_official.csv) | 410–419 (10) | 100410–100419 (10) |
| [base_null_r50_c100_eval_cpu_r420-429_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r420-429_official.csv) | 420–429 (10) | 100420–100429 (10) |
| [base_null_r50_c100_eval_cpu_r430-439_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r430-439_official.csv) | 430–439 (10) | 100430–100439 (10) |
| [base_null_r50_c100_eval_cpu_r440-449_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r440-449_official.csv) | 440–449 (10) | 100440–100449 (10) |
| [base_null_r50_c100_eval_cpu_r450-459_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r450-459_official.csv) | 450–459 (10) | 100450–100459 (10) |
| [base_null_r50_c100_eval_cpu_r460-469_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r460-469_official.csv) | 460–469 (10) | 100460–100469 (10) |
| [base_null_r50_c100_eval_cpu_r470-479_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r470-479_official.csv) | 470–479 (10) | 100470–100479 (10) |
| [base_null_r50_c100_eval_cpu_r480-489_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r480-489_official.csv) | 480–489 (10) | 100480–100489 (10) |
| [base_null_r50_c100_eval_cpu_r490-499_official.csv](eval_mode/official_split/base_null_r50_c100_eval_cpu_r490-499_official.csv) | 490–499 (10) | 100490–100499 (10) |

## Eval extraction (running BatchNorm statistics) · Random training/test splits

Tests: C2ST, MMDAgg.

Points per split: 5,000, 10,000.

| File | Repetitions | Test seeds |
| --- | --- | --- |
| [base_null_r50_c100_eval_cpu_r0-9.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r0-9.csv) | 0–9 (10) | 100000–100009 (10) |
| [base_null_r50_c100_eval_cpu_r10-19.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r10-19.csv) | 10–19 (10) | 100010–100019 (10) |
| [base_null_r50_c100_eval_cpu_r20-29.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r20-29.csv) | 20–29 (10) | 100020–100029 (10) |
| [base_null_r50_c100_eval_cpu_r30-39.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r30-39.csv) | 30–39 (10) | 100030–100039 (10) |
| [base_null_r50_c100_eval_cpu_r40-49.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r40-49.csv) | 40–49 (10) | 100040–100049 (10) |
| [base_null_r50_c100_eval_cpu_r50-59.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r50-59.csv) | 50–59 (10) | 100050–100059 (10) |
| [base_null_r50_c100_eval_cpu_r60-69.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r60-69.csv) | 60–69 (10) | 100060–100069 (10) |
| [base_null_r50_c100_eval_cpu_r70-79.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r70-79.csv) | 70–79 (10) | 100070–100079 (10) |
| [base_null_r50_c100_eval_cpu_r80-89.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r80-89.csv) | 80–89 (10) | 100080–100089 (10) |
| [base_null_r50_c100_eval_cpu_r90-99.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r90-99.csv) | 90–99 (10) | 100090–100099 (10) |
| [base_null_r50_c100_eval_cpu_r100-109.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r100-109.csv) | 100–109 (10) | 100100–100109 (10) |
| [base_null_r50_c100_eval_cpu_r110-119.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r110-119.csv) | 110–119 (10) | 100110–100119 (10) |
| [base_null_r50_c100_eval_cpu_r120-129.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r120-129.csv) | 120–129 (10) | 100120–100129 (10) |
| [base_null_r50_c100_eval_cpu_r130-139.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r130-139.csv) | 130–139 (10) | 100130–100139 (10) |
| [base_null_r50_c100_eval_cpu_r140-149.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r140-149.csv) | 140–149 (10) | 100140–100149 (10) |
| [base_null_r50_c100_eval_cpu_r150-159.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r150-159.csv) | 150–159 (10) | 100150–100159 (10) |
| [base_null_r50_c100_eval_cpu_r160-169.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r160-169.csv) | 160–169 (10) | 100160–100169 (10) |
| [base_null_r50_c100_eval_cpu_r170-179.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r170-179.csv) | 170–179 (10) | 100170–100179 (10) |
| [base_null_r50_c100_eval_cpu_r180-189.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r180-189.csv) | 180–189 (10) | 100180–100189 (10) |
| [base_null_r50_c100_eval_cpu_r190-199.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r190-199.csv) | 190–199 (10) | 100190–100199 (10) |
| [base_null_r50_c100_eval_cpu_r200-209.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r200-209.csv) | 200–209 (10) | 100200–100209 (10) |
| [base_null_r50_c100_eval_cpu_r210-219.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r210-219.csv) | 210–219 (10) | 100210–100219 (10) |
| [base_null_r50_c100_eval_cpu_r220-229.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r220-229.csv) | 220–229 (10) | 100220–100229 (10) |
| [base_null_r50_c100_eval_cpu_r230-239.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r230-239.csv) | 230–239 (10) | 100230–100239 (10) |
| [base_null_r50_c100_eval_cpu_r240-249.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r240-249.csv) | 240–249 (10) | 100240–100249 (10) |
| [base_null_r50_c100_eval_cpu_r250-259.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r250-259.csv) | 250–259 (10) | 100250–100259 (10) |
| [base_null_r50_c100_eval_cpu_r260-269.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r260-269.csv) | 260–269 (10) | 100260–100269 (10) |
| [base_null_r50_c100_eval_cpu_r270-279.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r270-279.csv) | 270–279 (10) | 100270–100279 (10) |
| [base_null_r50_c100_eval_cpu_r280-289.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r280-289.csv) | 280–289 (10) | 100280–100289 (10) |
| [base_null_r50_c100_eval_cpu_r290-299.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r290-299.csv) | 290–299 (10) | 100290–100299 (10) |
| [base_null_r50_c100_eval_cpu_r300-309.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r300-309.csv) | 300–309 (10) | 100300–100309 (10) |
| [base_null_r50_c100_eval_cpu_r310-319.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r310-319.csv) | 310–319 (10) | 100310–100319 (10) |
| [base_null_r50_c100_eval_cpu_r320-329.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r320-329.csv) | 320–329 (10) | 100320–100329 (10) |
| [base_null_r50_c100_eval_cpu_r330-339.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r330-339.csv) | 330–339 (10) | 100330–100339 (10) |
| [base_null_r50_c100_eval_cpu_r340-349.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r340-349.csv) | 340–349 (10) | 100340–100349 (10) |
| [base_null_r50_c100_eval_cpu_r350-359.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r350-359.csv) | 350–359 (10) | 100350–100359 (10) |
| [base_null_r50_c100_eval_cpu_r360-369.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r360-369.csv) | 360–369 (10) | 100360–100369 (10) |
| [base_null_r50_c100_eval_cpu_r370-379.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r370-379.csv) | 370–379 (10) | 100370–100379 (10) |
| [base_null_r50_c100_eval_cpu_r380-389.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r380-389.csv) | 380–389 (10) | 100380–100389 (10) |
| [base_null_r50_c100_eval_cpu_r390-399.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r390-399.csv) | 390–399 (10) | 100390–100399 (10) |
| [base_null_r50_c100_eval_cpu_r400-409.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r400-409.csv) | 400–409 (10) | 100400–100409 (10) |
| [base_null_r50_c100_eval_cpu_r410-419.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r410-419.csv) | 410–419 (10) | 100410–100419 (10) |
| [base_null_r50_c100_eval_cpu_r420-429.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r420-429.csv) | 420–429 (10) | 100420–100429 (10) |
| [base_null_r50_c100_eval_cpu_r430-439.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r430-439.csv) | 430–439 (10) | 100430–100439 (10) |
| [base_null_r50_c100_eval_cpu_r440-449.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r440-449.csv) | 440–449 (10) | 100440–100449 (10) |
| [base_null_r50_c100_eval_cpu_r450-459.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r450-459.csv) | 450–459 (10) | 100450–100459 (10) |
| [base_null_r50_c100_eval_cpu_r460-469.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r460-469.csv) | 460–469 (10) | 100460–100469 (10) |
| [base_null_r50_c100_eval_cpu_r470-479.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r470-479.csv) | 470–479 (10) | 100470–100479 (10) |
| [base_null_r50_c100_eval_cpu_r480-489.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r480-489.csv) | 480–489 (10) | 100480–100489 (10) |
| [base_null_r50_c100_eval_cpu_r490-499.csv](eval_mode/random_splits/base_null_r50_c100_eval_cpu_r490-499.csv) | 490–499 (10) | 100490–100499 (10) |
