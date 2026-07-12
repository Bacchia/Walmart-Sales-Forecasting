# Walmart Store Sales Forecasting

მაღაზიების ყოველკვირეული გაყიდვების პროგნოზირება: ჩვენი ML-ის ფინალური პროექტი.

პროექტი აგებულია Kaggle-ის შეჯიბრზე
[**Walmart Recruiting - Store Sales Forecasting**](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting).
მიზანია, დავაპრედიქტოთ თითოეული მაღაზიის თითოეული დეპარტამენტის ყოველკვირეული
გაყიდვები (`Weekly_Sales`) მომავალი ~39 კვირისთვის.

---

## რას ვაკეთებთ (ამოცანის აღწერა)

გვაქვს 45 მაღაზია, თითოეულში რამდენიმე დეპარტამენტი, სულ ~3300 ცალკეული დროითი მწკრივი
(Store × Dept). თითოეული მწკრივისთვის ვიცით წარსული გაყიდვები და უნდა ვიწინასწარმეტყველოთ
მომავალი.

შეფასების მეტრიკა არის **WMAE** (Weighted Mean Absolute Error), ჩვეულებრივი საშუალო
აბსოლუტური ცდომილება, ოღონდ **სადღესასწაულო კვირებს 5-ჯერ მეტი წონა აქვს**, რადგან
სწორედ მაშინ არის გაყიდვების რაოდენობა ყველაზე მაღალი და ყველაზე მნიშვნელოვანი.

```
WMAE = Σ(wᵢ · |yᵢ − ŷᵢ|) / Σ(wᵢ)      სადაც  wᵢ = 5 თუ სადღესასწაულო კვირაა, სხვა შემთხვევაში 1
```

მეტრიკა ერთ ფაილშია გატანილი და ყველა მოდელი **ერთი და იგივე** ფუნქციით ფასდება:
`src/evaluation/metrics.py -> calculate_wmae()`.

---

## მონაცემები

მონაცემები იწერება Kaggle-იდან და დევს `data/` ფოლდერში (ის `.gitignore`-შია, git-ში არ
ვინახავთ):

| ფაილი | რა არის შიგნით |
|---|---|
| `train.csv` | ისტორიული ყოველკვირეული გაყიდვები (Store, Dept, Date, Weekly_Sales, IsHoliday) |
| `test.csv` | კვირები, რომლებზეც პროგნოზი უნდა გავაკეთოთ |
| `stores.csv` | მაღაზიის ტიპი (A/B/C) და ზომა |
| `features.csv` | ტემპერატურა, საწვავის ფასი, CPI, უმუშევრობა, MarkDown აქციები |

---

## მონაცემების დამუშავება (Preprocessing)

ორი განსხვავებული გზა გვაქვს, იმის მიხედვით თუ რომელი მოდელია:

**ხის და კლასიკური მოდელებისათვის**: `src/features/preprocessing.py`
ვაერთიანებთ სამივე ცხრილს და ვაწყობთ ფიჩერებს: `Store`, `Dept`, ტემპერატურა,
`Fuel_Price_Delta` / `CPI_Delta` / `Unemployment_Delta` (კვირიდან კვირამდე ცვლილება),
MarkDown-ები, ზომა, `WeekOfYear`, `Is_Black_Friday`, `Pre_Christmas`, `IsHoliday`, `Type`.
ცარიელი მნიშვნელობები ივსება მედიანით, კატეგორიული ცვლადი One-Hot-ით.

**ნეირონული ქსელებისთვის**: `src/features/nn_preprocessing.py`
მონაცემებს ვაქცევთ „გრძელ“ ფორმატში (`unique_id`, `ds`, `y`), ვასწორებთ კვირების ბიჯს
(`W-FRI`) და გამოტოვებულ კვირებს ვავსებთ.

**ვალიდაცია:** ტრენინგისა და ვალიდაციის გაყოფა ხდება თარიღით `2012-01-01`, ანუ ვსწავლობთ
ძველ მონაცემებზე და ვამოწმებთ უფრო ახალ კვირებზე (როგორც რეალურ პროგნოზში). **ყველა
მოდელი ერთსა და იმავე Split-ზე ფასდება, რომ WMAE-ები შედარებადი იყოს.**

---

## მოდელები

ბევრი სხვადასხვა ტიპის მოდელი გავტესტეთ, რომ გვენახა, რომელი მუშაობს უკეთ ამ ამოცანაზე.

### ხის მოდელები (ჩვენი მთავარი მიდგომა)
- **XGBoost**: `notebooks/model_experiment_XGBoost.ipynb`
  Gradient Boosting-ის მოდელი, რომელიც ხეებს თანმიმდევრულად აშენებს და თითოეული ახალი ხე
  წინა ხეების შეცდომებს ასწორებს. ვტესტავთ ჰიპერპარამეტრების რამდენიმე კონფიგურაციას,
  თითო ცალკე MLflow run-ად.
- **LightGBM**: `notebooks/model_experiment_LightGBM.ipynb`
  ასევე Gradient Boosting, ოღონდ უფრო სწრაფი (leaf-wise ზრდა) და დიდ მონაცემებზე
  მოქნილი. ასევე ჰიპერპარამეტრების ძებნით.

### ანსამბლი: XGBoost × LightGBM (საბოლოო არჩეული მოდელი)
`notebooks/model_ensemble_trees.ipynb`

ორი ხის მოდელის „სუფთა“ გაერთიანება: ორივე ტრენინგდება **ერთსა და იმავე** ვალიდაციის
გაყოფაზე, ერთი და იგივე preprocessor-ის ასლებით, პროგნოზი კეთდება **ერთსა და იმავე**
სტრიქონებზე და შემდეგ ვეძებთ საუკეთესო თანაფარდობას. საუკეთესო აღმოჩნდა
**0.45 · XGBoost + 0.55 · LightGBM**.

### კლასიკური სტატისტიკური მოდელები (baseline)
- **SARIMA**: `notebooks/model_experiment_SARIMA.ipynb`
  კლასიკური სტატისტიკური მოდელი (AutoRegressive + სეზონურობა). თითოეულ მწკრივს ცალკე
  უჯდება და იჭერს ტრენდსა და სეზონურ განმეორებადობას.
- **Prophet**: `notebooks/model_experiment_Prophet.ipynb`
  Meta-ს მოდელი, რომელიც მწკრივს შლის სამ ნაწილად: ტრენდი, სეზონურობა და დღესასწაულები.
  მარტივი და სწრაფი per-series baseline.

### ნეირონული ქსელები
- **DLinear**: `notebooks/model_experiment_DLinear.ipynb`
  მარტივი ხაზოვანი მოდელი, რომელიც მწკრივს ყოფს ტრენდად და სეზონურ ნაწილად და თითოეულს
  ცალკე პროგნოზირებს. მარტივია, მაგრამ ხშირად ძალიან ძლიერი baseline-ია.
- **N-BEATS**: `notebooks/model_experiment_NBEATS.ipynb`
  ღრმა ნეირონული ქსელი, აწყობილი fully-connected ბლოკებისგან (backcast/forecast).
  გლობალური მოდელია, ერთდროულად ყველა მწკრივზე სწავლობს.
- **PatchTST**: `notebooks/model_experiment_PatchTST.ipynb`
  Transformer, რომელიც მწკრივს ჭრის პატარა „Patch“-ებად (მონაკვეთებად) და მათ ამუშავებს
  ტოკენების მსგავსად.
- **TFT** (Temporal Fusion Transformer): `notebooks/model_experiment_TFT.ipynb`
  Attention-ზე დაფუძნებული Transformer, რომელსაც დამატებითი ფიჩერების (exogenous)
  გამოყენებაც შეუძლია და შედარებით ინტერპრეტირებადია.

### Foundation მოდელი: TimesFM (ბონუსი)
`notebooks/model_experiment_TimesFM.ipynb`

TimesFM არის Google-ის ახალი **pretrained** დროითი მწკრივების მოდელი. ის პროგნოზს აკეთებს
**zero-shot**-ზე დაყრდნობით, რაც ნიშნავს, რომ ჩვენს მონაცემებზე საერთოდ არ ვატრენინგებთ, უბრალოდ თითოეული მწკრივის წარსულს
ვაძლევთ.

---

## შედეგები

WMAE ვალიდაციაზე (რაც ნაკლებია, მით უკეთესი) და Kaggle-ის ქულა:

| მოდელი | Validation WMAE | Kaggle (Public) |
|---|---|---|
| TimesFM (ბონუსი) | **1,635** | **2,969** |
| **XGBoost × LightGBM (არჩეული)** | **1,932** | **3,152** |
| LightGBM | 1,971 | — |
| XGBoost | ~1,985 | — |
| N-BEATS | ~4,143 | — |

დანარჩენი მოდელების run-ები და მეტრიკები DagsHub-ის MLflow-ზეა შენახული.

### რატომ ავირჩიეთ XGBoost × LightGBM და არა TimesFM?

TimesFM-მა ცოტა უკეთესი ქულა აჩვენა, მაგრამ საბოლოო მოდელად შევარჩიეთ **XGBoost × LightGBM
მიქსი** შემდეგი მიზეზების გამო:

- ხის მოდელები ჩვენთვის უფრო **ნაცნობი და გასაგებია**: სრულად გვესმის რას აკეთებს და
  როგორ ვმართავთ ჰიპერპარამეტრებს.
- ეს მიდგომა უფრო **პროფესიონალურად** გამოიყურება: ვაშენებთ საკუთარ pipeline-ს, ვატრენინგებთ და ვარეგისტრირებთ.

---

## ექსპერიმენტების თრექინგი (MLflow + DagsHub)

ყველა მოდელს ვაფიქსირებთ **საერთო DagsHub-ის MLflow** სერვერზე:
`slosa23/Walmart-Sales-Forecasting`.

- თითო ჰიპერპარამეტრის კონფიგურაცია = ცალკე **run** (პარამეტრები + WMAE).
- საუკეთესო მოდელები ინახება **Model Registry**-ში, მაგ. `Walmart_Tree_Ensemble`,
  `Walmart_XGBoost_Baseline`, `Walmart_LightGBM_Baseline`, `Walmart_Champion_TimesFM`.

**რეგისტრაცია:** `notebooks/register_tree_ensemble.ipynb`: არჩეულ ანსამბლს ვფუთავთ
MLflow `pyfunc`-ად (შიგნით ორივე მოდელი + შერევის წონა + ფიჩერების აწყობა) და
ვარეგისტრირებთ როგორც `Walmart_Tree_Ensemble`.

**ინფერენსი:** `notebooks/inference_tree_ensemble.ipynb`, მოდელს **პირდაპირ Registry-დან**
ვტვირთავთ და ვუშვებთ **დაუმუშავებელ** `test.csv`-ზე. აქ არავითარი preprocessing არ წერია,
ყველაფერს თვითონ pyfunc აკეთებს. შედეგად იწერება `submission.csv` Kaggle-ისთვის.

---

## პროექტის სტრუქტურა

```
Walmart-Sales-Forecasting/
├── configs/                  # თითო მოდელის ჰიპერპარამეტრები (YAML)
│   ├── xgboost_config.yaml
│   ├── lightgbm_config.yaml
│   ├── nbeats_config.yaml ... (და სხვები)
├── src/
│   ├── features/
│   │   ├── preprocessing.py       # ხის/კლასიკური მოდელების ფიჩერები
│   │   └── nn_preprocessing.py    # ნეირონული ქსელების მონაცემები
│   ├── models/                    # თითო მოდელის pipeline builder
│   │   ├── xgboost_pipeline.py
│   │   ├── lightgbm_pipeline.py
│   │   └── ... (nbeats, patchtst, tft, prophet, sarima)
│   └── evaluation/
│       └── metrics.py             # calculate_wmae()
├── notebooks/                # ექსპერიმენტები, ანსამბლი, რეგისტრაცია, ინფერენსი
│   ├── 01_EDA.ipynb
│   ├── model_experiment_*.ipynb
│   ├── model_ensemble_trees.ipynb
│   ├── register_tree_ensemble.ipynb
│   └── inference_tree_ensemble.ipynb
├── requirements.txt          # ხის/კლასიკური მოდელების ბიბლიოთეკები
├── requirements-nn.txt       # დამატებით ნეირონული ქსელებისთვის
└── README.md
```

---

## როგორ გავუშვათ

**1. ვირტუალური გარემო და ბიბლიოთეკები**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-nn.txt   # მხოლოდ თუ ნეირონულ ქსელებსაც უშვებ
```

**2. მონაცემები**: ჩამოტვირთე Kaggle-იდან და ჩააგდე `data/` ფოლდერში
(`train.csv`, `test.csv`, `stores.csv`, `features.csv`).

**3. DagsHub-ზე ავტორიზაცია**: პირველ გაშვებაზე ბრაუზერში დაგადასტურებინებს.

**4. ნოუთბუქების გაშვება**: გახსენი სასურველი ნოუთბუქი და დააჭირე **Run All**.
საბოლოო პროგნოზისთვის გაუშვი `notebooks/inference_tree_ensemble.ipynb`, ის Registry-დან
ჩამოტვირთავს არჩეულ მოდელს და დააგენერირებს `submission.csv`-ს.

---

## დასკვნა

გავტესტეთ მოდელების ფართო სპექტრი: ხის, კლასიკური, ნეირონული და foundation. საბოლოო
არჩევანი გავაკეთეთ **XGBoost × LightGBM ანსამბლზე** (val WMAE ≈ 1,932), როგორც ყველაზე
გასაგებ და საიმედო მიდგომაზე. TimesFM-ის ბონუს-ექსპერიმენტმა აჩვენა, რომ pretrained
foundation მოდელს ამ ამოცანაზეც კარგი შედეგის მოტანა შეუძლია, საინტერესო მიმართულებაა
სამომავლოდ.
