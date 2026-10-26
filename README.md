\# Customer Intelligence \& Anomaly Platform



An experimental financial transaction risk analytics platform combining \*\*Isolation Forest anomaly detection\*\* with a rule-based fraud detector. The project explores temporal evaluation, alert quality, and illustrative business-cost trade-offs.



> \*\*Disclaimer:\*\* This is a portfolio research project using simulated transaction data. It is not a production-ready fraud detection system.



\## Project Highlights



\- Analyzed approximately 6.36 million transaction records.

\- Built transaction features for unsupervised anomaly detection.

\- Used a time-based train/validation/test split.

\- Selected the anomaly threshold using validation data.

\- Compared Isolation Forest, a CASH\_OUT rule, and combined alert strategies.

\- Evaluated results using precision, recall, average precision, alert volume, and illustrative costs.

\- Documented limitations and intended use in a model card.



\## Held-Out Test Results



Test period: steps 632–743.



| Strategy | Alerts | True positives | False positives | Recall | Precision | Estimated cost |

|---|---:|---:|---:|---:|---:|---:|

| Isolation Forest | 2,167 | 318 | 1,849 | 25.40% | 14.67% | $149,345 |

| CASH\_OUT rule | 407 | 407 | 0 | 32.51% | 100.00% | $126,750 |

| Combined OR | 2,574 | 725 | 1,849 | 57.91% | 28.17% | $88,295 |

| Combined AND | 0 | 0 | 0 | 0.00% | 0.00% | $187,800 |



\*\*Test-set summary\*\*

\- Test transactions: 89,466

\- Fraud cases: 1,252

\- Isolation Forest average precision: 0.091017

\- Combined OR recall: 57.91%

\- Combined OR precision: 28.17%



The Combined OR strategy detected 725 of the 1,252 fraud cases in this test period. Its estimated cost was $61,050 lower than the Isolation Forest strategy under the project's illustrative cost assumptions.



The CASH\_OUT rule's observed 100% precision applies only to this test sample; it is not a guarantee of future performance.



\## Business-Cost Assumptions



The experiment assumes:

\- $5 for each false-positive investigation.

\- $150 for each missed fraud case.



These are illustrative assumptions, not verified industry costs or measured financial savings. Actual costs depend on transaction values, investigation workflows, recovery rates, and business policy.



\## Methodology



\### 1. Temporal split



| Dataset | Step range | Purpose |

|---|---|---|

| Training | 1–520 | Fit the anomaly detector |

| Validation | 521–631 | Evaluate strategies and select threshold |

| Test | 632–743 | Final held-out evaluation |



The selected anomaly threshold was `0.1239624561`, chosen using validation data with a target alert budget of approximately 2%.



\### 2. Isolation Forest



The unsupervised model was fitted on a deterministic sample of 200,000 training transactions. It uses engineered transaction and balance features, including log-transformed amounts, balance indicators, an amount-to-origin-balance ratio, and transaction-type indicators.



Higher transformed anomaly scores indicate more unusual transactions.



\### 3. Rule-based detector



The CASH\_OUT rule flags transactions when:

\- Transaction type is `CASH\_OUT`.

\- Amount is greater than `$200,000`.

\- The sender's old balance approximately equals the transaction amount, with an absolute tolerance of `$0.01`.



\### 4. Combined strategy



The Combined OR strategy flags a transaction when either the Isolation Forest or the CASH\_OUT rule flags it. This increased recall in the held-out test period, while also producing more alerts than either component alone.



\## Project Structure



```text

customer-intelligence-anomaly-platform/

├── configs/

├── dashboard/

├── data/

│   ├── raw/

│   ├── processed/

│   └── features/

├── notebooks/

├── reports/

│   ├── model\_card.md

│   ├── threshold\_analysis.json

│   ├── validation\_business\_cost.json

│   └── final\_test\_business\_cost.json

├── src/

│   ├── analytics/

│   ├── data/

│   ├── features/

│   └── risk/

├── tests/

├── .gitignore

├── requirements.txt

└── README.md

```



Generated data, trained model artifacts, and large report files may be excluded from version control. The exact reproducibility requirements should be checked against the files included in the repository.



\## Setup



Requires a compatible Python environment. The commands below use Windows PowerShell.



```powershell

git clone <YOUR-GITHUB-REPOSITORY-URL>

cd customer-intelligence-anomaly-platform



python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1



python -m pip install --upgrade pip

pip install -r requirements.txt

```



The raw dataset is not included in this repository. Obtain the dataset from its authorized source and place it at the expected path:



```text

data/raw/transactions.csv

```



Check individual scripts for their required input files and execution order before rerunning the pipeline.



\## Run Tests



```powershell

pytest -q

```



\## Reproducibility and Reports



Key evaluation scripts:

\- `src/risk/train\_isolation\_forest.py`

\- `src/risk/evaluate\_alert\_thresholds.py`

\- `src/risk/evaluate\_validation\_business\_cost.py`

\- `src/risk/evaluate\_final\_test.py`



Key reports:

\- `reports/model\_card.md`

\- `reports/threshold\_analysis.json`

\- `reports/validation\_business\_cost.json`

\- `reports/final\_test\_business\_cost.json`



The final test threshold is frozen from validation analysis; test labels should not be used to tune it.



\## Limitations



\- The dataset is simulated and may not represent real banking behavior.

\- Anomalies are not necessarily fraudulent transactions.

\- The rule's test-set precision may not generalize to future periods.

\- Costs are based on assumptions and require business validation.

\- Fraud patterns and prevalence may shift over time.

\- Production use would require additional validation, monitoring, security review, and operational testing.



\## Intended Use



This project is intended for learning, portfolio demonstration, and experimental comparison of anomaly detection and rule-based risk strategies. It should not be used to automatically block real customer transactions or make consequential decisions without further validation.



\## Author



Developed as a data science portfolio project focused on transaction risk analytics, anomaly detection, temporal evaluation, and business-oriented model assessment.



