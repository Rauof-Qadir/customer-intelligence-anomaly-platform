\# Model Card: Customer Intelligence \& Anomaly Platform



\## 1. Model Overview



\*\*Model:\*\* Isolation Forest  

\*\*Task:\*\* Transaction anomaly detection  

\*\*Project type:\*\* Financial transaction risk analytics  

\*\*Approach:\*\* Unsupervised anomaly detection with a complementary rule-based detector



The platform flags unusual transactions for investigation. An anomaly is not automatically proof of fraud.



\## 2. Business Problem



Financial institutions may process large volumes of transactions, making manual review difficult. This project explores how anomaly detection and simple business rules can help prioritize transactions for review.



\## 3. Dataset



\- Source file: `data/raw/transactions.csv`

\- Dataset size: approximately 6.36 million transactions

\- Target label: `isFraud`

\- Evaluation periods are separated by the `step` column.



The dataset is PaySim-style and should be treated as a simulated transaction dataset, not evidence of production banking performance.



\## 4. Temporal Data Split



| Split | Step range | Purpose |

|---|---|---|

| Training | 1–520 | Fit the Isolation Forest |

| Validation | 521–631 | Analyze thresholds and strategies |

| Test | 632–743 | Final held-out evaluation |



The model was fitted on a deterministic sample of 200,000 training rows. Imputation medians were calculated from training data only.



\## 5. Model Features



The model uses 11 features:



\- Log transaction amount

\- Log sender balance

\- Log recipient balance

\- Sender balance is zero

\- Recipient balance is zero

\- Transaction amount divided by sender balance

\- Transaction-type indicators for CASH\_IN, CASH\_OUT, DEBIT, PAYMENT, and TRANSFER



Post-transaction balance fields and entity identifiers are not used as model features.



\## 6. Model Configuration



\- Algorithm: `sklearn.ensemble.IsolationForest`

\- Trees: 150

\- `max\_samples`: 256

\- `contamination`: `auto`

\- Random seed: 42

\- Selected threshold: `0.1239624561`

\- Threshold selection: validation-based, targeting approximately a 2% alert budget



Higher transformed anomaly scores indicate more unusual transactions. A transaction is flagged when its score is greater than or equal to the selected threshold.



\## 7. Final Test Results



Test period: steps 632–743



\- Transactions evaluated: 89,466

\- Fraud cases: 1,252

\- Average precision: 0.091017



| Strategy | Alerts | True positives | False positives | Recall | Precision |

|---|---:|---:|---:|---:|---:|

| Isolation Forest | 2,167 | 318 | 1,849 | 25.40% | 14.67% |

| CASH\_OUT rule | 407 | 407 | 0 | 32.51% | 100.00% |

| Combined OR | 2,574 | 725 | 1,849 | 57.91% | 28.17% |

| Combined AND | 0 | 0 | 0 | 0.00% | 0.00% |



Metrics reflect this held-out test period only. The rule's observed 100% precision must not be interpreted as a guarantee of future performance.



\## 8. Illustrative Business-Cost Analysis



Assumptions used:



\- False-positive investigation cost: $5 per transaction

\- Missed-fraud cost: $150 per transaction



| Strategy | Estimated total cost |

|---|---:|

| Isolation Forest | $149,345 |

| CASH\_OUT rule | $126,750 |

| Combined OR | $88,295 |

| Combined AND | $187,800 |



Under these assumptions, Combined OR has the lowest estimated cost among the evaluated strategies.



These values are illustrative scenario estimates, not verified real-world costs or realized financial savings. The missed-fraud cost is modeled as a fixed amount per missed case, not the actual value of each transaction.



\## 9. Limitations and Risks



1\. Anomaly detection identifies unusual behavior, not confirmed fraud.

2\. The dataset is simulated; results may not transfer to live banking systems.

3\. The CASH\_OUT rule's test precision may change on future data.

4\. The cost assumptions require validation with real business stakeholders.

5\. Transaction patterns and fraud prevalence can change over time.

6\. False positives can create investigation workload, while false negatives can allow fraud to go undetected.

7\. This model has not been demonstrated to meet production latency, monitoring, fairness, security, or regulatory requirements.



\## 10. Intended Use



Suitable for:

\- Portfolio demonstrations

\- Model evaluation experiments

\- Comparing rule-based and unsupervised approaches

\- Demonstrating temporal validation and test methodology



Not suitable for:

\- Automatically blocking real customer transactions

\- Making decisions about real customers without further validation

\- Claiming production-ready fraud detection performance



\## 11. Reproducibility and Artifacts



Relevant project artifacts:



\- `data/features/isolation\_forest.joblib`

\- `reports/threshold\_analysis.json`

\- `reports/validation\_business\_cost.csv`

\- `reports/validation\_business\_cost.json`

\- `reports/final\_test\_business\_cost.csv`

\- `reports/final\_test\_business\_cost.json`



\## 12. Conclusion



The Combined OR strategy detected 725 of 1,252 fraud cases in the held-out test period, achieving 57.91% recall and 28.17% precision across 2,574 alerts.



This is a useful experimental result for a portfolio project, not evidence of production readiness. Further work should focus on alert quality, cost assumptions, monitoring, and evaluation on additional time periods.



