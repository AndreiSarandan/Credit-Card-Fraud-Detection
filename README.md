**Credit Card Fraud Detection Using SVM algorithm
🔍 Machine Learning Model to Identify Fraudulent Transactions**
This project builds a Credit Card Fraud Detection System using a Support Vector Machine (SVM) classifier.
The dataset contains transaction details, and the goal is to classify transactions as fraudulent (1) or non-fraudulent (0). Since fraud cases are rare, the dataset is highly imbalanced, requiring careful handling through class weighting and feature scaling.

****📊 Dataset**
•	Source: Kaggle – Credit Card Fraud Dataset -> https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud
•	Size: 1,000,000 transactions
•	Features: 
o	distance_from_home
o	distance_from_last_transaction
o	ratio_to_median_purchase_price
o	repeat_retailer
o	used_chip
o	used_pin_number
o	online_order
o	Target Variable: fraud (1 = Fraud, 0 = Non-Fraud)

**⚙️ Machine Learning Pipeline**
1.	Data Preprocessing
o	Feature scaling using StandardScaler.
o	Train-test split (80-20, 50-50, 20-80) with stratified sampling.
2.	Model Training
o	SVM with RBF and Linear Kernels.
o	Class weighting to handle imbalance.
3.	Model Evaluation
o	Confusion Matrix
o	Precision, Recall, and F1 Score
o	Comparison of False Positives & False Negatives

 **📊 Metrics**
Precision (Fraud): Measures how many transactions flagged as fraud are actually fraud. High precision means fewer false alarms (False Positives).
Recall (Fraud): Measures how many actual fraud transactions are successfully detected. High recall means fewer missed fraud cases (False Negatives).
F1 Score (Fraud): The harmonic mean of Precision and Recall, balancing both metrics for fraud detection.
False Positives: The number of non-fraud transactions incorrectly flagged as fraud.

**🚀 Results**
  Split     Kernel	  Precision(Fraud)  	Recall (Fraud)	  F1 Score (Fraud)  	False Positives
80% Train	  RBF	        93%	                    100%	            96%	                1,384
50% Train	  RBF	        92%	                    100%	            96%	                3,780
20% Train	  RBF	        90%	                    100%	            95%	                7,824
50% Train	  Linear	    56%	                    96%	              71%	                32,723
20% Train	  Linear	    56%	                    97%	              71%	                52,579

**📌 Key Findings:**
•	RBF Kernel performs better than Linear Kernel, achieving higher precision and fewer false positives.
•	Higher Recall is critical in fraud detection to minimize missed fraud cases.
•	Class weighting significantly improved fraud detection accuracy in an imbalanced dataset.

