import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the dataset
print("Loading dataset...")
# Load dataset
file_path = "card_transdata.csv"
data = pd.read_csv(file_path)
print("Dataset loaded successfully!")
print(f"Dataset shape: {data.shape}\n")

# Step 2: Split features and target
print("Splitting features and target...")
X = data.drop(columns=['fraud'])  # Features
y = data['fraud']  # Target
print(f"Number of features: {X.shape[1]}")
print(f"Class distribution:\n{y.value_counts()}\n")

# Step 3: Split into train and test sets
print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples\n")

# Step 4: Normalize the features
print("Normalizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Feature normalization complete!\n")

# Step 5: Train the SVM model
print("Training the SVM model (this may take some time)...")
svm = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm.fit(X_train, y_train)
print("SVM training complete!\n")

# Step 6: Evaluate the model
print("Making predictions and evaluating the model...")
y_pred = svm.predict(X_test)

# Print Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nPipeline execution complete!")