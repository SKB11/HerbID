#code 1
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

# # Load the dataset
# df = pd.read_csv("pn1.csv")
df = pd.read_csv("pn.csv", encoding='latin-1')

# Create a bag of words representation of the plant names
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Name"])


# Convert the target variable to binary
df["Medicinal"] = (df["Medicinal"] == 1)

# Split the data in a stratified way
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, df["Medicinal"]))

# Split the data into training and testing sets
X_train = X[train_idx]
X_test = X[test_idx]
y_train = df["Medicinal"][train_idx]
y_test = df["Medicinal"][test_idx]

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Use the model to make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = (y_pred == y_test).mean()
print("Accuracy:", accuracy)

from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

from sklearn.metrics import confusion_matrix, f1_score

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)
print("F1 score:", f1)


import pandas as pd
df_plants= pd.read_csv("pn.csv", encoding='latin-1')
plant_name = input("Enter plant name: ")
df_filtered = df_plants.loc[(df_plants['Name'].str.lower() == plant_name.lower()) | (df_plants['Hindi Name'].str.lower() == plant_name.lower())]
if not df_filtered.empty:
  plant_info = df_filtered[['Name', 'Hindi Name', 'Botanical Name', 'Uses']].values[0]
  print(f"\n\033[1m> {plant_name} is medicinal.\033[0m\n")
  print(f" * English name of {plant_name} is: {plant_info[0]}")
  print(f" * Hindi name of {plant_name} is: {plant_info[1]}")
  print(f" * The Botanical name of {plant_name} is: {plant_info[2]}")
  print(f"\n >> The Uses of {plant_name} are: {plant_info[3]}")
else:
  print(f"{plant_name} not found in the dataset.")
#code 2
# Get information on a plant's description and medicinal properties
plant_name = input("Enter plant name or keyword: ")

# Check if plant is present in either the Name or Hindi Name column, or if the keyword is present in the Uses column
df_filtered = df_plants.loc[(df_plants['Name'].str.lower() == plant_name.lower()) | (df_plants['Hindi Name'].str.lower() == plant_name.lower()) | (df_plants['Uses'].str.lower().str.contains(plant_name.lower()))]

if not df_filtered.empty:
    for i, row in df_filtered.iterrows():
        plant_info = row[['Botanical Name', 'Uses', 'Medicinal']].values
        plant_name = row['Name']
        if row['Medicinal'] == 1:
            print(f"\n\033[1m{plant_name} is medicinal.\033[0m \n")
        else:
            print(f"\n{plant_name} is not medicinal.\n")
        print(f" * The Botanical name of {plant_name} is: {plant_info[0]}")
        print(f" * The Uses of {plant_name} is: {plant_info[1]}")
        print()
else:
    print(f"No plants or keywords found in the dataset.")

#code 3
# Load the dataset
# df_plants = pd.read_csv('pn2.csv')
df_plants = pd.read_csv("pn.csv", encoding='latin-1')

# Get information on a plant's description and medicinal properties
plant_name = input("Enter plant name or keyword: ")

# Check if plant is present in either the Name or Hindi Name column, or if the keyword is present in the Uses column
df_filtered = df_plants.loc[(df_plants['Name'].str.lower() == plant_name.lower()) | (df_plants['Hindi Name'].str.lower() == plant_name.lower()) | (df_plants['Uses'].str.lower().str.contains(plant_name.lower()))]

if not df_filtered.empty:
    for i, row in df_filtered.iterrows():
        plant_info = row[['Botanical Name', 'Uses', 'Medicinal']].values
        plant_name = row['Name']
        if row['Medicinal'] == 1:
            print(f"\n\033[1m{plant_name} is medicinal.\033[0m \n")
        else:
            print(f"\n{plant_name} is not medicinal.\n")
        print(f" * The Botanical name of {plant_name} is: {plant_info[0]}")
        print(f" * The Uses of {plant_name} is: {plant_info[1]}")
        print()
else:
    print(f"No plants or keywords found in the dataset.")
