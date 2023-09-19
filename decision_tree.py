#-------------------------------------------------------------------------
# AUTHOR: Tim Hsieh
# FILENAME: decision_tree.py
# SPECIFICATION: draws a decision tree with data from contact_Lens.csv.
# FOR: CS 4210- Assignment #1
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv

# initialize empty lists for data storage
db = []
X = []
Y = []

# Mapping for categorical to numerical values
age_mapping = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
spectacle_mapping = {'Myope': 1, 'Hypermetrope': 2}
astigmatism_mapping = {'No': 1, 'Yes': 2}
tear_mapping = {'Reduced': 1, 'Normal': 2}
class_mapping = {'Yes': 1, 'No': 2}

# reading the data from a CSV file and transforming it
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # Skipping the header
            db.append(row)
            age = age_mapping[row[0]]
            spectacle = spectacle_mapping[row[1]]
            astigmatism = astigmatism_mapping[row[2]]
            tear = tear_mapping[row[3]]
            X.append([age, spectacle, astigmatism, tear])
            Y.append(class_mapping[row[4]])

# fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)

# plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes', 'No'], filled=True, rounded=True)
plt.show()
