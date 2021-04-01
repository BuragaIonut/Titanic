import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.drop(['Survived'], axis=1)
y_train = train['Survived']
X_test = test

X_train['Age'].fillna(int(X_train['Age'].mean()), inplace=True)
X_test['Age'].fillna(int(X_test['Age'].mean()), inplace=True)

X_train.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
X_test.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

label_encoder = LabelEncoder()

for col in ['Sex', 'Embarked']:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])

X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)
clf = RandomForestClassifier(n_estimators=100, max_depth=3)

clf.fit(X_train, y_train)

y_predicted = pd.DataFrame(X_test['PassengerId'])
y_predicted['Survived'] = pd.DataFrame(clf.predict(X_test))
y_predicted.set_index('PassengerId', inplace=True)

y_predicted.to_csv(r'C:\Users\burag\PycharmProjects\Kaggle\Titanic\predicted.csv')
