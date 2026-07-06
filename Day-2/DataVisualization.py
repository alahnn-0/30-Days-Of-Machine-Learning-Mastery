import numpy as np
import pandas as pd
import matplotlib.pylab as plt

df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())
# print(df.info())
# print(df.shape)
# print(df.describe())
print(df.isnull().sum())


# df['Survived'].value_counts().plot(
#     kind='bar'
# )
# plt.title("Survival Distribution")
# plt.xlabel("Survived")
# plt.ylabel("Passengers")
# plt.show()



df['Sex'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    shadow=True
)

plt.title("Passenger Gender Distribution")
plt.ylabel("")
plt.show()

plt.hist(
    df['Fare'],
    bins=30
)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Passengers")
plt.show()

#Survival rate by Gender

survival_by_gender = pd.crosstab(
    df['Sex'],
    df['Survived']
)

survival_by_gender.plot(
    kind='bar'
)

plt.title("Survival by Gender")
plt.show()