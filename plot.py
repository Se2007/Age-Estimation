import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

os.chdir('./IMDB-clean dataset')

df = pd.read_csv('12.csv')

df_train, temp = train_test_split(df, test_size=0.3, random_state=42)#
df_test, df_valid = train_test_split(temp, test_size=0.5 ,random_state=42)#, stratify=temp.age

# df2 = pd.read_csv('test.csv')
# df3 = pd.read_csv('train.csv')



plt.hist(df_train.age, len(df_train.age.unique()),density = 1,color ='green',alpha = 0.7)
plt.hist(df_test.age, len(df_test.age.unique()),density = 1,color ='red',alpha = 0.7)
plt.hist(df_valid.age, len(df_valid.age.unique()),density = 1,color ='blue',alpha = 0.7)

plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
 
plt.title('Age analyze\n\n',
          fontweight = "bold")
 
plt.show()



