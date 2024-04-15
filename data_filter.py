import pandas as pd
from IPython.core import getipython  as ph
import matplotlib.pyplot as plt
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer


#limpia consola
ph.get_ipython().run_line_magic('clear',"")

# %% [1] //bloque 1
#leemos el archivo de datos
Train=pd.read_csv('train.csv')

#eliminamos atributos inecesarios
Df_train_full = Train

#obtenemos el porcentaje de nan en cada atributo
Train_info=Df_train_full.describe()
Train_notnan = Df_train_full.count()
Train_nan = Df_train_full.isna().sum()
Droped= Df_train_full.dropna().shape
Nan_percentage = (Df_train_full.shape[0]-Droped[0])/Df_train_full.shape[0]*100
print('Porcentaje de registros con nan: ',Nan_percentage,'%\n\n/-------------------/ 0 /-------------------/\n')

df_train_v2=Df_train_full.drop(columns=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','PassengerId'])
Columns= df_train_v2.columns.drop(['Cabin','Name'])



#%%% [1]
#vemos la distribucion de Age

Age=df_train_v2["Age"].value_counts(ascending=True)
Age_sort=np.argsort(Age.index)
Age_sorted=Age.values[Age_sort]

plt.plot(Age_sorted,linestyle='solid', marker='o',color='orange')
plt.bar(Age.index,Age.values)
plt.title('Age')
plt.show()

#vemos la distribucion de HomePlanet

HomePlanet=df_train_v2['HomePlanet'].value_counts()
plt.bar(HomePlanet.index,HomePlanet.values)
plt.title('HomePlanet')
plt.show()

#vemos la distribucion de Destination

Destination=df_train_v2['Destination'].value_counts()
plt.bar(Destination.index,Destination.values)
plt.title('Destination')
plt.show()

#vemos la distribucion de CryoSleep

CryoSleep=df_train_v2['CryoSleep'].value_counts()
plt.bar(['Si','No'],CryoSleep.values)
plt.title('CryoSleep')
plt.show()

#vemos la distribucion de VIP

VIP=df_train_v2['VIP'].value_counts()
plt.bar(['No','Si'],VIP.values)
plt.title('VIP')
plt.show()



#%%% [2]
# Imputing
# encoding categorical
print(df_train_v2['Age','Index'])
imputador_1= SimpleImputer(missing_values=np.nan, strategy='mean')
imputando_iterador= imputador_1.fit_transform(df_train_v2['Age','Index'])
