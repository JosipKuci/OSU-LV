import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("titanic.csv")

data.columns = data.columns.str.strip()

#0.0.4
#a)
women=data[data['Sex']=='female']
print(f'Broj žena: {len(women)}')

#b)
dead=data[data['Survived']==0]
alive=data[data['Survived']==1]
dead_percentage=len(dead)/len(data)
print(f'Postotak umrlih:{dead_percentage}')

#c)
dead_men=data[data['Survived']==0]
dead_men=dead_men[dead_men['Sex']=='male']
dead_men_percentage=(len(dead_men)/len(data[data['Sex']=='male']))*100
alive_men_percentage=100-dead_men_percentage

dead_women=data[data['Survived']==0]
dead_women=dead_women[dead_women['Sex']=='female']
dead_women_percentage=(len(dead_women)/len(data[data['Sex']=='male']))*100
alive_women_percentage=100-dead_women_percentage

plt.bar(['Men', 'Women'],[alive_men_percentage,alive_women_percentage], color=['green','yellow'])
plt.xlabel('spol')
plt.ylabel('Postotak preživjelih (%)')
plt.show()

#d)
alive_men=data[data['Sex']=='male']
alive_men=alive_men[alive_men['Survived']==1]
alive_men_age=alive_men['Age'].mean()
print(f'Prosječna dob preživjelog muškarca: {alive_men_age}')

alive_women=data[data['Sex']=='female']
alive_women=alive_women[alive_women['Survived']==1]
alive_women_age=alive_women['Age'].mean()
print(f'Prosječna dob preživjele žene: {alive_women_age}')

#e)
import re
alive_men=alive_men.dropna()
print(alive_men['Cabin'].values)
alive_men['Cabin']=[re.sub(r'\d+', '', x) for x in alive_men['Cabin'].values]
alive_men_grouped=alive_men.groupby('Cabin')
print(alive_men_grouped.max())

