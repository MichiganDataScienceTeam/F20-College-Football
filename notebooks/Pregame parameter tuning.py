#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
df2 = pd.read_csv("../data/game_stats.csv")
def process_third(x):
    if(str(x).split('-')[-1] == '0'): return 1
    else: return float(str(x).split('-')[0])/float(str(x).split('-')[1])

stats = df2[['id', 'home.school','home.points', 'home.thirdDownEff','home.completionAttempts', 'home.yardsPerPass','home.yardsPerRushAttempt','home.totalPenaltiesYards','home.turnovers','home.possessionTime','away.school','away.points','away.thirdDownEff','away.completionAttempts', 'away.yardsPerPass','away.yardsPerRushAttempt','away.totalPenaltiesYards','away.turnovers','away.possessionTime','home.rushingAttempts','away.rushingAttempts','Year','Week']]
stats.dropna(inplace = True)
stats.loc[:,'home.pointDiff'] = df2['home.points'] - df2['away.points']
stats.loc[:,'away.pointDiff'] = -stats['home.pointDiff']
stats.loc[:,'home.completionAttempts'] = df2['home.completionAttempts'].apply(lambda x: float(str(x).split('-')[-1]))
stats.loc[:,'away.completionAttempts'] = df2['away.completionAttempts'].apply(lambda x: float(str(x).split('-')[-1]))
stats.loc[:,'home.yardsAllowedPerPlay'] = df2['away.totalYards']/(stats['away.completionAttempts'] + df2['away.rushingAttempts'])
stats.loc[:,'away.yardsAllowedPerPlay'] = df2['home.totalYards']/(stats['home.completionAttempts'] + df2['home.rushingAttempts'])
stats.loc[:,'home.yards_per_play'] = stats['away.yardsAllowedPerPlay']
stats.loc[:,'away.yards_per_play'] = stats['home.yardsAllowedPerPlay']
stats.loc[:,'home.forcedTO'] = df2['away.turnovers']
stats.loc[:,'away.forcedTO'] = df2['home.turnovers']
stats.loc[:,'home.passAttemptsAllowed'] = stats['away.completionAttempts']
stats.loc[:,'away.passAttemptsAllowed'] = stats['home.completionAttempts']
stats.loc[:,'home.yardsAllowedPerRush'] = df2['away.yardsPerRushAttempt']
stats.loc[:,'away.yardsAllowedPerRush'] = df2['home.yardsPerRushAttempt']
stats.loc[:,'home.thirdDownEff'] = stats.loc[:,'home.thirdDownEff'].apply(process_third)
stats.loc[:,'away.thirdDownEff'] = stats.loc[:,'away.thirdDownEff'].apply(process_third)
stats.loc[:,'home.totalPenaltiesYards'] = stats.loc[:,'home.totalPenaltiesYards'].apply(lambda x: float(x.split('-')[-1]))
stats.loc[:,'away.totalPenaltiesYards'] = stats.loc[:,'away.totalPenaltiesYards'].apply(lambda x: float(x.split('-')[-1]))
stats.loc[:,'home.possessionTime'] = stats['home.possessionTime'].apply(lambda x: float(x.split(':')[0])*60 + float(x.split(':')[1]))
stats.loc[:,'away.possessionTime'] = stats['away.possessionTime'].apply(lambda x: float(x.split(':')[0])*60 + float(x.split(':')[1]))
stats['home.win'] = (stats['home.points'] > stats['away.points']).astype(int)
stats['away.win'] = (stats['away.points'] > stats['home.points']).astype(int)
stats.iloc[1239,22] = 10
stats = stats.drop(9284, axis = 0)
away_index = [x for x in list(stats.columns) if 'away.' in x] + ['Week', 'Year', 'id']
away = stats[away_index].drop('away.points', axis = 1)
away.columns = [x.split('.')[-1] for x in list(away.columns)]
away['isHome'] = 0
away = away.groupby(['Year', 'school','Week']).mean()
home_index = [x for x in list(stats.columns) if 'home.' in x] + ['Week', 'Year', 'id']
home = stats[home_index].drop('home.points',axis = 1)
home.columns = [x.split('.')[-1] for x in list(home.columns)]
home['isHome'] = 1
home = home.groupby(['Year', 'school','Week']).mean()
total = pd.concat([home,away]).groupby(['Year', 'school','Week']).mean()
idx = list(total.index)
idx.reverse()
n = 1
ids = total['id']
field_status = total['isHome']
outcomes = total['win']
total.drop(['id','isHome'], axis = 1, inplace = True) 
todrop = []
for i,x in enumerate(idx[:-1]):
    if(x[1] != idx[i + 1][1]):
        todrop.append(x)
        continue
    j = 2
    agg = total.loc[idx[i+1],:]
    while((i + j < len(idx)) and x[1] == idx[i + j][1]):
        agg = agg + total.loc[idx[i+j],:]
        j += 1
        n += 1
    total.loc[x,:] = agg/n
    n = 1;
total['id'] = ids
total['isHome'] = field_status
total['won_game'] = outcomes
total.drop(todrop, axis = 0, inplace = True)
rowlist = []
for gameID in stats['id']:
    if len(total[total['id'] == gameID]) != 2: continue
    hometeam = list((total[(total['id'] == gameID) & (total['isHome'] == 1)].values)[0])
    awayteam = list((total[(total['id'] == gameID) & (total['isHome'] == 0)].values)[0])
    rowlist.append(hometeam + awayteam)
    rowlist.append(awayteam + hometeam)
matchups = pd.DataFrame(rowlist, columns = (list(total.columns) + ['other.' + x for x in list(total.columns)])).drop(['other.id','other.isHome','other.won_game'], axis = 1)
matchups = matchups.set_index('id')
matchups['win_pct_diff'] = matchups['win'] - matchups['other.win']
matchups.drop(['win', 'other.win'], axis = 1, inplace = True)
outcome = matchups['won_game']
matchups.drop('won_game', axis = 1, inplace = True)
matchups['outcome'] = outcome
matchups = matchups.astype(float).dropna()
X = matchups.iloc[:,:-1]
y = matchups.iloc[:,-1]
scaler = preprocessing.StandardScaler().fit(X)
X_t = scaler.transform(X)
grid = {"C":np.logspace(-3,3,7), 'kernel':['linear', 'poly', 'rbf', 'sigmoid']}
clf = SVC()
cv = GridSearchCV(clf, grid, cv=10)
cv.fit(X_t,y)
print(cv.best_params_)


# In[ ]:




