#!/usr/bin/env python
# coding: utf-8

# ## [ MNU Driver Impact Analysis ]

# ---

# ## Business Request
# - Identify correlation matrix: make sure that correlation is not causation in the business context
# - Basic Multi Regression Model to identify MNU driver
# - Limitation: this doesn't represent entire channels with complicated customer journey

# The brand team wants to have a comprehensive understanding of why the number of new players in Dead by Daylight has increased.

# 1) Check overall attributed impact
# 2) Then consider seasonality

# In[28]:


# to be updated
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


# In[2]:


# to have full visibility on dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)


# ## MMM for Monthly New Player

# In[27]:


# Load the data
df = pd.read_csv(r'C:DNU_Daily_Channel.csv')


# In[40]:


# date-type transfrom
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


# In[14]:


df.info()


# In[15]:


df.head()


# ### 01. Correlation
# - Overall vs. Year
# - Correlation analysis only captures linear relationships between variables, so non-linear relationships (e.g., complex interactions between events and player growth) may not be reflected.
# - A high correlation coefficient does not necessarily imply causation.
# - Without cost data, ROI (return on investment) cannot be calculated.
# - Comparing paid ads and organic channels may be challenging.

# In[16]:


# 1. overall corr
overall_corr = df.corr()['Daily New Player'].sort_values(ascending=False)
print("Overall Correlation with Daily New Player:\n", overall_corr)

# 2. Extract Year cols from Date Index
df['Year'] = df.index.year


# In[17]:


# yearly corr
years = df['Year'].unique()
yearly_corr = {}

for year in years:
    yearly_data = df[df['Year'] == year].drop(columns=['Year'])
    yearly_corr[year] = yearly_data.corr()['Daily New Player'].sort_values(ascending=False)
    print(f"\nCorrelation with Daily New Player for {year}:\n", yearly_corr[year])


# ### 02. Modeling
# - Ridge Regression: Uses L2 regularization to mitigate multicollinearity.
# - Adstock: Reflects the lingering effect of advertising, as its impact does not disappear immediately. For example, the effect of Paid FB Impressions may persist for several days. (Adstock t = Value t + decay × Adstock t−1)
# - Saturation: Simplifies the saturation of marketing effects (e.g., diminishing returns from increased ad spend) using logarithmic transformation.

# ### A. Ridge Regression with Adstock

# In[41]:


dnu = df.copy()


# In[42]:


dnu.columns


# #### Reflecting Adstock Effect (Time Decay)

# In[26]:


# Independent and Dependent variable
X = dnu.drop(columns=['Daily New Player', 'Daily Active Players'] + marketing_vars)  # 불필요한 컬럼 제외
y = dnu['Daily New Player']

# Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Ridge Regression training
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Eval
train_score = ridge.score(X_train, y_train)
test_score = ridge.score(X_test, y_test)
print(f"Ridge Regression - Train R^2: {train_score:.4f}, Test R^2: {test_score:.4f}")

# Coefficient = contribution
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': ridge.coef_})
print("\nFeature Contributions:\n", coef_df.sort_values(by='Coefficient', ascending=False))


# ### B. Elastic Net (Mix)

# 1. Add Organic Channels:
# In addition to paid channels (e.g., Paid TT Impressions), apply Adstock transformation to organic channels (e.g., YT Views, FB Impressions, IG Impressions, TT Impressions, X Impressions).
# Select one representative column per organic channel (e.g., YT Views, FB Impressions).
# 2. Refine Variables:
# Exclude variables that showed negative coefficients in previous results (e.g., X Impressions, IG Impressions).
# Remove ad spend-related columns (e.g., Paid TT Cost Amount, Google Ads Cost Amount, Paid FB Cost) and Daily Active Players.
# 3. Simplify Model:
# Skip yearly analysis and focus on the entire dataset.
# Use Elastic Net instead of Ridge Regression to enhance variable selection and model stability.

# - Stability Improvements: Ridge + Lasso
# - PCA: Dimension Reduction (Minimize Correlation)

# In[30]:


# available feature checks 
dnu.columns


# In[33]:


# Adstock Effect Func
def adstock_transform(series, decay=0.5):
    adstock = np.zeros_like(series, dtype=float)
    adstock[0] = series[0]
    for t in range(1, len(series)):
        adstock[t] = series[t] + decay * adstock[t-1]
    return adstock

# Application: Organic and Paid
organic_vars = ['YT Net Audience Growth', 'FB Impressions', 'IG Engagements', 
                'TT Engagements', 'X Net Audience Growth', 'Web Total']
paid_vars = ['Paid TT Impressions', 'Paid FB Impressions']  # 광고비 제외, 노출만 포함
all_vars = organic_vars + paid_vars

for var in all_vars:
    dnu[f'{var}_adstock'] = adstock_transform(dnu[var].values, decay=0.5)

# Saturation (Log)
for var in all_vars:
    dnu[f'{var}_adstock_log'] = np.log1p(dnu[f'{var}_adstock'])

# Feature selection
adstock_log_vars = [f'{var}_adstock_log' for var in all_vars]
event_vars = ['LTE', 'Rift', 'Global Event', 'New Mode', 'Franchise Collection', 
              'Original Collection', 'Franchise Chapter', 'Original Chapter', 
              '1st Party Sales', 'Free Trial']


X = dnu[event_vars + adstock_log_vars]
y = dnu['Daily New Player']

 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Elastic Net + parameter tuning
elastic_net = ElasticNet(alpha=0.5, l1_ratio=0.7)  # alpha 낮추고 l1_ratio 조정
elastic_net.fit(X_train, y_train)

# Eval
train_score = elastic_net.score(X_train, y_train)
test_score = elastic_net.score(X_test, y_test)
print(f"Elastic Net - Train R^2: {train_score:.4f}, Test R^2: {test_score:.4f}")

# same as before
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': elastic_net.coef_})
print("\nFeature Contributions:\n", coef_df.sort_values(by='Coefficient', ascending=False))


# ### - Exploration for Parameters (Elastic Net still works)

# In[43]:


# 2. Corr with Daily New Player
correlation = dnu.corr()['Daily New Player'].sort_values(ascending=False)
print("\nCORR with Daily New Player:\n", correlation)


# In[44]:


# Seasonality
dnu['Month'] = dnu.index.month
month_dummies = pd.get_dummies(dnu['Month'], prefix='Month', drop_first=True)
dnu = pd.concat([dnu, month_dummies], axis=1)

# Year
dnu['Year'] = dnu.index.year
year_dummies = pd.get_dummies(dnu['Year'], prefix='Year', drop_first=True)
dnu = pd.concat([dnu, year_dummies], axis=1)


# In[46]:


dnu.info()


# In[47]:


# Adstock  
def adstock_transform(series, decay):
    adstock = np.zeros_like(series, dtype=float)
    adstock[0] = series[0]
    for t in range(1, len(series)):
        adstock[t] = series[t] + decay * adstock[t-1]
    return adstock

#  
organic_vars = ['FB Impressions', 'X Net Audience Growth', 'Web Organic']
paid_vars = ['Paid TT Impressions', 'Paid FB Impressions', 'Google Ads Impressions', 
             'Google Ads Clicks', 'Paid FB Clicks', 'Paid TT Clicks']
all_vars = organic_vars + paid_vars

#  
decay_values = {
    'FB Impressions': 0.3,
    'X Net Audience Growth': 0.5,
    'Web Organic': 0.2,
    'Paid TT Impressions': 0.5,
    'Paid FB Impressions': 0.5,
    'Google Ads Impressions': 0.5,
    'Google Ads Clicks': 0.5,
    'Paid FB Clicks': 0.5,
    'Paid TT Clicks': 0.5
}

for var in all_vars:
    dnu[f'{var}_adstock'] = adstock_transform(dnu[var].values, decay=decay_values[var])
    dnu[f'{var}_adstock_log'] = np.log1p(dnu[f'{var}_adstock'])


# In[48]:


# Chapter + Year (as the impact from channel by year was different)
chapter_vars = ['Franchise Chapter', 'Original Chapter']
for chapter in chapter_vars:
    for year_col in year_dummies.columns:
        dnu[f'{chapter}_{year_col}'] = dnu[chapter] * dnu[year_col]


# In[49]:


# select
adstock_log_vars = [f'{var}_adstock_log' for var in all_vars]
event_vars = ['LTE', 'Rift', 'Global Event', 'New Mode', '1st Party Sales', 'Free Trial']
month_vars = [col for col in dnu.columns if col.startswith('Month_')]
year_vars = [col for col in dnu.columns if col.startswith('Year_')]
interaction_vars = [col for col in dnu.columns if 'Franchise Chapter_Year' in col or 'Original Chapter_Year' in col]

#  
X = dnu[event_vars + adstock_log_vars + month_vars + year_vars + interaction_vars + chapter_vars]
y = dnu['Daily New Player']

#  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Elastic Net training
elastic_net = ElasticNet(alpha=0.5, l1_ratio=0.7)
elastic_net.fit(X_train, y_train)

#  
train_score = elastic_net.score(X_train, y_train)
test_score = elastic_net.score(X_test, y_test)
print(f"Elastic Net (interaction feature added) - Train R^2: {train_score:.4f}, Test R^2: {test_score:.4f}")

#  
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': elastic_net.coef_})
print("\nFeature Contributions:\n", coef_df.sort_values(by='Coefficient', ascending=False))


# 3rd Round

# In[51]:


import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Adstock   
def adstock_transform(series, decay):
    adstock = np.zeros_like(series, dtype=float)
    adstock[0] = series[0]
    for t in range(1, len(series)):
        adstock[t] = series[t] + decay * adstock[t-1]
    return adstock

#  
organic_vars = ['FB Impressions', 'X Net Audience Growth', 'Web Organic', 
                'IG Net Audience Growth', 'YT Net Audience Growth']  # 유튜브 추가
paid_vars = ['Paid TT Impressions', 'Paid FB Impressions', 'Google Ads Impressions', 
             'Google Ads Clicks', 'Paid FB Clicks', 'Paid TT Clicks']
all_vars = organic_vars + paid_vars

#  
decay_values = {
    'FB Impressions': 0.5,
    'X Net Audience Growth': 0.7,
    'Web Organic': 0.2,
    'IG Net Audience Growth': 0.4,
    'YT Net Audience Growth': 0.7,  # 유튜브는 효과 오래 지속
    'Paid TT Impressions': 0.5,
    'Paid FB Impressions': 0.5,
    'Google Ads Impressions': 0.5,
    'Google Ads Clicks': 0.5,
    'Paid FB Clicks': 0.5,
    'Paid TT Clicks': 0.5
}

for var in all_vars:
    dnu[f'{var}_adstock'] = adstock_transform(dnu[var].values, decay=decay_values[var])
    dnu[f'{var}_adstock_log'] = np.log1p(dnu[f'{var}_adstock'])


#  
chapter_vars = ['Franchise Chapter', 'Original Chapter']
for chapter in chapter_vars:
    for year_col in year_dummies.columns:
        dnu[f'{chapter}_{year_col}'] = dnu[chapter] * dnu[year_col]

#  
adstock_log_vars = [f'{var}_adstock_log' for var in all_vars]
event_vars = ['LTE', 'Rift', 'Global Event', 'New Mode', '1st Party Sales', 'Free Trial']
month_vars = [col for col in dnu.columns if col.startswith('Month_')]
year_vars = [col for col in dnu.columns if col.startswith('Year_')]
interaction_vars = [col for col in dnu.columns if 'Franchise Chapter_Year' in col or 'Original Chapter_Year' in col]

#  
X = dnu[event_vars + adstock_log_vars + month_vars + year_vars + interaction_vars + chapter_vars]
y = dnu['Daily New Player']

#  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Elastic Net  
elastic_net = ElasticNet(alpha=0.5, l1_ratio=0.7)
elastic_net.fit(X_train, y_train)

#  
train_score = elastic_net.score(X_train, y_train)
test_score = elastic_net.score(X_test, y_test)
print(f"Elastic Net (유튜브 추가) - Train R^2: {train_score:.4f}, Test R^2: {test_score:.4f}")

#  
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': elastic_net.coef_})
print("\nFeature Contributions:\n", coef_df.sort_values(by='Coefficient', ascending=False))


# In[ ]:





# In[ ]:




