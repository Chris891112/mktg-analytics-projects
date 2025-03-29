#!/usr/bin/env python
# coding: utf-8

# ## [ MNU Driver Impact Analysis ]

# ---

# # Business Request
# - Identify correlation matrix: make sure that correlation is not causation in the business context
# - Basic Multi Regression Model to identify MNU driver
# - Limitation: this doesn't represent entire channels with complicated customer journey (in that case, we will use Markov Chain)

# The brand team wants to have a comprehensive understanding of why the number of new players in Dead by Daylight has increased.

# 1) Check overall attributed impact
# 2) Then consider seasonality
# 3) Compare: Multiple Model with hyper parameter tuned results
# - why? R Square is not the only metrics that explain everything and this could be driven by overfitting
# - Gap between training and test result should be minimized
# - at the end of the day, Regression result will be key message for the stake holders

# In[10]:


# to be updated
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor


import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
# grid search for tuning
from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf


# In[2]:


# to have full visibility on dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)


# ## MMM for Daily Active Users

# In[4]:


# Load the data - baseline data set
df = pd.read_csv(r'sample.csv')


# In[5]:


# date-type transfrom
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


# In[6]:


# 3-year data set (should we consider COVID time should be removed? we can discuss)
df.info()


# In[7]:


dnu = df.copy()


# # 1. EDA - Exploratory Data Analysis

# **Adstock Effect**: Advertising impact doesn’t just occur when an ad is seen or heard—it lingers over time, even after the ad stops running. To capture this, ACF uses all possible date pairs to calculate average autocorrelation by lag, scanning the entire time series to measure how similar it is to, say, 30 days ago.
# 
# 1) Step 1: Check if ad spending has memory—apply adstock processing where memory is detected.
# 2) Step 2: Assess correlation to see if the lingering ad effect influences actual behavior, like new player acquisition.

# 

# In[8]:


ad_cost_cols = ["Paid TT Cost Amount", "Google Ads Cost Amount", "Paid FB Cost"]
ad_click_cols = ["Paid TT Clicks", "Google Ads Clicks", "Paid FB Clicks"]
ad_impression_cols = ["Paid TT Impressions", "Google Ads Impressions", "Paid FB Impressions"]


# In[9]:


organic_views_cols = ["YT Views", "YT Live Stream", 
                      "Web Total", "Web Organic","FB Video Views", "IG Video Views", "TT Video Views", "X Video Views"]


# In[12]:


# cost trend 
plt.figure(figsize=(15, 5))
for col in ad_cost_cols:
    plt.plot(df.index, df[col], label=col)
plt.legend()
plt.title("Daily Ad Cost Trends")
plt.show()


# In[13]:


# ACF - residual impact (to get hints on alpha)
for col in ad_cost_cols:
    print(f"ACF for {col}")
    plot_acf(df[col], lags=30)
    plt.show()

# expense vs click (CTR) distribution
for cost_col, click_col in zip(ad_cost_cols, ad_click_cols):
    ctr = df[click_col] / df[cost_col].replace(0, 1)
    sns.histplot(ctr, bins=50, kde=True)
    plt.title(f"CTR Distribution: {click_col}/{cost_col}")
    plt.show()

# corr matrix
ad_related_cols = ad_cost_cols + ad_click_cols + ad_impression_cols
corr = df[ad_related_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Ad Channels Correlation")
plt.show()


# **Step 1**: Estimate Alpha candidates using ACF (planned range: 0.5–0.8)
# 
# - Y-axis: Autocorrelation (0–1); X-axis: Lag (days), showing similarity between past and current values.
# - High bar? Previous day’s spending affects today = lingering effect.
# - Low bar? Effect fades quickly = short memory.
# - Slow decline? Ad effect lasts longer = extended lingering effect.
# 
# 1) TikTok Ads Cost: Drops to near 0 fast; lingering effect sharply declines within 3–5 days.
# 2) Google Ads Cost: ACF at lag 1 ≥ 0.8, significant; strong autocorrelation, effect accumulates into the next day and beyond.
# 3) Facebook Ads Cost: Significant from lag 0–7; lingering period is about 1 week.

# - If ad spending fluctuates wildly every day, adstock becomes meaningless. However, a high ACF suggests costs accumulate or repeat, which could also mean a cumulative effect on new users. Thus, autocorrelation in spending signals the potential for lingering ad effects.

# In[20]:


data = {
    "Channel": ["Google Ads","Facebook Ads","TikTok Ads"], 
    "Residual Impact": ["Decreased Slowly","Medium Pace","Decreased Quickly"],
    "Suggested Alpha": ["0.6 ~ 0.8","0.5 ~ 0.7","0.3 ~ 0.5"],
    "Rationale": ["Longlasting Ad Effect","Medium Effect","Effect Gone Quickly"]
}


# In[21]:


adstock_table = pd.DataFrame(data)


# In[22]:


# longlasting lag effect, then 0.6-0.8 otherwise 0.3 to 0.5 
adstock_table


# **Step 2:** Correlation by Year

# ### 1) Overall Ad vs. DNU

# In[24]:


# ad_cost_cols = ["Paid TT Cost Amount", "Google Ads Cost Amount", "Paid FB Cost"]
# ad_click_cols = ["Paid TT Clicks", "Google Ads Clicks", "Paid FB Clicks"]
# ad_impression_cols = ["Paid TT Impressions", "Google Ads Impressions", "Paid FB Impressions"]


# In[31]:


# corr
corr = dnu[["Daily New Player"] + ad_click_cols + ad_cost_cols + ad_impression_cols].corr()

corr_with_target = corr["Daily New Player"].drop("Daily New Player").sort_values(ascending=False)

# viz
plt.figure(figsize=(8, 6))
sns.barplot(x=corr_with_target.values, y=corr_with_target.index)
plt.title("corr with target")
plt.xlabel("corr")
plt.ylabel("ad variable")
plt.xlim(-1, 1)
plt.tight_layout()
plt.show()


# ### 2) Year vs. DNU

# In[32]:


# let's add year col
dnu["year"] = dnu.index.year


# In[33]:


# Corr by year

for year in sorted(dnu["year"].unique()):
    print(f"\n==== {year} corr ====")
    yearly_df = dnu[dnu["year"] == year]
    corr = dnu[["Daily New Player"] + ad_click_cols + ad_cost_cols + ad_impression_cols].corr()
    corr_with_target = corr["Daily New Player"].drop("Daily New Player").sort_values(ascending=False)

    print(corr_with_target)

    # viz
    plt.figure(figsize=(8, 6))
    sns.barplot(x=corr_with_target.values, y=corr_with_target.index)
    plt.title(f"{year} dnu and ads variables")
    plt.xlabel("Corr")
    plt.ylabel("Ad Vars")
    plt.xlim(-1, 1)
    plt.tight_layout()
    plt.show()



# ### Summary
# 
# - ACF: Alpha estimated based on ACF decay rate = Geometric Adstock.
# 
# ### Yearly Correlation:
# 1) 2022: Strong FB / TT Cost & Impression performance.
# 2) 2023: Pattern nearly identical to the previous year.
# 3) 2024: Trend holds, while Google remains weak.
# * Over 3 years, Paid FB / TT ad spending consistently shows a strong correlation with new user acquisition.

# Ad effects vary by channel:
# 
# 1) TikTok: Responds to impressions, but clicks have little impact.
# 2) Google: Relatively effective with clicks.
# 3) Facebook: Ad spend itself drives strong responses.

# # 2. Adstock Feature Engineering

# In[35]:


def geometric_adstock(x, alpha):
    """Geometric adstock sample"""
    result = []
    for i in range(len(x)):
        if i == 0:
            result.append(x[i])
        else:
            result.append(x[i] + alpha * result[i - 1])
    return result


# In[36]:


dnu["FB_adstock_0.6"] = geometric_adstock(dnu["Paid FB Cost"], alpha=0.6)
dnu["TT_adstock_0.4"] = geometric_adstock(dnu["Paid TT Cost Amount"], alpha=0.4)
dnu["Google_adstock_0.7"] = geometric_adstock(dnu["Google Ads Cost Amount"], alpha=0.7)


# In[ ]:




