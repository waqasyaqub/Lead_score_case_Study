#!/usr/bin/env python
# coding: utf-8

# # Lead Scoring Case Study

# ## Following are the steps

# #### 1.  Loading libraries and Dataframe
# #### 2.  Checking Dataframe values
# #### 3.  Cleaning the Data
# #### 4.  Univariate Analysis (Categorical Variables)
# #### 5.  Univariate Analysis (Numeric Variables)
# #### 6.  Multivariate Analysis
# #### 7.  Dummy Variables
# #### 8.  Train-Test Split
# #### 9. Feature Selection & Model building
# #### 10. Evaluating Model
# #### 11. Predict on Test
# 
# 

# ## 1. Loading Libraries and Dataframe

# In[1]:


# importing the respective libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# loading the data in Dataframe
inp_data = pd.read_csv(r'E:\Projects\Degree\5. Machine learning\11. lead scoring case study\Lead Scoring Assignment\Leads.csv')


# ## 2. Checking Dataframe Values

# In[2]:


# checking the null values from the dataframe
inp_data.info()


# In[3]:


# checking number of rows and columns
inp_data.shape


# In[4]:


# checking the numeric values 
inp_data.describe()


# In[5]:


# displaying the data
inp_data.head()


# ## 3. Cleaning The Data

# In[6]:


# checking for the pecentage of missing values in dataframe

missing_values_percentage = round(100*(inp_data.isnull().sum()/len(inp_data.index)), 2)
print(missing_values_percentage)


# ### Insight
# 
#  * Columns such as  'Last Activity', 'Last notable Activity' and 'lead quality' no part in analysis.
#  * Prospect ID and Lead Number both are both unique identifiers and can be dropped as no role in analysis.
#  * The Column 'Tags' is used for reference, as per business it has no relevance to the outcome, hence droping this as well.
# 

# In[7]:


# dropping the respective columns
inp_data.drop(['Prospect ID', 'Lead Number', 'Tags', 'Last Activity', 'Lead Quality','Last Notable Activity'], axis = 1, inplace = True)


# In[8]:


inp_data.head()


# Checking the values of non cloumns containing more than 45% null values
# 
# * Asymmetrique Activity Index
# * Asymmetrique Profile Index
# * Asymmetrique Activity Score
# * Asymmetrique Profile Score

# In[9]:


inp_data['Asymmetrique Activity Index'].value_counts(normalize = True, dropna = False)*100


# In[10]:


inp_data['Asymmetrique Profile Index'].value_counts(normalize = True, dropna = False)*100


# In[11]:


inp_data['Asymmetrique Activity Score'].value_counts(normalize = True, dropna = False)*100


# In[12]:


inp_data['Asymmetrique Profile Score'].value_counts(normalize = True, dropna = False)*100


# In[13]:


inp_data['How did you hear about X Education'].value_counts(normalize = True, dropna = False)*100


# In[14]:


# replacing Select with NaN
inp_data['How did you hear about X Education'] = inp_data['How did you hear about X Education'].replace('Select', np.nan)


# 
# #### Above mentioned columns data is mostly null so we can drop respective columns

# In[15]:


# dropping the respective columns
inp_data.drop(['Asymmetrique Activity Index', 'Asymmetrique Profile Score', 'Asymmetrique Activity Score', 'Asymmetrique Profile Index','How did you hear about X Education'], axis = 1, inplace = True)


# In[16]:


# checking data and maximium displaying columns

pd.set_option('display.max_columns', None)
inp_data.head(5)


# In[17]:


# checking the null values again
print(round(100*(inp_data.isnull().sum()/len(inp_data.index)), 2))


# In[18]:


# checking the country values
inp_data['Country'].value_counts(normalize = True, dropna = False)*100


# In[19]:


# checking city values
inp_data['City'].value_counts(normalize = True, dropna = False)*100


# In[20]:


# replacing Select with NaN
inp_data['City'] = inp_data['City'].replace('Select', np.nan)


# In[21]:


# checking city values
inp_data['City'].value_counts(normalize = True, dropna = False)*100


# #### We can take India as countries where City is Mumbai, Thane & Outskirts and  Other Cities of Maharashtra

# In[22]:


inp_data['Country'] = np.where(inp_data['City']=='Mumbai', 'India', inp_data['Country'])
inp_data['Country'] = np.where(inp_data['City']=='Thane & Outskirts', 'India', inp_data['Country'])
inp_data['Country'] = np.where(inp_data['City']=='Other Cities of Maharashtra', 'India', inp_data['Country'])


# In[23]:


# checking the country values
inp_data['Country'].value_counts(normalize = True, dropna = False)*100


# #### Checking other missing values columns

# In[24]:


# checking specialization column
inp_data['Specialization'].value_counts(normalize = True, dropna = False)*100


# In[25]:


# replacing Select with NaN
inp_data['Specialization'] = inp_data['Specialization'].replace('Select', np.nan)


# In[26]:



inp_data['Specialization'].value_counts(normalize = True, dropna = False)*100


# In[27]:


# changing the name of some columns for ease

inp_data.rename(columns = {'Total Time Spent on Website': 'time_on_website',  
                     'What is your current occupation': 'profession',
                    'What matters most to you in choosing a course' : 'course_selection_reason', 
                    'Receive More Updates About Our Courses': 'courses_updates', 
                     'Update me on Supply Chain Content': 'supply_chain_content_updates',
                    'Get updates on DM Content': 'dm_content_updates',
                    'I agree to pay the amount through cheque': 'cheque_payment',
                    'A free copy of Mastering The Interview': 'mastering_interview'}, inplace = True)


# In[28]:


inp_data['course_selection_reason'].value_counts(normalize = True, dropna = False)*100


# In[29]:


inp_data['profession'].value_counts(normalize = True, dropna = False)*100


# #### We will impute the values for columns specialization, course selection reason and profession as data is distributed

# In[30]:


# importing impute
import random

inp_data['Specialization'].fillna(random.choice(inp_data['Specialization'].notna()), inplace=True)
inp_data['course_selection_reason'].fillna(random.choice(inp_data['course_selection_reason'].notna()), inplace=True)
inp_data['profession'].fillna(random.choice(inp_data['profession'].notna()), inplace=True)


# #### As the most of city and country are Mumbai and India so filling the missing values

# In[31]:


inp_data['City'].fillna('Mumbai', inplace=True)
inp_data['Country'].fillna('India', inplace=True)


# In[32]:


# checking the missing values again
round(100*(inp_data.isnull().sum())/len(inp_data.index),2)


# In[33]:


# checking the lead profile

inp_data['Lead Profile'].value_counts(normalize = True, dropna = False)*100


# In[34]:


# replacing Select with NaN
inp_data['Lead Profile'] = inp_data['Lead Profile'].replace('Select', np.nan)


# In[35]:


inp_data['Lead Profile'].value_counts(normalize = True, dropna = False)*100


# #### The respective column is not of much significance. Therefore, we will drop respective column.

# In[36]:


# dropping the respective column
inp_data.drop(['Lead Profile'], axis = 1, inplace = True)


# In[37]:


# checking the missing values again
round(100*(inp_data.isnull().sum())/len(inp_data.index),2)


# In[38]:


# checking the page views per visit
inp_data['Page Views Per Visit'].describe()


# In[39]:


# filling the missing values with median values
inp_data['Page Views Per Visit'].fillna(inp_data['Page Views Per Visit'].median(), inplace=True)


# In[40]:


# checking the total visits
inp_data['TotalVisits'].describe()


# In[41]:


# similarly filling the missing values with median values
inp_data['TotalVisits'].fillna(inp_data['TotalVisits'].median(), inplace=True)


# In[42]:


# checking the Lead source values
inp_data['Lead Source'].value_counts(normalize = True, dropna = False)*100


# In[43]:


# we can see google two times with lower case and higher case

inp_data['Lead Source'] = inp_data['Lead Source'].replace('google', 'Google')
inp_data['Lead Source'].value_counts(normalize = True, dropna = False)*100


# In[44]:


# replacing the NaN values by median values such as Google
inp_data['Lead Source'].fillna(inp_data['Lead Source'].mode()[0], inplace=True)
inp_data['Lead Source'].value_counts(normalize = True, dropna = False)*100


# In[45]:


# checking dataframe again
round(100*(inp_data.isnull().sum())/len(inp_data.index),2)


# ### Data cleaning has been done here. We will start the analysis now.

# ## 4. Univariate Analysis (Categorical Variables)

# In[46]:


# checking some columns values 
inp_data['Do Not Email'].value_counts(normalize = True, dropna = False)*100


# In[47]:


inp_data['Do Not Call'].value_counts(normalize = True, dropna = False)*100


# In[48]:


inp_data['Search'].value_counts(normalize = True, dropna = False)*100


# In[49]:


inp_data['Newspaper Article'].value_counts(normalize = True, dropna = False)*100


# In[50]:


inp_data['X Education Forums'].value_counts(normalize = True, dropna = False)*100


# In[51]:


inp_data['Newspaper'].value_counts(normalize = True, dropna = False)*100


# In[52]:


inp_data['Digital Advertisement'].value_counts(normalize = True, dropna = False)*100


# In[53]:


inp_data['mastering_interview'].value_counts(normalize = True, dropna = False)*100


# In[54]:


inp_data['Through Recommendations'].value_counts(normalize = True, dropna = False)*100


# it is clear from above columns that following mentioned columns have almost one values and no value in term of analysis
# 
# * Do not call
# * Search
# * Newspaper Article
# * X education forum
# * Newspaper
# * Through recommnedation
# 
# We are not dropping other columns as they still contains some data that is useful.

# In[55]:


# dropping respective mentioned columns
inp_data.drop(['Through Recommendations', 'Newspaper', 'X Education Forums', 'Newspaper Article', 'Search','Do Not Call'], axis = 1, inplace = True)


# #### checking more columns for values

# In[56]:


# checking some columns values 
inp_data['Magazine'].value_counts(normalize = True, dropna = False)*100


# In[57]:


inp_data['courses_updates'].value_counts(normalize = True, dropna = False)*100


# In[58]:


inp_data['supply_chain_content_updates'].value_counts(normalize = True, dropna = False)*100


# In[59]:


inp_data['dm_content_updates'].value_counts(normalize = True, dropna = False)*100


# In[60]:


inp_data['cheque_payment'].value_counts(normalize = True, dropna = False)*100


# #### All the above columns just contain one value and there is not other data there. So there is value in analysis. So we will drop them.

# In[61]:


# dropping respective mentioned columns
inp_data.drop(['Magazine', 'courses_updates', 'supply_chain_content_updates', 'dm_content_updates', 'cheque_payment'], axis = 1, inplace = True)


# In[62]:


# checking the dataframe
inp_data.head()


# ## 5. Univariate Analysis (Numeric Variables)

# In[63]:


# plotting the course selection reason

cs=sns.countplot(inp_data['course_selection_reason'], hue=inp_data['Converted'])
cs.set_xticklabels(cs.get_xticklabels(),rotation=90)
plt.title('Course Selection Reasons')


# In[64]:


# plotting the profession 

pf=sns.countplot(inp_data['profession'], hue=inp_data['Converted'])
pf.set_xticklabels(pf.get_xticklabels(),rotation=90)
plt.title('Profession')


# In[65]:


# plotting the specialization
sp=sns.countplot(inp_data['Specialization'], hue=inp_data['Converted'])
sp.set_xticklabels(sp.get_xticklabels(),rotation=90)
plt.title('Specialization')


# In[66]:


#plotting Lead origin and converted

lo=sns.countplot(inp_data['Lead Origin'], hue=inp_data['Converted'])
lo.set_xticklabels(lo.get_xticklabels(),rotation=90)
plt.title('Lead Origin')
plt.show()


# In[67]:


# checking the lead source
ls=sns.countplot(inp_data['Lead Source'], hue=inp_data['Converted'])
ls.set_xticklabels(ls.get_xticklabels(),rotation=90)
plt.title('Lead Source')


# In[68]:


# plotting the country distribution

cy=sns.countplot(inp_data['Country'], hue=inp_data['Converted'])
cy.set_xticklabels(cy.get_xticklabels(),rotation=90)
plt.title('Country')


# In[69]:


# plotting the cities

ci=sns.countplot(inp_data['City'], hue=inp_data['Converted'])
ci.set_xticklabels(ci.get_xticklabels(),rotation=90)
plt.title('City')


# ### Insights from Above Charts
# * Courses are predominately selected for better career prospects.
# * Unemployed people are the highest category of professional that are potential customers.
# * Mostly people are interested in International business management specialization courses.
# * Landing page submission are highest for potential customer.
# * Mostly customers are coming to website from Google search.
# * Most traffic is predominately coming from India specially from Mumbai city.

# ## 6. Multivariate Analysis

# In[70]:


inp_data.head()


# In[71]:


sns.pairplot(inp_data, vars=['TotalVisits','time_on_website','Page Views Per Visit'])
plt.show()


# ### Observations:
# * It is clear that there are high values in the data specially time on web
# * We need to handle the outliers
# 

# In[72]:


# making the correlation matrix

sns.heatmap(inp_data.corr(), cmap="YlOrRd", annot = True)


# #### Findings
# * Page views per visit has negative correlation with coverted
# * Time spend on website has strong and postive correlation with converted.

# In[73]:


# building the boxplots to check the values
# building the total visit box plot

sns.boxplot(inp_data.TotalVisits)


# In[74]:


# building the total visit box plot

sns.boxplot(inp_data.time_on_website)


# In[75]:


# checking page views per visit box plot
sns.boxplot(inp_data['Page Views Per Visit'])


# #### it is clear from above boxplots that
# * Total visits and page views per visit got the outliers that can be treated by capping them to 99 percentile.
# * Moreover, a user can view or vist a page many times and total visits are be more. So we will leave the outliers as it is.

# ## 7. Dummy Variables

# In[76]:


inp_data.info()


# In[77]:


inp_data.head(5)


# In[78]:


# changing the values of "Yes" or "No" to 1 and 0 for catgorey columns

cat_cols = ['Do Not Email', 'mastering_interview','Digital Advertisement']
# Defining the function for the map
def change_value(x):
    return x.map({'Yes': 1, "No": 0})

inp_data[cat_cols] = inp_data[cat_cols].apply(change_value)
inp_data.head()


# In[79]:


# creating the dummy variables for category columns

cat_cols_details = ['Lead Origin', 'Lead Source', 'Country', 'Specialization', 'profession', 'course_selection_reason','City']
dummy = pd.get_dummies(inp_data[cat_cols_details], drop_first = True)

# Adding the results to the lead_data dataframe
inp_data = pd.concat([inp_data, dummy], axis=1)


# In[80]:


inp_data.head()


# In[81]:


# droping columns for which dummy's has been created
inp_data.drop(cat_cols_details, axis=1, inplace=True)


# In[82]:


inp_data.columns


# ## 8. Train Test Split

# In[83]:


# testing and training data split
X=inp_data.drop('Converted', axis=1)
y=inp_data['Converted']
X.head()


# In[84]:


# Split the dataset into 70% and 30% for train and test respectively
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=10)
print("X Train Shape",X_train.shape)
print("X Test Shape",X_test.shape)
print("Y Train Shape",y_train.shape)
print("Y Test Shape",y_test.shape)


# In[85]:


X_train.head()


# scaling the columns total visits, time on websites and page views per vist

# In[86]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train[['TotalVisits', 'time_on_website', 'Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits', 'time_on_website', 'Page Views Per Visit']])
X_train.head()


# ## 9. Feature Selection & Model building

# In[87]:


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
# Initialize model
lr = LogisticRegression()
# Lets select 1st 15 features to start with
rfe = RFE(lr, step=15)
rfe = rfe.fit(X_train, y_train)
# Features that have been selected by RFE
list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[88]:


# Put all the columns selected by RFE in the variable 'col'
columns2 = X_train.columns[rfe.support_]
columns2


# checking the model with StatsModel

# In[89]:


# 1st X_train with only columns provided by RFE
import statsmodels.api as sm
X_train_rfe = X_train[columns2]
X_train_sm = sm.add_constant(X_train_rfe)
logm1 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[90]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# #### Creating the dataframe from prediction probabilities and converted flag

# In[91]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_Prob':y_train_pred})
y_train_pred_final.head()


# In[92]:


# Creating with potential convertable customer if Converted_Prob > 0.5 else 0

y_train_pred_final['predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[93]:


# Let's check the overall accuracy.
from sklearn.metrics import confusion_matrix
from sklearn import tree, model_selection, metrics
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted)*100)


# #### It is clear from above answer that model accuracy is around 80.16%

# In[94]:


# Make a VIF dataframe for all the variables present
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Features'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# #### please note following features have very high VIF
# 
# * course_selection_reason_Better Career Prospects
# * Lead Origin_Landing Page Submission

# In[95]:


# so dropping the them and checking again
columns2 = columns2.drop('course_selection_reason_Better Career Prospects', 1)
columns2


# In[96]:


# checking VIF Again

X_train_rfe_2 = X_train[columns2]
vif = pd.DataFrame()
vif['Features'] = X_train_rfe_2.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe_2.values, i) for i in range(X_train_rfe_2.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[97]:


# now the Lead Origin_Landing Page Submission have high VIF, so repeating the process dropping it and checking again

columns2 = columns2.drop('Lead Origin_Landing Page Submission', 1)
X_train_rfe_3 = X_train[columns2]
vif = pd.DataFrame()
vif['Features'] = X_train_rfe_3.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe_3.values, i) for i in range(X_train_rfe_3.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# #### now the VIF seems to be under control
# #### Let making the model using features

# In[98]:


X_train_sm3 = sm.add_constant(X_train_rfe_3)
logm3 = sm.GLM(y_train, X_train_sm3, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# Dropping the all the columns whose P value is near to 1 or near

# In[99]:



columns2=columns2.drop(['Lead Source_NC_EDM','Lead Source_WeLearn','Lead Source_bing','Country_Germany','Country_Italy','Country_Nigeria','Country_Sweden','profession_Housewife','Specialization_Services Excellence'])


# In[100]:


X_train_rfe_4 = X_train[columns2]
X_train_sm4 = sm.add_constant(X_train_rfe_4)
logm4 = sm.GLM(y_train, X_train_sm4, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[101]:


columns2 = columns2.drop(['Country_France','Lead Source_Facebook','Country_Oman','Country_Singapore','Country_South Africa','Specialization_Healthcare Management','profession_Student'], 1)
X_train_rfe_5 = X_train[columns2]
X_train_sm5 = sm.add_constant(X_train_rfe_5)
logm5 = sm.GLM(y_train, X_train_sm5, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# In[102]:


vif = pd.DataFrame()
vif['Features'] = X_train_rfe_5.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe_5.values, i) for i in range(X_train_rfe_5.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[103]:


# checking accuracy after dropping columns
y_train_pred = res.predict(X_train_sm5)
y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_Prob':y_train_pred})
y_train_pred_final['predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)
# Creating confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)


# In[104]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted)*100)


# In[106]:


# accuracy not dropped much. dropping some more columns

columns2 = columns2.drop(['Lead Source_Referral Sites','City_Other Metro Cities','City_Other Cities of Maharashtra'], 1)
X_train_rfe_6 = X_train[columns2]
X_train_sm6 = sm.add_constant(X_train_rfe_6)
logm5 = sm.GLM(y_train, X_train_sm6, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# In[107]:


#Lets check accuracy after dropping these many features
y_train_pred = res.predict(X_train_sm6)
y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_Prob':y_train_pred})
y_train_pred_final['predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)
# Creating confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)


# In[109]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted)*100)


# ## 10. Model Evaluation

# In[110]:


y_train_pred = res.predict(X_train_sm6)
y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_Prob':y_train_pred})
y_train_pred_final['predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)
# Creating confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# In[111]:


# Metrics beyond simply accuracy

# Substituting the value of true positive
TP = confusion[1,1]
# Substituting the value of true negatives
TN = confusion[0,0]
# Substituting the value of false positives
FP = confusion[0,1] 
# Substituting the value of false negatives
FN = confusion[1,0]


# In[112]:


# Calculating the sensitivity
print("sensitivity=", round(TP*100/(TP+FN), 2), "%")
# Calculating the specificity
print("specificity=", round(TN*100/(TN+FP), 2), "%")


# With the current cut off as 0.5 we have around 79.7% accuracy, sensitivity of around 65.35% and specificity of around 88.85%.

# In[114]:


# ROC function( shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[115]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_Prob, drop_intermediate = False )


# In[116]:


# Call the ROC function
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)


# In[117]:


# area is under 0.86 that is good
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[118]:


# Creating a dataframe to see the values of accuracy, sensitivity, and specificity at different values of probabiity cutoffs
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
# Making confusing matrix to find values of sensitivity, accurace and specificity for each level of probablity

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
cutoff_df


# In[119]:


cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[120]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_Prob.map( lambda x: 1 if x > 0.3 else 0)
y_train_pred_final.head()


# In[121]:


confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# In[124]:


# Substituting the value of true positive
TP = confusion[1,1]
# Substituting the value of true negatives
TN = confusion[0,0]
# Substituting the value of false positives
FP = confusion[0,1] 
# Substituting the value of false negatives
FN = confusion[1,0]

print("Accuracy=",round(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted)*100,2), "%")
# Calculating the sensitivity
print("sensitivity=", round(TP*100/(TP+FN), 2), "%")
# Calculating the specificity
print("specificity=", round(TN*100/(TN+FP), 2), "%")


# ## 11. Prediction on Test

# In[123]:


# Scaling (only the transform not the fit) on test set
#X_trn, X_test, y_trn, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=10)
X_test[['TotalVisits', 'time_on_website', 'Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits', 'time_on_website', 'Page Views Per Visit']])
X_test.head()


# In[126]:


# Selecting only those columns for test set, which are available in train set
X_test = X_test[columns2]
X_test.head()


# In[127]:


X_test_sm = sm.add_constant(X_test)
X_test_sm


# In[128]:


# Prdicting

y_test_pred = res.predict(X_test_sm)
y_pred_df = pd.DataFrame(y_test_pred)
y_test_df = pd.DataFrame(y_test)
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)
y_pred_final= y_pred_final.rename(columns = {0 : 'Converted_Prob'})
y_pred_final.head()


# In[129]:


# Making prediction using cut off 0.3
y_pred_final['final_predicted'] = y_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.3 else 0)
y_pred_final


# In[130]:


print(metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted))

# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[131]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]

print("Accuracy=",round(metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)*100,2), "%")
# Calculating the sensitivity
print("sensitivity=", round(TP*100/(TP+FN), 2), "%")
# Calculating the specificity
print("specificity=", round(TN*100/(TN+FP), 2), "%")


# In[132]:


# Precision-Recall calculation for Train dataset

# Precision = TP / TP + FP
print("Precision(Train)=", round(confusion[1,1]/(confusion[0,1]+confusion[1,1])*100,2), "%")
#Recall = TP / TP + FN
print("Recall(Train)=", round(confusion[1,1]/(confusion[1,0]+confusion[1,1])*100,2),"%")


# In[133]:


p, r, thresholds = metrics.precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)


# In[134]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# #### The Precision-Recall curve shows that the cut off is at 0.35

# In[135]:


# lets predict with ths cut off value
y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_Prob.map( lambda x: 1 if x > 0.35 else 0)
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
# Substituting the value of true positive
TP = confusion[1,1]
# Substituting the value of true negatives
TN = confusion[0,0]
# Substituting the value of false positives
FP = confusion[0,1] 
# Substituting the value of false negatives
FN = confusion[1,0]

print("Accuracy=",round(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted)*100,2), "%")
# Calculating the sensitivity
print("sensitivity=", round(TP*100/(TP+FN), 2), "%")
# Calculating the specificity
print("specificity=", round(TN*100/(TN+FP), 2), "%")


# With the current cut off as 0.35 
# we have around 79.72% accuracy, sensitivity of around 65.35% and specificity of around 88.85%

# In[137]:


# Prediction on test data with this new cutoff value of 0.35
# Making prediction using cut off 0.35
y_pred_final['final_predicted'] = y_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.35 else 0)
y_pred_final
# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2
# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]
print("Accuracy=",round(metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)*100,2), "%")
# Calculating the sensitivity
print("sensitivity=", round(TP*100/(TP+FN), 2), "%")
# Calculating the specificity
print("specificity=", round(TN*100/(TN+FP), 2), "%")


#  With the current cut off as 0.35 we have accuracy=79.22%, sensitivity=74.88% and specificity of around 81.68% for test dataset

# In[ ]:




