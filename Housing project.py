
# coding: utf-8

# In[1]:


import os
import tarfile
from six.moves import urllib


# In[2]:


DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[3]:


fetch_housing_data()
import pandas as pd 
def Data_Loading(housing_path = HOUSING_PATH) :
    CSV_path = os.path.join(housing_path , 'housing.csv')
    return pd.read_csv(CSV_path)


# In[4]:


dataset = Data_Loading()
dataset.head()
dataset.info()
dataset['ocean_proximity'].value_counts()
dataset.describe(include=['O'])
dataset.describe()


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
dataset.hist(bins=50, figsize = (20 ,15))
plt.show()


# In[6]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)


# In[7]:


import numpy as np
dataset["income_cat"] = np.ceil(dataset["median_income"] / 1.5)
dataset["income_cat"].where(dataset["income_cat"] < 5, 5.0, inplace=True)
# dataset[['income_cat' , 'median_income']]
# dataset['income_cat'].unique()
dataset["income_cat"].value_counts()
dataset["income_cat"].hist()
#dataset["income_cat"].value_counts() /len(dataset)


# In[8]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset["income_cat"]):
    strat_train_set = dataset.loc[train_index]
    strat_test_set = dataset.loc[test_index]


# In[9]:


train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
def generate_portions(dataframe) :
    return dataframe["income_cat"].value_counts() / len(dataframe)
test_methods_comparison =pd.DataFrame({'OverAll' : generate_portions(dataset) , 'Random_Method':generate_portions(test_set),
                                      'strat_Method':generate_portions(strat_test_set)}).sort_index()
test_methods_comparison["Rand. %error"] = 100 * test_methods_comparison["Random_Method"] / test_methods_comparison["OverAll"] - 100
test_methods_comparison["Strat. %error"] = 100 * test_methods_comparison["strat_Method"] / test_methods_comparison["OverAll"] - 100
test_methods_comparison


# In[10]:


housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y='latitude')
plt.savefig('expl.png', format='png', dpi=1200)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# In[11]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="income_cat", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
plt.savefig('mix.png', format='png', dpi=1200)


# In[12]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
plt.savefig('mix2.png', format='png', dpi=1200)


# In[13]:



import matplotlib.image as mpimg
california_img=mpimg.imread('california.png')
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="Population",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
plt.show()


# In[14]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[15]:


from pandas.plotting import scatter_matrix            
scatter_matrix(housing[["median_house_value", "median_income", "total_rooms","housing_median_age"]], figsize=(12, 8))
plt.savefig('pairplot.png', format='png', dpi=1200)


# In[16]:


housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=.7)


# In[17]:


housing = housing.drop("income_cat", axis=1)
housing


# In[18]:


X = strat_train_set.drop(["median_house_value",'income_cat'], axis=1) # drop labels for training set
Y = strat_train_set['median_house_value']
XNUM = X.drop('ocean_proximity', axis=1)


# In[19]:


# Taking care of missing data
from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
# imputer.fit(XNUM)
# X = imputer.transform(XNUM)
# Xt = pd.DataFrame(X, columns=XNUM.columns)
# Xt.describe()


# In[20]:


X_cat = X[['ocean_proximity']]
list(X_cat)


# In[21]:


# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
# housing_cat_encoded = encoder.fit_transform(housing_cat.values.reshape(-1,))
# print(encoder.classes_)


# In[22]:


# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder()
# housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
# housing_cat_1hot.toarray() #to convert from Scip matrix to numpy array 


# In[23]:


#Option (2)By ,no need for hot encoder and result will be Numpy array by default
# from sklearn.preprocessing import LabelBinarizer
# encoder = LabelBinarizer()
# housing_cat_1hot = encoder.fit_transform(housing_cat)
# housing_cat_1hot


# In[24]:


from sklearn.base import BaseEstimator, TransformerMixin
# column index from the main dataset 
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin): #creat a class with some objects
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs (bed rooms per rooms will be optional for ML)
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
# housing_extra_attribs = attr_adder.transform(X.values)
# housing_extra_attribs


# In[25]:


# housing_extra_attribs = pd.DataFrame(
#     housing_extra_attribs,
#     columns=list(X.columns)+["rooms_per_household", "population_per_household"])
# housing_extra_attribs.head()


# In[26]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),])


# In[27]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names=attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[28]:


from sklearn import preprocessing
class CustomBinarizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None,**fit_params):
        return self
    def transform(self, X):
        return preprocessing.LabelBinarizer().fit(X).transform(X)


# In[29]:


from sklearn import preprocessing
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer
num_attribs = list(XNUM)
cat_attribs = list(X_cat)
num_pipeline = Pipeline([
('selector', DataFrameSelector(num_attribs)),
('imputer', Imputer(strategy="median")),
('attribs_adder', CombinedAttributesAdder()),
('std_scaler', StandardScaler()),
])
cat_pipeline = Pipeline([
('selector', DataFrameSelector(cat_attribs)),
('Custom_Binarizer', CustomBinarizer()),
])
full_pipeline = FeatureUnion(transformer_list=[
("num_pipeline", num_pipeline),
("cat_pipeline", cat_pipeline),
])


# In[30]:


housing_prepared = full_pipeline.fit_transform(X)


# In[31]:


X.shape


# In[32]:


housing_prepared.shape


# In[33]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
regressor = lin_reg.fit(housing_prepared, Y)
prediction = regressor.predict(housing_prepared)


# In[34]:


from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(Y, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[35]:


from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(Y, housing_predictions)
lin_mae


# In[36]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
regresor = tree_reg.fit(housing_prepared, Y)


# In[37]:


housing_predictions = regresor.predict(housing_prepared)
tree_mse = mean_squared_error(Y, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[38]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, Y,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[39]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)


# In[40]:


lin_scores = cross_val_score(lin_reg, housing_prepared, Y,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[43]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
regresor = forest_reg.fit(housing_prepared, Y)
scores = cross_val_score(forest_reg, housing_prepared, Y,
                         scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)


# In[46]:


# fine and tune your model
from sklearn.model_selection import GridSearchCV
param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, Y)


# In[52]:


final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse

