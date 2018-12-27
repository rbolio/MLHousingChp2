from mlhousingchp2.housingInterface import load_housing_data
from mlhousingchp2.dataUtilities import split_train_test_by_id

# fetch_housing_data()
housing = load_housing_data()

housing.describe()

# print data graphs
# data_histograms(housing)


# For random sampling, this may create sampling bias!
# to separate train and test data locally
housing_with_id = housing.reset_index()
housing_with_id['id'] = housing['longitude'] * 1000 + housing['latitude']
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

# to separate train and test data with sklearn:
"""
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, rnadom_state=42)
"""

# stoping at page 52
