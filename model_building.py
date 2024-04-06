from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from collections import Counter

# test train split 
def train_test_val_split(fraud_df):
    y = fraud_df['isFraud']
    x = fraud_df.drop('isFraud',axis=1)
    # x = pd.get_dummies(x, columns=['type'], drop_first = True)
    x_train, y_train = x[:4000000],y[:4000000]
    x_test, y_test = x[4000000:5000000],y[4000000:5000000]
    x_val, y_val = x[5000000:],y[5000000:]
 
    return x_train, x_test, x_val, y_train, y_test, y_val

def fit_and_evaluate_model(x_train, x_test, y_train, y_test,max_depth=5,min_samples_split=0.01,max_features=0.8,max_samples=0.8):
    random_forest =  RandomForestClassifier(random_state=0,\
                                            max_depth=max_depth,\
                                            min_samples_split=min_samples_split,\
                                            max_features=max_features,\
                                            max_samples=max_samples,\
                                            class_weight={0:1,1:100}, )

    model = random_forest.fit(x_train, y_train)
    random_forest_predict = random_forest.predict(x_test)
    random_forest_conf_matrix = confusion_matrix(y_test, random_forest_predict)
    random_forest_acc_score = accuracy_score(y_test, random_forest_predict)
    print("confussion matrix")
    print(random_forest_conf_matrix)
    print("\n")
    print("Accuracy of Random Forest:",random_forest_acc_score*100,'\n')
    print(classification_report(y_test,random_forest_predict))
    return model

def balance_imbalance_data(x_train, y_train):
    #Handling Imbalance Data
    # Define the percentage of oversampling 
    sampling_strategy = 0.2  # number between 0 to 1 
    # after resampling minority class would be 20% of majority class

    # Apply SMOTE with specified oversampling percentage
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    x_res, y_res = smote.fit_resample(x_train, y_train)
    print('Resampled dataset shape %s' % Counter(y_res))
    return x_res, y_res

# def get_important_features(model, features):
