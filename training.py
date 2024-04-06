from data_processing_features import get_data, convert_to_numeric, data_transformation
from model_building import train_test_val_split, balance_imbalance_data, fit_and_evaluate_model
import joblib

# get data from the database
fraud_df = get_data('Fraud_detection')


# convert columns to numarical
num_col = ['step', 'amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFlaggedFraud', 'isFraud']
fraud_df = convert_to_numeric(fraud_df, num_col)
fraud_df = data_transformation(fraud_df)

# split train and test data
x_train, x_test, x_val, y_train, y_test, y_val = train_test_val_split(fraud_df)

x_train, y_train = balance_imbalance_data(x_train, y_train)

model = fit_and_evaluate_model(x_train, x_test, y_train, y_test)
print("completed")
joblib.dump(model , 'model_classifier.pkl')
print('model saved')