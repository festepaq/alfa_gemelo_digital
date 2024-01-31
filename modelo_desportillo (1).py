import sys
import ast
import numpy as np
import xgboost as xgb

# Load the XGBoost model from file
model = xgb.Booster(model_file='xgboost11vars.model')
input = np.zeros((1,11))


while True:
    input_dictionary = sys.stdin.readline()
    # Replace "null" with "0"
    input_dictionary = input_dictionary.replace("null", "0")
    dictionary = ast.literal_eval(input_dictionary)
    input = np.array(list(dictionary.values()))
    input = input.reshape(1,11)
    output = model.predict(xgb.DMatrix(input))
    print(str(np.round(output* 100, decimals=2)))