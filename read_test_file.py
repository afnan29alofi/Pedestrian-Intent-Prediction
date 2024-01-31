import pickle

# Load the data from the pickle file
with open('/media/akshay/My Book/Crossing_Intention_Prediction-main/data/models/pie/PPCI_att_mult/24Sep2023-17h31m43s/test_output.pkl', 'rb') as file:
    data = pickle.load(file)


print(data)
