import pickle

history_path = '/media/akshay/My Book/Crossing_Intention_Prediction-main/data/models/jaad/PPCIM/03Aug2023-15h50m07s/history.pkl'

with open(history_path, 'rb') as fid:
    history = pickle.load(fid)
    #print(history)

val_loss = history['loss']
best_epoch = val_loss.index(min(val_loss)) + 1
best_accuracy = history['accuracy'][best_epoch - 1]
best_loss = history['loss'][best_epoch - 1]

print("Best Epoch: ", best_epoch)
print("Best Accuracy: ", best_accuracy)
print("Best Loss: ", best_loss)
import pickle

with open(history_path, 'rb') as fid:
    history = pickle.load(fid)

accuracy = history['accuracy']
highest_accuracy = max(accuracy)
highest_accuracy_epoch = accuracy.index(highest_accuracy) + 1

print("Highest Accuracy: ", highest_accuracy)
print("Epoch of Highest Accuracy: ", highest_accuracy_epoch)


