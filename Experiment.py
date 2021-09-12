from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn


warnings.filterwarnings('ignore')


class MultiLayer_Perceptron_Classifier:
    def __init__(self, alpha=0.001, theta=0.0001):
        self.bias = 1
        self.alpha = alpha
        self.theta = theta
        self.x = {}
        self.z = {}
        self.b_z = {}
        self.weights_z = {}
        self.net_input_z = {}
        self.net_input_y = {}
        self.b_y = {}
        self.weights_y = {}
        self.output_y = {}


    def sigmoid(self, x):
        #return (1 / (1 + np.exp(-x))) # binary sigmoid
        return (2 / (1 + np.exp(-x))) - 1 # bipolar sigmoid


    def sigmoid_derivative(self, f):
        #return f * (1 - f) # binary sigmoid derivative
        return 0.5 * (1 + f) * (1 - f) # bipolar sigmoid derivative


    def instantiate_weights(self, length, labels):
        length = length
        labels = labels
        temporary = []
        for i in range(0, length):
            for j in range(0, length):
                temporary.append(np.random.uniform(-0.5, 0.5))

            self.weights_z[i] = temporary
            self.b_z[i] = np.random.uniform(-0.5, 0.5)

        temporary = []
        for label in labels:
            temporary = []
            for j in range(0, length):
                temporary.append(np.random.uniform(-0.5, 0.5))

            self.weights_y[label] = temporary
            self.b_y[label] = np.random.uniform(-0.5, 0.5)


    def feedforward(self, x_data, y_data):
        record = x_data
        label = y_data

        # for input layer:
        for j in range(0, len(record)):
            self.x[j] = record[j]

        # for hidden layer:
        for j in range(0, len(record)):
            temporary = self.weights_z[j] # temporary now is a list
            weighted_sum = 0
            for i in range(0, len(record)):
                weighted_sum += temporary[i] * self.x[i]

            self.z[j] = self.b_z[j] + weighted_sum
            self.net_input_z[j] = self.sigmoid(self.z[j])

        # for output layer:
        for j in range(0, len(record)):
            temporary = self.weights_y[label]
            weighted_sum = 0
            for i in range(0, len(record)):
                weighted_sum += temporary[i] * self.net_input_z[i]

            self.net_input_y[label] = self.b_y[label] + weighted_sum
            self.output_y[label] = self.sigmoid(self.net_input_y[label])


    def backpropagation(self, x_train, y_train):
        record = x_train
        label = y_train

        delta_factor_y = (label - self.output_y[label]) * self.sigmoid_derivative(self.output_y[label])

        delta_weights_zy = {}
        for j in range(0, len(record)):
            delta_weights_zy[j] = self.alpha * self.z[j] * delta_factor_y

        delta_b_y = self.alpha * delta_factor_y

        weighted_sum = 0
        for j in range(0, len(record)):
            weighted_sum += delta_factor_y * self.weights_y[label][j]

        delta_factor_z = {}
        for j in range(0, len(record)):
            delta_factor_z[j] = weighted_sum * self.sigmoid_derivative(self.net_input_z[j])

        delta_weights_xz = {}
        for j in range(0, len(record)):
            delta_weights_xz[j] = self.alpha * self.x[j] * delta_factor_z[j]

        delta_b_z = {}
        for j in range(0, len(record)):
            delta_b_z[j] = self.alpha * delta_factor_z[j]

        temporary = []
        next_weights_y = 0
        for j in range(0, len(record)):
            next_weights_y = self.weights_y[label][j] + delta_weights_zy[j]
            temporary.append(next_weights_y)

        self.weights_y[label] = temporary

        for j in range(0, len(record)):
            self.weights_z[j] = self.weights_z[j] + delta_weights_xz[j]

        self.b_y[label] += delta_b_y

        for j in range(0, len(record)):
            self.b_z[j] += delta_b_z[j]

    def fit(self, x_train, y_train, epochs=2):
        x_train = x_train
        y_train = y_train
        epochs = epochs
        self.output_y[0] = 0
        self.output_y[1] = 0

        for i in range(0, epochs):
            for j in range(0, len(x_train)):
                if j == 0:
                    self.instantiate_weights(len(x_train[j]), np.unique(y_train))

                for label in np.unique(y_train):
                    self.feedforward(x_train[j], y_train[j])
                    self.backpropagation(x_train[j], label)

            # for batch-updating. if we want to do this, we have to change the called method too.
            #self.backpropagation(x_train[j], y_train[j])

    def predict(self, x_test, y_test):
        x_test = x_test
        y_test = y_test
        results = []

        for j in range(0, len(x_test)):
            self.feedforward(x_test[j], y_test[j])
            key = max(self.output_y, key=self.output_y.get)
            results.append(key)

        return results

    def score(self, y_test, predictions):
        predictions = predictions
        y_test = y_test
        true_counts = 0
        false_counts = 0
        for j in range(0, len(y_test)):
            if y_test[j] == predictions[j]:
                true_counts += 1
            else:
                false_counts += 1

        print(f'True Counts: {true_counts}')
        print(f'False Counts: {false_counts}')
        accuracy = float(true_counts) / float(true_counts + false_counts)
        return accuracy


class MultiClass_Perceptron_Classifier:
    def __init__(self, alpha=0.001, theta=0):
        self.bias = 1
        self.alpha = alpha
        self.theta = theta
        self.weights = {}
        self.b = {}
        self.x = []
        self.y = 0
        self.net_input = {}
        self.output = {} # indices of this list ==> 0: X and 1: O ... and fyi, label values ==> X =: 1 and O =: -1

    def fit(self, x_train, y_train):
        x_train = x_train
        y_train = y_train
        for k in range(0, len(x_train)):
            record = x_train[k]
            for label in np.unique(y_train): # {1: all the stuff about label X ... -1: all the stuff about label O}
                if k == 0:
                    self.b[label] = 0
                    temporary = [0] * len(record)
                    self.weights[label] = temporary

                x = {}
                for j in range(0, len(record)):
                    x[j] = record[j]

                weighted_sum = 0
                for j in range(0, len(record)):
                    weighted_sum += self.weights[label][j] * x[j]

                self.net_input[label] = self.b[label] + weighted_sum

                if self.net_input[label] > self.theta:
                    exit_neuron = 1
                elif self.net_input[label] <= self.theta and self.net_input[label] >= -self.theta: # how to get rid of this?
                    exit_neuron = 0
                elif self.net_input[label] < -self.theta:
                    exit_neuron = -1

                if exit_neuron == 1:
                    self.output[label] = 1
                else:
                    self.output[label] = 0

                if self.output[label] != 1:
                    delta_weights = {}
                    for j in range(0, len(record)):
                        delta_weights[j] = self.alpha * x[j] * label

                    temporary = []
                    next_weights = 0
                    for j in range(0, len(record)):
                        next_weights = self.weights[label][j] + delta_weights[j]
                        temporary.append(next_weights)

                    self.weights[label] = temporary

                    delta_b = self.alpha * self.bias * label
                    self.b[label] += delta_b

    def predict(self, x_test, y_test):
        x_test = x_test
        y_test = y_test
        results = []
        for k in range(0, len(x_test)):
            record = x_test[k]
            for label in np.unique(y_train):
                x = {}
                for j in range(0, len(record)):
                    x[j] = record[j]

                weighted_sum = 0
                for j in range(0, len(record)):
                    weighted_sum += self.weights[label][j] * x[j]

                self.net_input[label] = self.b[label] + weighted_sum

                if self.net_input[label] > self.theta:
                    exit_neuron = 1
                elif self.net_input[label] <= self.theta and self.net_input[label] >= -self.theta: # how to get rid of this?
                    exit_neuron = 0
                elif self.net_input[label] < -self.theta:
                    exit_neuron = -1

                if exit_neuron == 1:
                    self.output[label] = 1
                else:
                    self.output[label] = 0

            if self.output[0] == self.output[1] == self.output[2] or self.output[0] == self.output[1] or self.output[0] == self.output[2] or self.output[1] == self.output[2]:
                key = max(self.net_input, key=self.net_input.get)
                self.output[key] == 1
                other_label = list(self.output.keys())
                other_label.remove(key)
                for item in other_label:
                    self.output[item] = 0

            key = max(self.output, key=self.output.get)
            results.append(key)

        return results

    def score(self, y_test, predictions):
        predictions = predictions
        y_test = y_test
        true_counts = 0
        false_counts = 0
        for j in range(0, len(y_test)):
            if y_test[j] == predictions[j]:
                true_counts += 1
            else:
                false_counts += 1

        print(f'True Counts: {true_counts}')
        print(f'False Counts: {false_counts}')
        accuracy = float(true_counts) / float(true_counts + false_counts)
        return accuracy


class MultiClass_Adaline_Classifier:
    def __init__(self, alpha=0.000001, threshold=0.0001):
        self.bias = 1
        self.alpha = alpha
        self.threshold = threshold
        self.weights = {}
        self.b = {}
        self.x = {}
        self.y = 0
        self.net_input = {}
        self.output = {} # indices of this list ==> 0: X and 1: O ... and fyi, label values ==> X =: 1 and O =: -1
        self.maximum_delta_weights = {}
        self.condition = False

    def fit(self, x_train, y_train):
        x_train = x_train
        y_train = y_train
        while self.condition == False:
            for k in range(0, len(x_train)):
                record = x_train[k]
                for label in np.unique(y_train):
                    if k == 0:
                        self.b[label] = 0
                        temporary = [0] * len(record)
                        self.weights[label] = temporary
                        for j in range(0, len(record)):
                            self.maximum_delta_weights[j] = 0

                    for j in range(0, len(record)):
                        self.x[j] = record[j]

                    weighted_sum = 0
                    for j in range(0, len(record)):
                        weighted_sum += self.weights[label][j] * self.x[j]

                    self.net_input[label] = self.b[label] + weighted_sum

                    delta_weights = {}
                    for j in range(0, len(record)):
                        delta_weights[j] = self.alpha * self.x[j] * (label - self.net_input[label])
                        if abs(delta_weights[j]) > abs(self.maximum_delta_weights[j]):
                            self.maximum_delta_weights[j] = abs(delta_weights[j])

                    temporary = []
                    next_weights = 0
                    for j in range(0, len(record)):
                        next_weights = self.weights[label][j] + delta_weights[j]
                        temporary.append(next_weights)

                    self.weights[label] = temporary

                    delta_b = self.alpha * self.bias * (y_train[k] - label)
                    self.b[label] += delta_b

            max_difference = max(self.maximum_delta_weights.values())
            if abs(max_difference) < self.threshold:
                #print('Breaking out the algorithm...')
                self.condition = True
                break
            else:
                self.maximum_delta_weights = {key: 0 for key in self.maximum_delta_weights}

    def predict(self, x_test, y_test):
        x_test = x_test
        y_test = y_test
        results = []
        for k in range(0, len(x_test)):
            record = x_test[k]
            for label in np.unique(y_test):
                for j in range(0, len(record)):
                    self.x[j] = record[j]

                weighted_sum = 0
                for j in range(0, len(record)):
                    weighted_sum += self.weights[label][j] * self.x[j]

                self.net_input[label] = self.b[label] + weighted_sum

                if self.net_input[label] >= 0:
                    self.output[label] = 1
                else:
                    self.output[label] = 0

            if self.output[0] == self.output[1] == self.output[2] or self.output[0] == self.output[1] or self.output[0] == self.output[2] or self.output[1] == self.output[2]:
                key = max(self.net_input, key=self.net_input.get)
                self.output[key] == 1
                other_label = list(self.output.keys())
                other_label.remove(key)
                for item in other_label:
                    self.output[item] = 0

            key = max(self.output, key=self.output.get)
            results.append(key)

        return results


    def score(self, y_test, predictions):
        predictions = predictions
        y_test = y_test
        true_counts = 0
        false_counts = 0
        for j in range(0, len(y_test)):
            if y_test[j] == predictions[j]:
                true_counts += 1
            else:
                false_counts += 1

        print(f'True Counts: {true_counts}')
        print(f'False Counts: {false_counts}')
        accuracy = float(true_counts) / float(true_counts + false_counts)
        return accuracy


###########################################################################################################################
# preprocessing
###########################################################################################################################

X = pd.read_csv('input.txt', delimiter='\t', header=None)

labels = pd.read_csv('targets.txt', delimiter='\t', header=None)
labels['class'] = labels.idxmax(1)
cols = [col for col in labels.columns if col in ['class']]
labels = labels[cols]

df = X.join(labels)
df = df.sample(frac=1).reset_index(drop=True)

labels2 = df['class']
X2 = df.drop(['class'], axis=1)

X = np.array(X2)
labels = np.array(labels2)
labels = labels.ravel()

split_size = int(len(X) * 2 / 3.0)

x_train = X[:split_size]
y_train = labels[:split_size]
x_test = X[split_size:]
y_test = labels[split_size:]

###########################################################################################################################
# MLP with Backpropagation
###########################################################################################################################

print('###############################################################################')
print('MLP with Backpropagation method:')

model = MultiLayer_Perceptron_Classifier()

model.fit(x_train, y_train)
predictions = model.predict(x_test, y_test)

#print(y_test)
#print(predictions)

accuracy = model.score(y_test, predictions)
cm1 = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
df_cm1 = pd.DataFrame(cm1, index=[0, 1, 2], columns=[0, 1, 2])

print('Accuracy is: ', accuracy)
print('F1-score is: ', f1_score(y_test, predictions, average='macro'))
print('Recall is: ', recall_score(y_test, predictions, average='macro'))
print('Precision is: ', precision_score(y_test, predictions, average='macro'))
print('Confusion Matrix is: \n', cm1)

###########################################################################################################################
# Multiclass Perceptron
###########################################################################################################################

print('\n###############################################################################')
print('Multiclass Perceptron method:')

model = MultiClass_Perceptron_Classifier()

model.fit(x_train, y_train)
predictions = model.predict(x_test, y_test)

#print(y_test)
#print(predictions)

accuracy = model.score(y_test, predictions)
cm2 = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
df_cm2 = pd.DataFrame(cm2, index=[0, 1, 2], columns=[0, 1, 2])

print('Accuracy is: ', accuracy)
print('F1-score is: ', f1_score(y_test, predictions, average='macro'))
print('Recall is: ', recall_score(y_test, predictions, average='macro'))
print('Precision is: ', precision_score(y_test, predictions, average='macro'))
print('Confusion Matrix is: \n', cm2)

###########################################################################################################################
## Multiclass Adaline
###########################################################################################################################

print('\n###############################################################################')
print('Multiclass Adaline method:')

model = MultiClass_Adaline_Classifier()

model.fit(x_train, y_train)
predictions = model.predict(x_test, y_test)

#print(y_test)
#print(predictions)

accuracy = model.score(y_test, predictions)
cm3 = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
df_cm3 = pd.DataFrame(cm3, index=[0, 1, 2], columns=[0, 1, 2])

print('Accuracy is: ', accuracy)
print('F1-score is: ', f1_score(y_test, predictions, average='macro'))
print('Recall is: ', recall_score(y_test, predictions, average='macro'))
print('Precision is: ', precision_score(y_test, predictions, average='macro'))
print('Confusion Matrix is: \n', cm3)

#############################################################################################################################
# Scikit-Learn
#############################################################################################################################

print('\n###############################################################################')
print('Scikit-Learn method:')

clf = MLPClassifier()

clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

accuracy = model.score(y_test, predictions)
cm4 = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
df_cm4 = pd.DataFrame(cm4, index=[0, 1, 2], columns=[0, 1, 2])

print('Accuracy is: ', accuracy)
print('F1-score is: ', f1_score(y_test, predictions, average='macro'))
print('Recall is: ', recall_score(y_test, predictions, average='macro'))
print('Precision is: ', precision_score(y_test, predictions, average='macro'))
print('Confusion Matrix is: \n', cm4)

#############################################################################################################################
# Plotting all the Confusion Matrices
#############################################################################################################################

fig, ax = plt.subplots(2, 2)
sn.heatmap(df_cm1, ax=ax.flat[0], annot=True)
ax.flat[0].set_title('MLP with Backpropagation')
sn.heatmap(df_cm2, ax=ax.flat[1], annot=True)
ax.flat[1].set_title('Multiclass Perceptron')
sn.heatmap(df_cm3, ax=ax.flat[2], annot=True)
ax.flat[2].set_title('Multiclass Adaline')
sn.heatmap(df_cm4, ax=ax.flat[3], annot=True)
ax.flat[3].set_title('Scikit-Learn Method')
plt.show()
