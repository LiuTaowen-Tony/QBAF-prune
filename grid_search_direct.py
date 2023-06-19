# Import necessary libraries
import numpy as np
from sklearn.model_selection import ParameterGrid
# ... other imports ...

import torch
from torch import nn
from torch.nn.utils import prune
import time
import cv2
import csv
import copy

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets.iris import load_iris
from datasets.adult import load_adult
from datasets.mushrooms import load_mushroom
from sklearn.model_selection import train_test_split
from visualise import visualize_neural_network

# import dataloader
from torch.utils.data import DataLoader, TensorDataset


class BaseModel(nn.Module):
    def conn_sum(self):
        pass
    def get_connections(self):
        pass
    def prune(self):
        pass
    def terminate_prune(self):
        pass

class baseline(BaseModel):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size,
                 target_conn1,
                 target_conn2,
                 target_conn_skip) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.linear_skip = nn.Linear(input_size, output_size, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.target_conn1 = target_conn1
        self.target_conn2 = target_conn2
        self.target_conn_skip = target_conn_skip

    def forward(self, x):
        tmp = self.linear_skip(x)
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = x + tmp
        return self.softmax(x)
    
    def conn_sum(self):
        conn0 = self.linear1.weight.data
        conn1 = self.linear2.weight.data
        conn_skip = self.linear_skip.weight.data
        conn0_sum = conn0.abs().sum().item()
        conn1_sum = conn1.abs().sum().item()
        conn_skip_sum = conn_skip.abs().sum().item()
        return conn0_sum + conn1_sum + conn_skip_sum

    def get_connections(self):
        conn0 = self.linear1.weight.data
        conn1 = self.linear2.weight.data
        conn_skip = self.linear_skip.weight.data
        conn0 = conn0.nonzero().tolist()
        conn1 = conn1.nonzero().tolist()
        conn_skip = conn_skip.nonzero().tolist()
        return conn0, conn1, conn_skip

    def prune(self):
        conn0, conn1, conn_skip = self.get_connections()
        if len(conn0) > self.target_conn1:
            prune.l1_unstructured(self.linear1, name='weight', amount=0.3)
        if len(conn1) > self.target_conn2:
            prune.l1_unstructured(self.linear2, name='weight', amount=0.3)
        if len(conn_skip) > self.target_conn_skip:
            prune.l1_unstructured(self.linear_skip, name='weight', amount=0.3)

    def terminate_prune(self):
        conn0, conn1, conn_skip = self.get_connections()
        return len(conn0) <= self.target_conn1 and len(conn1) <= self.target_conn2 and len(conn_skip) <= self.target_conn_skip


def train(model: BaseModel, X, y, X_test, y_test, epochs, lr, decay):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    previous_conn_sum = 0
    previous_loss = 0
    for epoch in range(epochs):
        for x_b, y_b in loader:
            optimizer.zero_grad()
            y_pred = model(x_b)
            loss = criterion(y_pred, y_b)
            loss.backward()
            optimizer.step()
            conn_sum = model.conn_sum()

        if epoch % 10 == 0:
            test(model, X_test, y_test)
            test(model, X, y, "Train")
            print(f'Epoch {epoch} loss: {loss.item()}')
            print(f'Number of connections: {conn_sum}')
        
        if abs(conn_sum - previous_conn_sum) < 0.002:
            return

        if abs(loss.item() - previous_loss) < 0.0002:
            return
        previous_conn_sum = conn_sum
        previous_loss = loss.item()



# def accuracy(y_pred, y_true):
#     classes = torch.argmax(y_pred, dim=1)
#     if len(y_true.shape) > 1:
#         labels = torch.argmax(y_true, dim=1)
#     else:
#         labels = y_true
#     _accuracy = torch.mean((classes == labels).float())
#     return _accuracy

def test(model, x, y, name="Test"):
    y_pred = model(x)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(y_pred, y)
    classes = torch.argmax(y_pred, dim=1)
    if len(y.shape) > 1:
        labels = torch.argmax(y, dim=1)
    else:
        labels = y
    test_acc = accuracy_score(classes, labels)
    test_pre = precision_score(classes, labels, average='macro')
    test_rec = recall_score(classes, labels, average='macro')
    test_f1 = f1_score(classes, labels, average='macro')

    print(f'{name} accuracy: {test_acc.item()}')
    print(f'{name} precision: {test_pre.item()}')
    print(f'{name} recall: {test_rec.item()}')
    print(f'{name} f1: {test_f1.item()}')

    print(f'Loss: {loss.item()}')
    return (test_acc.item(), test_pre.item(), test_rec.item(), test_f1.item())


def prune_model(model, params, X_train, y_train, X_val, y_val, visualise, 
                dataset_name, model_name, is_fuzzy):
    # Unpack parameters
    lr, decay = params['lr'], params['decay']
    for i in range(100):
        train(model, X_train, y_train, X_val, y_val, 100, lr, decay)
        test(model, X_val, y_val)
        connections = model.get_connections()
        print(connections)
        if visualise:
            visualize_neural_network(connections)
        if model.terminate_prune():
            break
        model.prune()


# def prune_model(model: BaseModel, X_train, y_train, X_test, y_test, visualise, nth_run,
#                 dataset_name, model_name, is_fuzzy, lr, decay):
#     for i in range(100):
#         train(model, X_train, y_train, X_test, y_test, 100, lr,decay)
#         test(model, X_test, y_test)
#         connections = model.get_connections()
#         print(connections)
#         if visualise:
#             visualize_neural_network(connections)
#         if model.terminate_prune():
#             break
#         model.prune()



# Define the hyperparameters for grid search
param_grid = {'lr': [0.01, 0.003, 0.001], 'decay': [0.01, 0.001, 0.0001]}

# Your main function
def main(dataset_name, model, X, y, *, model_name, visualise, nth_run, 
         is_fuzzy):

    is_visualise = visualise == "visualise"

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32 if is_iris else torch.long)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125)

    best_score = -np.inf
    best_params = None

    # Perform grid search on hyperparameters
    for params in ParameterGrid(param_grid):
        model_copy = copy.deepcopy(model)  # Copy the model to avoid in-place changes
        prune_model(model_copy, params, X_train, y_train, X_val, y_val, 
                    visualise=is_visualise, dataset_name=dataset_name,
                    model_name=model_name, is_fuzzy=is_fuzzy)
        # Evaluate the model
        if dataset_name == 'iris':
            a, *_ = test(model_copy, X_train, y_train)
        else:
            a, *_ = test(model_copy, X_val, y_val)
        if a > best_score:
            best_score = a
            best_params = params
            # best_model = model_copy

    # Evaluate the best model on the test set
    # test(best_model, X_test, y_test, "Test")
    
    for i in range(10):
        model_copy = copy.deepcopy(model)
        prune_model(model_copy, best_params, X_train, y_train, X_val, y_val,
                    visualise=is_visualise,  dataset_name=dataset_name,
                    model_name=model_name, is_fuzzy=is_fuzzy)
        file_name = f'{dataset_name}_{model_name}_{is_fuzzy}_{best_params}.csv'
        import os
        if file_name not in os.listdir():
            with open(file_name, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['connections', 'test_acc', 'precision', 'recall', 'f1'])
        with open(file_name, 'a') as f:
            test_acc, precision, recall, f1 = test(model_copy, X_test, y_test)
            writer = csv.writer(f)
            writer.writerow(
                [
                    str(model_copy.get_connections()),
                    round(test_acc, 4),
                    round(precision, 4),
                    round(recall, 4),
                    round(f1, 4),
                ]
            )

    print(f'Best parameters are {best_params} with score {best_score}')


for is_fuzzy in [True, False]:
    for dataset_name in ['iris', 'adult', 'mushroom']:
        is_iris = False
        target_conn1 = 4
        target_conn2 = 3
        target_conn_skip = 1
        if dataset_name == 'iris':
            X, y, *_= load_iris(is_fuzzy)
            is_iris = True
            target_conn1 = 5
        elif dataset_name == 'adult':
            X, y, *_= load_adult(is_fuzzy)
        elif dataset_name == 'mushroom':
            X, y, *_= load_mushroom(is_fuzzy)
        model = baseline(X.shape[1], 10, 3 if is_iris else 2,  target_conn1, target_conn2, target_conn_skip)
        main(dataset_name, model, X, y, 
                model_name = "direct", visualise = "nv", 
                nth_run = 0, is_fuzzy = is_fuzzy, )
    






# Define your prune_model function

