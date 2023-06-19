import torch
from torch import nn
from torch.nn.utils import prune

from datasets.mushrooms import load_mushroom
from sklearn.model_selection import train_test_split

# import dataloader
from torch.utils.data import DataLoader, TensorDataset


class iris_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return self.softmax(x)
    
    def num_conn(self):
        conn0 = self.linear1.weight.data


def train(model, X, y, x_test, y_test, epochs, lr, decay = 2e-5):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    previous_conn0_sum = 0
    previous_conn1_sum = 0

    for epoch in range(epochs):
        for x_b, y_b in loader:
            optimizer.zero_grad()
            y_pred = model(x_b)
            loss = criterion(y_pred, y_b)
            loss.backward()
            optimizer.step()
            conn0 = model.linear1.weight.data
            conn1 = model.linear2.weight.data
            conn0_sum = conn0.abs().sum().item()
            conn1_sum = conn1.abs().sum().item()
            # print(f'Epoch {epoch} loss: {loss.item()}')
            # train_acc = accuracy(y_pred, y)
            # print(f'Train accuracy: {train_acc.item()}')
            
            # if difference of sum of weights is less than 0.01, stop

        if epoch % 10 == 0:
            # print sum of weights
            conn0 = model.linear1.weight.data
            conn1 = model.linear2.weight.data
            conn0_sum = conn0.abs().sum().item()
            conn1_sum = conn1.abs().sum().item()

            test(model, x_test, y_test)
            test(model, X, y, "Train")
            print(f'conn0 sum: {conn0_sum}')
            print(f'conn1 sum: {conn1_sum}')

        if abs(conn0_sum - previous_conn0_sum) < 0.01 and abs(conn1_sum - previous_conn1_sum) < 0.01:
            return model
        previous_conn0_sum = conn0_sum
        previous_conn1_sum = conn1_sum

    return model

def accuracy(y_pred, y_true):
    """Gets the accuracy."""
    classes = torch.argmax(y_pred, dim=1)
    if len(y_true.shape) > 1:
        labels = torch.argmax(y_true, dim=1)
    else:
        labels = y_true
    _accuracy = torch.mean((classes == labels).float())
    return _accuracy

def test(model, x, y, name="Test"):
    y_pred = model(x)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(y_pred, y)
    test_acc = accuracy(y_pred, y)
    print(f'{name} accuracy: {test_acc.item()}')
    print(f'Loss: {loss.item()}')

def main():

    # X, y, inputs, label = load_iris(False)
    X, y, inputs, label = load_mushroom(False)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    model = iris_model(X.shape[1], 10, 2)


    
    for i in range(100):
        model = train(model, X_train, y_train, X_test, y_test, 1000, 0.001, 1e-4)
        test(model, X_test, y_test)
        print(model.linear1.weight.nonzero())
        print(model.linear2.weight.nonzero())

        flag1 = model.linear1.weight.nonzero().size()[0] > 5
        flag2 = model.linear2.weight.nonzero().size()[0] > 3

        if not flag1 and not flag2:
            torch.save(model.state_dict(), 'mushroom_model.pt')
            test(model, X_test, y_test)
            return

        if model.linear1.weight.nonzero().size()[0] > 5:
            prune.l1_unstructured(model.linear1, name='weight', amount=0.5)
        if model.linear2.weight.nonzero().size()[0] > 3:
            prune.l1_unstructured(model.linear2, name='weight', amount=0.5)



main()
