import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo 
from model import Model, CustomDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 

y = pd.DataFrame(y.Diagnosis.map(dict(M=1,B=0)))
min_x = X.min()
max_x = X.max()
# X = (X-min_x )/ (max_x-min_x) # do normalization itself in the model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

def train_fn(loader, model, optimizer, loss_fn, device="cpu"):
    loop = tqdm(loader)

    average_loss = 0
    count = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        pred = model.forward(data)
        loss = loss_fn(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
        average_loss += loss.item()
        count += 1
    
    average_loss = average_loss / count
    return average_loss

dataset = CustomDataset(X_train.values, y_train.values)
loader = DataLoader(
    dataset,
    batch_size = 1,
    shuffle = True
)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([y_train.value_counts()[0] / y_train.value_counts()[1]]).to(device=DEVICE))
min_x = torch.Tensor(min_x).to(DEVICE)
max_x = torch.Tensor(max_x).to(DEVICE)
model = Model(dataset.features, min_x, max_x).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())

NUM_EPOCHS = 50
# 30 should be enough already

for i in range(NUM_EPOCHS):
    ave_loss = train_fn(loader, model, optimizer, criterion, device=DEVICE)
    print(f'Epoch {i+1}: {ave_loss}')

test_loader = DataLoader(
    CustomDataset(X_test.values, y_test.values),
    batch_size = 1,
    shuffle = True
)

y_pred = []
y_true = []

model.eval()

for batch_idx, (data, targets) in enumerate(test_loader):
    data = data.to(DEVICE)
    targets = targets.to(DEVICE)

    with torch.no_grad():
        pred = model(data)
    
    pred = torch.sigmoid(pred)
    pred = (pred >= 0.5).int()
    y_pred.append(pred.cpu().numpy()[0][0])
    y_true.append(targets.int().cpu().numpy()[0][0])

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f'True negatives: {tn}')
print(f'False positives: {fp}')
print(f'False negatives: {fn}')
print(f'True positives: {tp}')
print(f'Balanced accuracy score: {balanced_accuracy_score(y_true, y_pred)}')

torch.save(model, 'model.pth')