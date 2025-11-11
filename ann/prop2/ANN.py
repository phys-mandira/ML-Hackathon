import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.EState import Fingerprinter
from rdkit.Chem import Descriptors
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim


# Feature Engineering
def fps_plus_mw(mol):
    """Combine EState fingerprint and molecular weight."""
    fp, _ = Fingerprinter.FingerprintMol(mol)
    return np.append(fp, Descriptors.MolWt(mol))


def load_data(csv_path):
    smiles, y = [], []
    for row in pd.read_csv(csv_path, header=None).values:
        mol = Chem.MolFromSmiles(row[0])  # change the representation of molecule from smile string to matrix
        if mol is None:
            continue
        smiles.append(fps_plus_mw(mol))
        y.append(float(row[2]))     # prop2
    X = np.vstack(smiles)
    y = np.array(y)
    return X, y


# PyTorch Model
class ANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


def pytorch_train_cv(X, y, test_X, test_y, n_splits=5, lr=1e-3, epochs=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    test_X = scaler.transform(test_X)

    fold_results = []
    best_state = None
    best_val_loss = np.inf
    f = open("epoch_errors_pytorch.csv", "w")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n===== Fold {fold} =====")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = ANN(X.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_t)
            loss = criterion(output, y_train_t)
            loss.backward()
            optimizer.step()
        
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()
                train_loss = criterion(model(X_train_t), y_train_t).item()
            #print(f"Fold {fold} | Train MSE: {train_loss/2:.6f} | Val MSE: {val_loss/2:.6f}")
        
            if fold == n_splits-1:
                f.write(str(epoch)+" "+str(train_loss)+" "+str(val_loss))
                f.write("\n")

        fold_results.append((train_loss/2, val_loss/2))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
    
    f.close()
    #pd.DataFrame(fold_results, columns=["train_mse", "val_mse"]).to_csv("cv_errors_pytorch.csv", index=False)
    #print("\n Saved PyTorch CV errors → cv_errors_pytorch.csv")

    # Final evaluation with best model
    final_model = ANN(X.shape[1]).to(device)
    final_model.load_state_dict(best_state)
    final_model.eval()

    with torch.no_grad():
        y_train_pred = final_model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy().ravel()
        y_test_pred = final_model(torch.tensor(test_X, dtype=torch.float32, device=device)).cpu().numpy().ravel()

    pd.DataFrame({"actual": y, "predicted": y_train_pred}).to_csv("train_predictions_pytorch.csv", index=False, sep=' ')
    pd.DataFrame({"actual": test_y, "predicted": y_test_pred}).to_csv("test_predictions_pytorch.csv", index=False, sep=' ')
    print(" Saved PyTorch train/test predictions")

    return final_model, scaler


# ===============================================================
# MAIN
# ===============================================================
if __name__ == "__main__":
    X, y = load_data("dataset_train_train.csv")
    X_test, y_test = load_data("dataset_train_test.csv")

    print("\n=== Running PyTorch Model ===")
    pytorch_train_cv(X, y, X_test, y_test)

