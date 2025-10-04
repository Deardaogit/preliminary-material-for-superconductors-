"""
AI Tc Predictor for Superconductor Candidates
Refined Neural Network using PyTorch. 10 features, 200 epochs training.
Includes SHAP for interpretability.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

class TcPredictor(nn.Module):
    def __init__(self, input_size=10):
        super(TcPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Load synthetic data (assume CSV with 10 features + Tc target; 50 samples)
# In real: generate with np.random or load from SuperCon
data = pd.read_csv('../data/synthetic_supercon.csv')  # Columns: feat1-10, Tc
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Tc targets

# Normalize and split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train), batch_size=8, shuffle=True
)

# Model setup
model = TcPredictor(input_size=10)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# Training loop (200 epochs)
losses = []
for epoch in range(200):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    if epoch % 50 == 0:
        print(f'Epoch {epoch+1}/200, Loss: {avg_loss:.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    test_pred = model(X_test)
    test_mse = criterion(test_pred, y_test).item()
print(f'Test MSE: {test_mse:.4f}')

# Plot loss curve
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.savefig('../data/loss_curve.png')
plt.close()

# Predictions for candidates (10 features: atomic num, electroneg, doping ratio, lattice, bandgap, etc.)
# Virtual inputs (scale them!)
candidate_features = {
    'Grokene': np.array([28, 2.55, 0.05, 2.46, 0.12, 7.5, 11.3, 0.71, 0.0, 0.01]),   # Ni, C props, etc.
    'xHydride': np.array([57, 2.20, 0.17, 4.2, 0.08, 5.4, 6.2, 0.85, 0.1, 0.05]),     # La, H, S
    'AIronix': np.array([26, 2.02, 0.33, 3.1, 0.15, 9.8, 7.9, 0.46, 2.1, 0.02])       # Fe, Bi, Si
}

scaled_candidates = scaler.transform(list(candidate_features.values()))
with torch.no_grad():
    for name, feats in candidate_features.items():
        scaled_feats = torch.tensor(scaler.transform([feats]), dtype=torch.float32)
        pred_tc = model(scaled_feats).item()
        sigma = np.sqrt(test_mse * len(candidate_features))  # Uncertainty estimate
        print(f"{name}: Predicted Tc = {pred_tc:.1f} ± {sigma:.1f} K")

# SHAP Analysis (for interpretability)
explainer = shap.DeepExplainer(model, X_test[:10])  # Subset for speed
shap_values = explainer.shap_values(X_test[:10])
shap.summary_plot(shap_values[0], X_test[:10], feature_names=data.columns[:-1], show=False)
plt.savefig('../data/shap_summary.png')
plt.close()

print("AI predictions complete! Check data/ for plots and results.")
