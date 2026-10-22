import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from model import AeroSurrogateNet

def generate_aerodynamic_data(n_samples=2500):
    """
    Simulates DrivAerNet 26 geometric parameter morphing space
    and high-fidelity CFD aerodynamic drag ground-truth distribution.
    """
    np.random.seed(42)
    # 26 Geometric Parameters (e.g., diffuser angle, trunk taper, ride height, wheel arches)
    X = np.random.uniform(-1.0, 1.0, size=(n_samples, 26))
    
    # Non-linear physics mapping based on real car aerodynamics
    # Base Cd ~0.26 to 0.38
    cd_base = 0.28
    nonlinear_drag = (
        0.045 * np.sin(X[:, 0] * 1.5) +          # Rear boat-tail taper
        0.035 * (X[:, 1] ** 2) +                 # Ground clearance & underbody
        0.025 * np.maximum(0, X[:, 2]) +         # Side mirror vortex detachment
        0.015 * np.cos(X[:, 3]) +                # Windshield rake angle
        0.010 * np.sum(X[:, 4:10], axis=1) * 0.1 # Surface micro-features
    )
    noise = np.random.normal(0, 0.002, size=n_samples) # CFD mesh discretization noise
    y = (cd_base + nonlinear_drag + noise).reshape(-1, 1)
    return X, y

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device}")

    X, y = generate_aerodynamic_data()
    split = int(0.85 * len(X))
    
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X[:split])
    X_val = scaler_x.transform(X[split:])
    y_train, y_val = y[:split], y[split:]

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = AeroSurrogateNet(in_features=26, hidden_dim=128, num_blocks=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("[*] Training Residual Surrogate Neural Network on 26 Geometric Design Features...")
    model.train()
    for epoch in range(1, 26):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:02d}/25 | Batch MSE Loss: {total_loss / len(X_train):.6f}")

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        val_preds = model(val_tensor).cpu().numpy()
        r2 = r2_score(y_val, val_preds)
        mae = mean_absolute_error(y_val, val_preds)

    print("\n" + "="*50)
    print(f"[+] Surrogate Model Validation R^2 Score : {r2:.4f}")
    print(f"[+] Mean Absolute Error (MAE) on Cd    : {mae:.6f} counts")
    print("="*50)

    torch.save(model.state_dict(), "surrogate_cd_weights.pt")
    np.save("scaler_mean.npy", scaler_x.mean_)
    np.save("scaler_scale.npy", scaler_x.scale_)
    print("[+] Model weights saved -> 'surrogate_cd_weights.pt'")

if __name__ == "__main__":
    main()
