import torch
import numpy as np
from model import AeroSurrogateNet

def predict_single_car(params_26):
    model = AeroSurrogateNet(in_features=26, hidden_dim=128, num_blocks=2)
    model.load_state_dict(torch.load("surrogate_cd_weights.pt", map_location="cpu"))
    model.eval()
    
    with torch.no_grad():
        x = torch.tensor(params_26, dtype=torch.float32).unsqueeze(0)
        cd_pred = model(x).item()
    return cd_pred

if __name__ == "__main__":
    # Sample 26 geometric features (normalized)
    sample_car = np.random.uniform(-0.5, 0.5, size=26)
    cd = predict_single_car(sample_car)
    print(f"[+] Predicted Aerodynamic Drag Coefficient (Cd): {cd:.4f}")