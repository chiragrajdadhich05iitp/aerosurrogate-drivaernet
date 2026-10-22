import torch
import numpy as np
import time
from model import AeroSurrogateNet

def run_inference():
    device = torch.device("cpu")
    model = AeroSurrogateNet(in_features=26, hidden_dim=128, num_blocks=2)
    model.load_state_dict(torch.load("surrogate_cd_weights.pt", map_location=device))
    model.eval()

    scaler_mean = np.load("scaler_mean.npy")
    scaler_scale = np.load("scaler_scale.npy")

    # Sample test vehicle (26 morphing box variables)
    np.random.seed(101)
    sample_design = np.random.uniform(-0.8, 0.8, size=(1, 26))
    
    # Normalize with saved distribution
    sample_scaled = (sample_design - scaler_mean) / scaler_scale
    x_input = torch.tensor(sample_scaled, dtype=torch.float32)

    start_time = time.perf_counter()
    with torch.no_grad():
        cd_pred = model(x_input).item()
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print("="*60)
    print("        AERODYNAMIC DRAG SURROGATE INFERENCE")
    print("="*60)
    print(f" Input Geometry Shape Vector : 26 Parametric Design Tokens")
    print(f" Predicted Drag Coefficient   : Cd = {cd_pred:.4f}")
    print(f" Surrogate Inference Latency : {elapsed_ms:.2f} ms")
    print(f" Traditional CFD Simulation  : ~4 to 6 CPU-Hours (Equiv)")
    print("="*60)

if __name__ == "__main__":
    run_inference()
