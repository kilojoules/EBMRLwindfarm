import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

# -----------------------------------------------------------
# 1. Load model and scalers
# -----------------------------------------------------------
model_path   = "Teodor_surrogates/models/ann_dlc12_out_wrot_Bl1Rad0FlpMnt_rank1.keras"
scaler_in_p  = "Teodor_surrogates/scalers/scaler_input_DLC12_wrot_Bl1Rad0FlpMnt.pkl"
scaler_out_p = "Teodor_surrogates/scalers/scaler_output_DLC12_wrot_Bl1Rad0FlpMnt.pkl"
model        = tf.keras.models.load_model(model_path)
scaler_in    = joblib.load(scaler_in_p)
scaler_out   = joblib.load(scaler_out_p)
print("Loaded model and scalers successfully.")

# -----------------------------------------------------------
# 2. Input variables for the surrogate (10 inputs)
# -----------------------------------------------------------
INPUT_VARS = [
    'saws_left', 'saws_right', 'saws_up', 'saws_down',
    'sati_left', 'sati_right', 'sati_up', 'sati_down',
    'pset', 'yaw'
]

# -----------------------------------------------------------
# 3. Ranges for heatmaps (update if needed)
# -----------------------------------------------------------
ranges = {
    'saws_left':  np.linspace(-1, 1, 20),
    'saws_right': np.linspace(-1, 1, 20),
    'saws_up':    np.linspace(-1, 1, 20),
    'saws_down':  np.linspace(-1, 1, 20),
    'sati_left':  np.linspace(-1, 1, 20),
    'sati_right': np.linspace(-1, 1, 20),
    'sati_up':    np.linspace(-1, 1, 20),
    'sati_down':  np.linspace(-1, 1, 20),
    'pset':       np.linspace(0, 1, 20),
    'yaw':        np.linspace(-30, 30, 20),
}

# -----------------------------------------------------------
# 4. Baseline values for variables not varied
# -----------------------------------------------------------
baseline = {
    'saws_left': 0.0,
    'saws_right': 0.0,
    'saws_up': 0.0,
    'saws_down': 0.0,
    'sati_left': 0.0,
    'sati_right': 0.0,
    'sati_up': 0.0,
    'sati_down': 0.0,
    'pset': 0.5,
    'yaw': 0.0,
}

# -----------------------------------------------------------
# 5. Prediction wrapper
# -----------------------------------------------------------
def predict_surrogate(inputs):
    # inputs: (N, 10)
    x = scaler_in.transform(inputs)
    y_scaled = model.predict(x, verbose=0)
    y = scaler_out.inverse_transform(y_scaled)
    return y.ravel()

# -----------------------------------------------------------
# 6. Heatmap function for ANY two of the 10 inputs
# -----------------------------------------------------------
def make_heatmap(var_i, var_j):
    """
    var_i, var_j: variable names (strings) from INPUT_VARS
    """
    # Get index positions
    i = INPUT_VARS.index(var_i)
    j = INPUT_VARS.index(var_j)
    
    # Ranges for these two variables
    xi = ranges[var_i]
    xj = ranges[var_j]
    
    # Make meshgrid
    XI, XJ = np.meshgrid(xi, xj)
    
    # Build full input matrix for surrogate
    full_inputs = np.zeros((XI.size, len(INPUT_VARS)))
    for idx, name in enumerate(INPUT_VARS):
        if name == var_i:
            full_inputs[:, idx] = XI.ravel()
        elif name == var_j:
            full_inputs[:, idx] = XJ.ravel()
        else:
            full_inputs[:, idx] = baseline[name]
    
    # Predict model output
    Y = predict_surrogate(full_inputs)
    Z = Y.reshape(XI.shape)
    
    # Plot heatmap
    plt.figure(figsize=(7, 6))
    cp = plt.contourf(XI, XJ, Z, levels=20)
    plt.colorbar(cp)
    plt.xlabel(var_i)
    plt.ylabel(var_j)
    plt.title(f"Heatmap: {var_i} vs {var_j}")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------
# 7. Yaw offset sweep function
# -----------------------------------------------------------
def plot_yaw_sweep(yaw_range=(-45, 45), n_points=91, 
                   saws_val=10.0, sati_val=0.08, pset_val=0.5):
    """
    Sweep yaw offset from yaw_range[0] to yaw_range[1].
    Keep all other inputs at reasonable values.
    
    Parameters:
    -----------
    yaw_range : tuple
        (min_yaw, max_yaw) in degrees
    n_points : int
        Number of points in sweep
    saws_val : float
        Sector-averaged wind speed (reasonable value ~10 m/s)
    sati_val : float
        Sector-averaged turbulence intensity (reasonable value ~0.08)
    pset_val : float
        Power setpoint, normalized [0, 1]
    """
    yaw_offsets = np.linspace(yaw_range[0], yaw_range[1], n_points)
    
    # Build input matrix: all yaw values, others fixed
    full_inputs = np.zeros((len(yaw_offsets), len(INPUT_VARS)))
    
    for idx, name in enumerate(INPUT_VARS):
        if name == 'yaw':
            full_inputs[:, idx] = yaw_offsets
        elif name.startswith('saws'):
            # All sector wind speeds same value
            full_inputs[:, idx] = saws_val
        elif name.startswith('sati'):
            # All sector turbulence intensities same value
            full_inputs[:, idx] = sati_val
        elif name == 'pset':
            full_inputs[:, idx] = pset_val
    
    # Predict DEL
    del_predictions = predict_surrogate(full_inputs)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(yaw_offsets, del_predictions, 'b-', linewidth=2, marker='o', 
            markersize=4, label='DEL prediction')
    
    # Highlight zero yaw
    idx_zero = np.argmin(np.abs(yaw_offsets - 0))
    ax.plot(yaw_offsets[idx_zero], del_predictions[idx_zero], 'r*', 
            markersize=15, label='Zero yaw offset', zorder=5)
    
    ax.set_xlabel('Yaw Offset (degrees)', fontsize=12)
    ax.set_ylabel('Damage Equivalent Load (DEL)', fontsize=12)
    ax.set_title(f'DEL vs Yaw Offset\n(SAWS={saws_val} m/s, SATI={sati_val}, Pset={pset_val})', 
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\nYaw Sweep Analysis:")
    print(f"  Yaw range: [{yaw_range[0]}, {yaw_range[1]}] degrees")
    print(f"  Fixed inputs: SAWS={saws_val} m/s, SATI={sati_val}, Pset={pset_val}")
    print(f"  DEL range: [{del_predictions.min():.2f}, {del_predictions.max():.2f}]")
    print(f"  DEL at 0° yaw: {del_predictions[idx_zero]:.2f}")
    print(f"  DEL increase at ±45°: {(del_predictions[0] - del_predictions[idx_zero]):.2f}")

# -----------------------------------------------------------
# 8. Example usage
# -----------------------------------------------------------
# Example heatmap
make_heatmap("saws_right", "yaw")

# Yaw sweep with default reasonable values
plot_yaw_sweep(yaw_range=(-45, 45), n_points=91, 
               saws_val=10.0, sati_val=0.05, pset_val=1.0)