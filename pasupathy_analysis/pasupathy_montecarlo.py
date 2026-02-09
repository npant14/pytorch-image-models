import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# V4 Data
V4_Bar_Data_Pasupathy = [
    0.014147909967845543, 0.0, 0.029581993569131756, 0.08874598070739544,
    0.14019292604501604, 0.5260450160771705, 0.1659163987138263,
    0.014147909967845543, 0.0398713826366559, 0.0, 0.0
]

N_V4 = 80
v4_counts = np.round(np.array(V4_Bar_Data_Pasupathy) * N_V4).astype(int)
print("V4 Counts:", v4_counts)
print("V4 Sum:", v4_counts.sum())

# File Paths
files = {
    "AlexNet (features.0)": "/users/xyu110/pytorch-image-models/pasupathy_results_1111/neuron_analysis_ALEXNET-AUG/neuron_data/features_0.csv",
    "ResNet18 (layer2)": "/users/xyu110/pytorch-image-models/pasupathy_results_1111/neuron_analysis_RESNET18-AUG/neuron_data/layer2.csv",
    "HMAX V4 (S2)": "/users/xyu110/pytorch-image-models/pasupathy_results_1111/neuron_analysis_HMAX_V3_ADJ/neuron_data/model_backbone_s2.csv",
    "HMAX V4 (C2)": "/users/xyu110/pytorch-image-models/pasupathy_results_1111/neuron_analysis_HMAX_V3_ADJ/neuron_data/model_backbone_c2.csv"
}

# Bins
# Width = 0.4 per bin. Center bin [-0.2, 0.2].
# 11 bins total.
# Edges: -2.2, -1.8, -1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8, 2.2
bins = np.array([-2.2, -1.8, -1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8, 2.2])
print("\nBins:", bins)

def monte_carlo_independence_test(counts1, counts2, num_simulations=10000):
    """
    Performs a Monte Carlo test of independence between two samples.
    H0: Both samples are drawn from the same underlying multinomial distribution.
    """
    n1 = counts1.sum()
    n2 = counts2.sum()
    
    # Observed statistic
    # We use the chi2 statistic as our distance metric
    # We add a small epsilon to expected frequencies to avoid division by zero in the statistic calculation
    obs_table = np.array([counts1, counts2])
    
    # Function to calculate Chi2 statistic safely
    def get_chi2_stat(table):
        # Calculate expected based on row/col sums
        row_sums = table.sum(axis=1)
        col_sums = table.sum(axis=0)
        total = table.sum()
        
        # Outer product to get expected counts
        expected = np.outer(row_sums, col_sums) / total
        
        # Avoid division by zero
        # If expected is 0, it means col_sum is 0 (no observations in that bin for either group)
        # In that case, (O-E)^2 is 0, so contribution is 0.
        # We can just mask those out or add epsilon.
        # Adding epsilon is safer for vectorized ops.
        expected_safe = expected.copy()
        expected_safe[expected_safe == 0] = 1e-9
        
        return np.sum((table - expected)**2 / expected_safe)

    actual_stat = get_chi2_stat(obs_table)
    
    # Pooled distribution (best estimate of H0 distribution)
    pooled_counts = counts1 + counts2
    pooled_probs = pooled_counts / pooled_counts.sum()
    
    # Simulation
    sim_stats = []
    for _ in range(num_simulations):
        # Generate synthetic samples from the pooled distribution
        sim1 = np.random.multinomial(n1, pooled_probs)
        sim2 = np.random.multinomial(n2, pooled_probs)
        
        sim_table = np.array([sim1, sim2])
        sim_stats.append(get_chi2_stat(sim_table))
        
    sim_stats = np.array(sim_stats)
    
    # P-value: Proportion of simulations where synthetic difference >= observed difference
    p_value = (sim_stats >= actual_stat).mean()
    
    return actual_stat, p_value

results = []

for name, filepath in files.items():
    print(f"\nProcessing {name}...")
    try:
        df = pd.read_csv(filepath)
        if 'scale_invariance_score' not in df.columns:
             print(f"Error: 'scale_invariance_score' not found in {filepath}")
             continue
            
        scores = df['scale_invariance_score'].dropna().values
        
        # Bin the scores
        model_counts, _ = np.histogram(scores, bins=bins)
        
        print(f"Model Counts: {model_counts}")
        print(f"Model Sum: {model_counts.sum()}")
        
        # --- 1. Standard Chi-Square Test ---
        obs = np.array([v4_counts, model_counts])
        # Filter out columns where both counts are zero to avoid expected frequency of zero
        valid_cols = np.sum(obs, axis=0) > 0
        obs_filtered = obs[:, valid_cols]
        
        try:
            if obs_filtered.shape[1] < 2:
                chi2_std, p_std = np.nan, np.nan
                print("Standard Chi2: Not enough valid bins")
            else:
                chi2_std, p_std, dof, expected = chi2_contingency(obs_filtered)
                print(f"Standard Chi2: {chi2_std:.4f}, p-value: {p_std:.4e}")
        except Exception as e:
            chi2_std, p_std = np.nan, np.nan
            print(f"Standard Chi2 Error: {e}")

        # --- 2. Monte Carlo Test ---
        chi2_mc, p_mc = monte_carlo_independence_test(v4_counts, model_counts, num_simulations=20000)
        print(f"Monte Carlo Chi2: {chi2_mc:.4f}, p-value: {p_mc:.4e}")
        
        results.append({
            "Model": name,
            "Std_Chi2": chi2_std,
            "Std_p-value": p_std,
            "MC_Chi2": chi2_mc,
            "MC_p-value": p_mc
        })
        
    except Exception as e:
        print(f"Error processing {name}: {e}")
        import traceback
        traceback.print_exc()

print("\nSummary Results:")
results_df = pd.DataFrame(results)
# Reorder columns for readability
cols = ["Model", "Std_Chi2", "Std_p-value", "MC_Chi2", "MC_p-value"]
print(results_df[cols])
