"""
Utility functions for feature extraction and visualization.
"""

import matplotlib.pyplot as plt
import json

def plot_results(results_file='../data/results.json'):
    """
    Plot bar chart of predicted Tc from results.
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    materials = list(results.keys())
    tcs = [results[m].get('ai_tc', 250) for m in materials]  # Assume AI Tc in results
    
    plt.figure(figsize=(8, 5))
    plt.bar(materials, tcs, color=['blue', 'green', 'red'])
    plt.title('Predicted Critical Temperature (Tc) for Superconductor Candidates')
    plt.ylabel('Tc (K)')
    plt.xlabel('Material')
    plt.ylim(200, 350)
    plt.savefig('../data/tc_bar.png', dpi=300)
    plt.close()
    print("Plot saved to data/tc_bar.png")

# Usage: plot_results()
