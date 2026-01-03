
import matplotlib.pyplot as plt
import numpy as np
import os

# Data from previous analysis (Layer 40)
# N = 30272 total, N(4_inc) = 10495, N(5_inc) = 10607

# Marginals: P(Inc)
turns = ['Turn 4', 'Turn 5', 'Turn 6']
marginals = [0.3467, 0.3504, 0.3655]
n_total = 30272

# Conditionals: P(Next Inc | Current Inc)
transitions = ['T4 Inc → T5 Inc', 'T5 Inc → T6 Inc']
conditionals = [0.7500, 0.7816]
n_conds = [10495, 10607] # N(4_inc), N(5_inc)

def calc_moe(p, n):
    """Calculates 95% Margin of Error for a proportion."""
    if n == 0: return 0
    se = np.sqrt((p * (1 - p)) / n)
    return 1.96 * se

def plot_probs():
    # Calculate intervals
    marg_errs = [calc_moe(p, n_total) for p in marginals]
    cond_errs = [calc_moe(p, n) for p, n in zip(conditionals, n_conds)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Marginal Error Rates
    colors_marg = ['#ff9999', '#ff6666', '#ff3333']
    bars1 = ax1.bar(turns, marginals, yerr=marg_errs, capsize=5, 
                    color=colors_marg, edgecolor='black', alpha=0.8, error_kw=dict(lw=1.5, capthick=1.5))
    ax1.set_ylim(0, 1.0)
    ax1.set_title('Marginal Error Rate\n(Overall Probability of Error)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add values on top
    for bar, err in zip(bars1, marg_errs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + err + 0.02,
                 f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Conditional "Stickiness"
    colors_cond = ['#ffcc99', '#ff9933']
    bars2 = ax2.bar(transitions, conditionals, yerr=cond_errs, capsize=5,
                    color=colors_cond, edgecolor='black', alpha=0.8, error_kw=dict(lw=1.5, capthick=1.5))
    ax2.set_ylim(0, 1.0)
    ax2.set_title('Error Persistence\n(Probability of STAYING Incorrect)', fontsize=12, fontweight='bold')
    # ax2.set_ylabel('Conditional Probability')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add values on top
    for bar, err in zip(bars2, cond_errs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + err + 0.02,
                 f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Layer 40 Error Dynamics: Prevalence vs. Persistence\n(with 95% Confidence Intervals)', fontsize=14, y=1.05)
    plt.tight_layout()
    
    output_path = "graphs/layer_40_error_dynamics_with_ci.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_probs()
