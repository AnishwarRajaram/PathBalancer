import pandas as pd
import matplotlib.pyplot as plt

def plot_training_results(csv_path="training_log_4.csv"):
    df = pd.read_csv(csv_path)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Loss on the left axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='tab:red', linestyle='--')
    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='tab:red', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, alpha=0.3)

    # Plot Learning Rate on a second axis (right side)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color='tab:blue')
    ax2.plot(df['epoch'], df['lr'], label='LR', color='tab:blue', alpha=0.6)
    ax2.set_yscale('log') # LR drops are easier to see on a log scale
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('U-Net Training Progress (Roughness-Aware)')
    fig.tight_layout()
    plt.show()
    #plt.savefig('progress.png')

if __name__ == "__main__":
    plot_training_results()