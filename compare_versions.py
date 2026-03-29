import pandas as pd
import os

def generate_comparison_report():
    # Define the logs we want to compare
    logs = {
        "v2 (8-sample Overfit)": "training_log_v2.csv",
        "v3 (404-sample Baseline)": "training_log_v3.csv",
        "v4 (High-Res + Dice Loss)": "training_log_4.csv"
    }
    
    report_data = []
    
    for version, path in logs.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Get the metrics from the best validation epoch
            best_epoch = df.loc[df['val_loss'].idxmin()]
            
            report_data.append({
                "Version": version,
                "Best Val Loss": f"{best_epoch['val_loss']:.4f}",
                "Final LR": f"{best_epoch['lr']:.6e}",
                "Epochs Run": len(df)
            })
    
    if not report_data:
        print("No log files found. Ensure you've run the training scripts!")
        return

    comparison_df = pd.DataFrame(report_data)
    print("\n" + "="*50)
    print("      PHASE 2: ARCHITECTURE COMPARISON")
    print("="*50)
    print(comparison_df.to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    generate_comparison_report()