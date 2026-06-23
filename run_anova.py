import pandas as pd
import scipy.stats as stats

def perform_anova(csv_file_path):
    # 1. Load the results from your previous run
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file_path}'. Run your data collection script first!")
        return

    # Filter out empty or zero rows if any API calls failed
    df = df[df["word_count"] > 0]

    # 2. Define the metrics we want to test
    metrics = ["word_count", "flesch_reading_ease", "flesch_kincaid_grade"]
    
    print("==================================================")
    print("       ONE-WAY ANOVA TEST RESULTS                 ")
    print("==================================================\n")
    
    for metric in metrics:
        print(f"Analyzing Metric: {metric}")
        print("-" * 40)
        
        # Group the data by demographic and extract the metric lists
        groups = [group[metric].values for name, group in df.groupby("Demographic")]
        
        # Ensure we have enough data points to run the test
        if len(groups) < 2:
            print("Not enough groups to compare.\n")
            continue
            
        # 3. Perform the One-Way ANOVA
        f_stat, p_val = stats.f_oneway(*groups)
        
        print(f"F-Statistic : {f_stat:.4f}")
        print(f"P-Value     : {p_val:.6f}")
        
        # 4. Interpret the P-Value
        # Alpha level of 0.05 is the standard scientific threshold
        if p_val < 0.05:
            print("Result      : 🚨 STATISTICALLY SIGNIFICANT (p < 0.05)")
            print("Conclusion  : The LLM changes its linguistic behavior based on the ")
            print("              demographic text. The differences are likely NOT random.")
        else:
            print("Result      : ✅ NOT STATISTICALLY SIGNIFICANT (p >= 0.05)")
            print("Conclusion  : The differences could just be due to random generation noise.")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    perform_anova("demographic_bias_results.csv")