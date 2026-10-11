import pandas as pd
import sys
import scipy.stats as stats
from textstat import flesch_reading_ease, flesch_kincaid_grade, coleman_liau_index, gunning_fog, smog_index, difficult_words, lexicon_count
from textblob import TextBlob
from collections import Counter

# Optional politeness detection (if available)
try:
    from convokit import PolitenessStrategies
    _HAS_CONVOKIT = True
except ImportError:
    _HAS_CONVOKIT = False
    print("Warning: convokit not installed. Politeness metrics will be skipped.")

def compute_metrics(text):
    """Compute all metrics for a single text."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None
    
    # --- Readability scores (textstat) ---
    flesch_reading_ease_val = flesch_reading_ease(text)
    flesch_kincaid_grade_val = flesch_kincaid_grade(text)
    gunning_fog_val = gunning_fog(text)
    smog_val = smog_index(text)
    coleman_liau_val = coleman_liau_index(text)
    difficult_words_val = difficult_words(text)
    word_count_val = lexicon_count(text, removepunct=True)
    
    # --- Lexical diversity (manual TTR calculation) ---
    words = text.lower().split()
    unique_words = len(set(words))
    length = len(words)
    lexical_diversity_ttr_val = unique_words / length if length > 0 else 0
    
    # --- Sentiment (TextBlob) ---
    blob = TextBlob(text)
    sentiment_polarity_val = blob.sentiment.polarity  # -1 to +1
    
    # --- Politeness (convokit) ---
    if _HAS_CONVOKIT:
        try:
            ps = PolitenessStrategies()
            politeness_score_val = ps.get_politeness_score(text)
        except:
            politeness_score_val = 0
    else:
        politeness_score_val = 0

    # Combine into a dictionary
    result = {
        'word_count': word_count_val,
        'flesch_reading_ease': flesch_reading_ease_val,
        'flesch_kincaid_grade': flesch_kincaid_grade_val,
        'coleman_liau': coleman_liau_val,
        'gunning_fog': gunning_fog_val,
        'smog': smog_val,
        'difficult_words': difficult_words_val,
        'sentiment_polarity': sentiment_polarity_val,
        'lexical_diversity_ttr': lexical_diversity_ttr_val,
        'politeness_score': politeness_score_val
    }
    return result

def analyze_llm_results(csv_file_path):
    """
    Analyze LLM bias results from CSV file.
    Computes metrics on the fly and displays mean ± standard deviation for each metric by demographic group.
    Compares each minority group to Control using t-tests.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file_path}'")
        return
    
    # Check if metrics already exist in CSV
    metrics_exist = all(m in df.columns for m in ['word_count', 'flesch_reading_ease', 'flesch_kincaid_grade', 'coleman_liau', 'gunning_fog', 'smog', 'difficult_words', 'sentiment_polarity', 'lexical_diversity_ttr', 'politeness_score'])
    
    if not metrics_exist:
        print("Computing metrics on the fly...")
        df_metrics = df['Generated_Questions'].apply(compute_metrics).apply(pd.Series)
        df = pd.concat([df, df_metrics], axis=1)
    
    # Filter out empty rows
    df = df[df["word_count"] > 0]
    
    # Define metrics to analyze
    metrics = [
        "word_count",
        "flesch_reading_ease", 
        "flesch_kincaid_grade",
        "coleman_liau",
        "gunning_fog",
        "smog",
        "difficult_words",
        "sentiment_polarity",
        "lexical_diversity_ttr",
        "politeness_score"
    ]
    
    # Filter to only include metrics that exist in the CSV
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print("Error: No recognized metrics found in CSV file")
        return
    
    print("=" * 60)
    print(f"LLM BIAS ANALYSIS: {csv_file_path}")
    print("=" * 60)
    
    # Display descriptive statistics (mean ± std) by demographic
    print("\n--- DESCRIPTIVE STATISTICS (Mean ± Standard Deviation) ---\n")
    
    for metric in available_metrics:
        print(f"{metric.replace('_', ' ').title()}:")
        summary = df.groupby("Demographic")[metric].agg(['mean', 'std', 'count'])
        
        for demo in summary.index:
            mean_val = summary.loc[demo, 'mean']
            std_val = summary.loc[demo, 'std']
            count_val = summary.loc[demo, 'count']
            print(f"  {demo}: {mean_val:.4f} ± {std_val:.4f} (n={int(count_val)})")
        print()
    
    # Overall summary across all demographics
    print("=" * 60)
    print("OVERALL SUMMARY (All Demographics Combined)")
    print("=" * 60)
    print("\nMetric | Mean | Std | Min | Max")
    print("-" * 50)
    
    for metric in available_metrics:
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        min_val = df[metric].min()
        max_val = df[metric].max()
        print(f"{metric.replace('_', ' ').title():30} | {mean_val:7.4f} | {std_val:6.4f} | {min_val:6.4f} | {max_val:6.4f}")
    
    print("\n" + "=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Demographic groups: {df['Demographic'].nunique()}")
    print("=" * 60)
    
    # Perform t-tests comparing each minority group to Control
    print("\n" + "=" * 60)
    print("T-TEST RESULTS (Each Minority Group vs Control)")
    print("=" * 60 + "\n")
    
    if 'Control' not in df['Demographic'].values:
        print("Error: Control group not found in data")
        return
    
    control_data = df[df['Demographic'] == 'Control']
    
    for group in df['Demographic'].unique():
        if group != 'Control':
            print(f"--- {group} vs Control ---")
            print("-" * 50)
            
            group_data = df[df['Demographic'] == group]
            
            for metric in available_metrics:
                try:
                    t_stat, p_val = stats.ttest_ind(control_data[metric], group_data[metric], nan_policy='omit')
                    
                    print(f"Metric: {metric.replace('_', ' ').title()}")
                    print(f"T-Statistic: {t_stat:.4f}")
                    print(f"P-Value: {p_val:.6f}")
                    
                    if p_val < 0.05:
                        print("Result: 🚨 STATISTICALLY SIGNIFICANT (p < 0.05)")
                    else:
                        print("Result: ✅ NOT STATISTICALLY SIGNIFICANT (p >= 0.05)")
                    print()
                except Exception as e:
                    print(f"Error performing t-test for {metric}: {e}")
                    print()
            
            print("=" * 50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = input("Enter CSV file path: ")
    
    analyze_llm_results(csv_file)