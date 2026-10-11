import pandas as pd
import os
from textblob import TextBlob
from collections import Counter

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _HAS_VADER = True
    _VADER_ANALYZER = SentimentIntensityAnalyzer()
except ImportError:
    _HAS_VADER = False
    print("Warning: vaderSentiment not installed. VADER sentiment metrics will be skipped.")

try:
    from Dbias.bias_classification import classifier
    from Dbias.bias_recognition import recognizer
    _HAS_DBIAS = True
    print("Note: dbias library loaded. Bias detection and recognition metrics will be calculated.")
except ImportError:
    _HAS_DBIAS = False
    print("Warning: Dbias not installed. Bias metrics will be skipped.")

try:
    from convokit import PolitenessStrategies
    _HAS_CONVOKIT = True
except ImportError:
    _HAS_CONVOKIT = False
    print("Warning: convokit not installed. Politeness/formality metrics will be skipped.")

def calculate_additional_metrics(text):
    """Calculate sentiment, lexical diversity, politeness, VADER, and dbias metrics for text."""
    if not text or pd.isna(text):
        return {
            "sentiment_polarity": 0,
            "sentiment_subjectivity": 0,
            "vader_compound": 0,
            "vader_positive": 0,
            "vader_negative": 0,
            "vader_neutral": 0,
            "lexical_diversity_ttr": 0,
            "politeness_score": 0,
            "dbias_classification": "unknown",
            "dbias_bias_count": 0
        }
    
    # Sentiment analysis using TextBlob
    blob = TextBlob(str(text))
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    
    # VADER sentiment analysis
    vader_compound = 0
    vader_positive = 0
    vader_negative = 0
    vader_neutral = 0
    if _HAS_VADER:
        vader_scores = _VADER_ANALYZER.polarity_scores(str(text))
        vader_compound = vader_scores['compound']
        vader_positive = vader_scores['pos']
        vader_negative = vader_scores['neg']
        vader_neutral = vader_scores['neu']
    
    # Lexical diversity (Type-Token Ratio)
    words = str(text).lower().split()
    length = len(words)
    unique_words = len(set(words))
    lexical_diversity_ttr = unique_words / length if length > 0 else 0
    
    # Politeness analysis using heuristic approach
    # Count politeness markers in the text
    politeness_markers = [
        "please", "could you", "would you", "might you", "may i",
        "thank", "thanks", "appreciate", "kindly", "respectfully",
        "excuse me", "pardon", "sorry", "apologize", "regret"
    ]
    
    text_lower = str(text).lower()
    politeness_count = sum(1 for marker in politeness_markers if marker in text_lower)
    
    # Normalize by text length (politeness score per 100 words)
    word_count = len(text_lower.split())
    politeness_score = (politeness_count / word_count * 100) if word_count > 0 else 0
    
    # Dbias metrics
    dbias_classification = "unknown"
    dbias_bias_count = 0
    if _HAS_DBIAS:
        try:
            # Bias classification (biased vs non-biased)
            dbias_classification = classifier(str(text))
            
            # Bias recognition (count of biased words/phrases)
            biased_entities = recognizer(str(text))
            dbias_bias_count = len(biased_entities) if biased_entities else 0
        except Exception as e:
            print(f"Warning: Dbias analysis failed: {e}")
    
    return {
        "sentiment_polarity": sentiment_polarity,
        "sentiment_subjectivity": sentiment_subjectivity,
        "vader_compound": vader_compound,
        "vader_positive": vader_positive,
        "vader_negative": vader_negative,
        "vader_neutral": vader_neutral,
        "lexical_diversity_ttr": lexical_diversity_ttr,
        "politeness_score": politeness_score,
        "dbias_classification": dbias_classification,
        "dbias_bias_count": dbias_bias_count
    }

def add_metrics_to_csv(csv_file_path):
    """Add additional metrics to existing CSV file incrementally."""
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file_path}'")
        return
    
    output_file = csv_file_path.replace(".csv", "_with_metrics.csv")
    
    # Check if output file already exists and load it to resume
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            existing_count = len(existing_df)
            print(f"Found existing output file with {existing_count} rows. Resuming from row {existing_count}...")
            
            # Only process new rows
            if existing_count >= len(df):
                print("All rows already processed. Nothing to do.")
                return
            
            df = df.iloc[existing_count:].reset_index(drop=True)
            start_idx = existing_count
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}. Starting fresh.")
            start_idx = 0
    else:
        start_idx = 0
    
    print(f"Processing {len(df)} new rows (starting from row {start_idx})...")
    
    # Calculate additional metrics for each row
    metrics_list = []
    for idx, row in df.iterrows():
        text = row.get("Generated_Questions", "")
        metrics = calculate_additional_metrics(text)
        metrics_list.append(metrics)
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} rows...")
    
    # Create DataFrame with new metrics
    metrics_df = pd.DataFrame(metrics_list)
    
    # Combine with original data for the new rows
    for col in metrics_df.columns:
        df[col] = metrics_df[col]
    
    # If resuming, append to existing file; otherwise create new
    if start_idx > 0:
        existing_df = pd.read_csv(output_file)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(output_file, index=False)
    else:
        df.to_csv(output_file, index=False)
    
    print(f"\nDone! Updated file saved as: {output_file}")
    print(f"Total rows in file: {start_idx + len(df)}")
    print(f"New columns added: sentiment_polarity, sentiment_subjectivity, vader_compound, vader_positive, vader_negative, vader_neutral, lexical_diversity_ttr, politeness_score, dbias_classification, dbias_bias_count")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "demographic_bias_results.csv"
    
    add_metrics_to_csv(csv_file)
