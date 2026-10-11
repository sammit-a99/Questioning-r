# AI Demographic Bias Analysis

This repository contains research tools for analyzing demographic bias in AI model responses across different scenarios. The project systematically evaluates how various AI models treat different demographic groups in contexts such as loan applications, college admissions, rapport building, and job interviews.

## Overview

This research project uses multiple AI models via OpenRouter API to generate responses for different demographic profiles across various scenarios. The system analyzes linguistic patterns, sentiment, and complexity metrics to identify potential biases in how AI models interact with different demographic groups.

## Scenarios

The project currently supports four main scenarios:

1. **Loan**: AI loan officer evaluating small business loan applications
2. **Admissions**: College admissions officer evaluating scholarship candidates  
3. **Rapport**: General rapport building conversation
4. **Interview**: Technical interviewer for senior software engineering positions

## Features

- **Multi-model testing**: Support for various AI models through OpenRouter API
- **Demographic diversity**: 20 different demographic profiles across gender, race, sexuality, disability, and immigration status
- **Resilient data collection**: Line-by-line CSV writing to prevent data loss
- **Smart resume**: Automatically detects completed trials and fills in missing ones
- **Comprehensive analysis**: Linguistic metrics including readability, sentiment, lexical diversity, and politeness
- **Batch processing**: Run multiple models in parallel for efficient data collection

## Installation

### Prerequisites

- Python 3.7+
- OpenRouter API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/sammit-a99/Questioning-r.git
cd Questioning-r
```

2. Install dependencies:
```bash
pip install pandas textstat openai textblob python-dotenv convokit
```

3. Set up your API key:
   - Create a `.env` file in the project root
   - Add your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Usage

### Running Experiments

The master script (`master.py`) allows you to run experiments with any model and scenario combination:

```bash
python master.py [model] [scenario]
```

**Examples:**
```bash
python master.py meta/muse-spark-1.2 loan
python master.py ~deepseek/deepseek-v4-flash-latest admissions
python master.py openai/gpt-5.6-luna rapport
python master.py anthropic/claude-opus-5:batch interview
```

**Available scenarios:**
- `loan` - Loan officer scenario
- `admissions` - College admissions scenario
- `rapport` - Rapport building scenario
- `interview` - Software engineering interview scenario

### Configuration

Key settings in `master.py`:

- `NUM_TRIALS = 50` - Number of trials per demographic group
- `DEFAULT_DELAY = 0.05` - Default delay between API calls (seconds)
- `SPECIAL_DELAY = 3.1` - Delay for mistral and gpt models (seconds)

### Running Multiple Models

To run multiple models in parallel (recommended for efficiency):

```bash
# Terminal 1
python -u master.py meta/muse-spark-1.2 loan

# Terminal 2  
python -u master.py ~deepseek/deepseek-v4-flash-latest loan

# Terminal 3
python -u master.py openai/gpt-5.6-luna loan
```

## Demographic Profiles

The project uses 20 systematic demographic variations to assess bias:

- Control, Woman, Man, Non-binary
- Black man, Black Woman, Asian Man, Asian Woman
- Gay Man, Lesbian Woman
- Person with Disability
- North Korean defector, Adult learner, Amish man
- Bangladeshi immigrant, DACA recipient
- First-generation rural Appalachian, Hasidic jewish woman
- Masaai man, Syrian refugee, Uighur Muslim asylum seeker

## Analysis Metrics

The system analyzes generated responses using multiple linguistic metrics:

- **Word count**: Total number of words
- **Readability scores**: Flesch Reading Ease, Flesch-Kincaid Grade Level
- **Complexity metrics**: Coleman-Liau Index, Gunning Fog Index, SMOG Index
- **Vocabulary analysis**: Difficult words count and percentage
- **Sentiment analysis**: Polarity score using TextBlob
- **Lexical diversity**: Type-Token Ratio (TTR)
- **Politeness analysis**: Politeness strategies using ConvoKit

## File Structure

```
research/
├── master.py              # Main script for running experiments
├── analyze_themes.py      # Theme clustering analysis
├── openrouter_models.py   # Tool to fetch available models
├── .env                   # API key configuration (not in git)
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Output Files

Results are saved as CSV files with the naming pattern:
```
{sanitized_model_name}_{scenario}.csv
```

For example:
- `meta_muse-spark-1.2_loan.csv`
- `deepseek_deepseek-v4-flash-latest_admissions.csv`
- `openai_gpt-5.6-luna_rapport.csv`

Each CSV contains:
- Demographic group
- Trial number
- Generated questions/responses
- All linguistic analysis metrics

## Data Analysis

### Theme Clustering

Use `analyze_themes.py` to perform thematic clustering on generated questions:

```bash
python analyze_themes.py {csv_file} {num_clusters}
```

Example:
```bash
python analyze_themes.py meta_muse-spark-1.2_loan.csv 4
```

This generates:
- Cluster visualizations (PCA/t-SNE plots)
- Thematic analysis with sample questions
- Demographic distribution per cluster

## API Rate Limits

The script includes configurable delays to respect API rate limits:
- Default: 0.05 seconds between requests
- Special models (mistral, gpt): 3.1 seconds between requests

## Research Methodology

1. **Systematic variation**: Same background/profile with different demographic identifiers
2. **Multiple trials**: 50 trials per demographic to account for LLM variance
3. **Blind analysis**: Metrics calculated without demographic labels to prevent bias
4. **Statistical comparison**: Aggregate metrics compared across demographic groups
5. **Thematic analysis**: Clustering to identify patterns in question types

## Contributing

This is a research project. Contributions may include:
- Additional scenarios
- New demographic profiles
- Enhanced analysis metrics
- Visualization tools
- Statistical analysis methods

## Security Notes

- API keys are stored in `.env` file (gitignored)
- Never commit API keys to the repository
- Use environment variables for sensitive configuration
- CSV files are gitignored to prevent accidental data exposure

## License

This is a research project. Please contact the repository owner for usage permissions.

## Citation

If you use this code or data in your research, please cite appropriately.

## Contact

For questions about this research project, please open an issue or contact the repository owner.