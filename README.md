# Interpretable Prediction and Large-Scale Analysis of Judging in Professional Boxing

**MIT Sloan Sports Analytics Conference Research Paper Competition Submission**

## Paper Summary

This research presents the first large-scale automated analysis of professional boxing performance, leveraging AI-generated statistics from DeepStrike to understand what drives judges' scoring decisions. Using data from 1,003 professional bouts (7,323 rounds), we developed interpretable points-based and neural network models that predict round winners with accuracy comparable to professional judges.

**Key Findings:**
- Punch impact differentiation is critical: maximum impact punches are valued over 10x more than minimum impact punches by judges
- Aggression power (throwing high-commitment punches) emerged as the second most important metric, even when punches don't land cleanly
- Simple interpretable models with just 5-7 parameters can match the predictive accuracy of more complex neural networks
- The points-based model achieved 76% pairwise agreement with judges, ranking within the range of professional judges while remaining transparent, reproducible, and bias-free

---

This repo serves to build and analyze a database of thousands of boxing matches with accompanying round-by-round scorecards. Through our analysis we determine the relative importance of different aspects of a fighter's performance. We also evaluate top judges and analyze differences in their style that emerge from the data. Lastly, we create an automated judging system that is interpretable and able to accurately predict judging outcomes.

This code was developed as part of a research collaboration between Jabbr and INS Quebec.

The stats used in the investigation are entirely generated using [Jabbr's DeepStrike platform](https://jabbr.ai/blog/deepstrike-stats-explained).



## Running the Analysis

Run the analysis script using Python 3:

```bash
python3 analysis.py [flags]
```

**Note:** The set of parameters used in gradient descent and the MLP can be modified inside `analysis.py` by editing the `parameters` array in the `main()` function. Several predefined parameter sets are available as commented options.

### Command Line Flags

**Data Source Flags:**
| Flag | Description |
|------|-------------|
| `-quadcam` | Use quad-cam stats from JSON files instead of single-cam Excel files |
| `-singlecam` | Use single-cam stats (useful for validation comparisons) |
| `-ignorescores` | Skip scorecard processing (for comparing quadcam vs singlecam stats without judge data) |

**Model Selection Flags:**
| Flag | Description |
|------|-------------|
| `-mlp` | Use a multi-layer perceptron neural network instead of gradient descent |
| `-best` | Skip gradient descent and use pre-saved coefficient values (parameters must match the `bestValues` dict in main) |

**Mapping Parameter Flags:**
| Flag | Description |
|------|-------------|
| `-dampener <value>` or `-d <value>` | Set the dampening parameter D for the ratio-based scoring equation (default: 150.0) |
| `-sharpness <value>` or `-s <value>` | Set the sharpness parameter S for the scoring equation (default: 9.0) |
| `-optimizemapping` | Include D and S parameters in gradient descent optimization |

**Data Splitting Flags:**
| Flag | Description |
|------|-------------|
| `-split <test_ratio> [validation_ratio]` | Split data into training/testing sets. Provide one decimal (0-1) for test ratio, optionally add a second for validation ratio |
| `-seed <integer>` | Set random seed for reproducible data splits (default: 42) |

**Judge Analysis Flags:**
| Flag | Description |
|------|-------------|
| `-j "<Judge Name>"` | Filter data to only rounds scored by the specified judge |
| `-dt <decimal 0-1>` | Disagreement threshold - ignore rounds with absolute predicted value below this threshold when ranking judges |
| `-costrank` | Rank judges by average cost of their scores instead of accuracy percentage |
| `-sampleRank` | Randomly sample judges' rounds so all judges have equal sample sizes in rankings |
| `-shrink <value or 'opt'>` | Apply Empirical Bayes shrinkage to judge rankings. Use a positive number for k, or 'opt' to calculate optimal k via Method of Moments |

**Parameter Exploration Flags:**
| Flag | Description |
|------|-------------|
| `-combo <int>` or `-combos <int>` | Run gradient descent with every combination of `<int>` parameters from the parameters array, print best combinations (1-16) |
| `-combostart <int>` | Like `-combo` but always includes `startingParams`, finding the best `<int>` parameters to add (1-16) |

**Other Flags:**
| Flag | Description |
|------|-------------|
| `-lookup` | After analysis, prompt user to search for specific fights by fighter names and date |
| `-includeinserted` | Include fights whose stat sheets are marked as INCLUDED |

### Example Commands

```bash
# basic gradient descent analysis
python3 analysis.py

# use pre-saved best coefficients with 80/20 train/test split
python3 analysis.py -best -split 0.2

# run MLP with 60/20/20 train/validation/test split
python3 analysis.py -mlp -split 0.2 0.2

# analyze only rounds scored by a specific judge
python3 analysis.py -j "Judge_001"

# find best 5-parameter combinations
python3 analysis.py -combo 5

# use quad-cam data with custom mapping parameters
python3 analysis.py -quadcam -d 50.0 -s 8.0

# rank judges with Empirical Bayes shrinkage using optimal k
python3 analysis.py -shrink opt -split 0.2

# run gradient descent with mapping parameter optimization starting at D=100 and S=8; use a 80/20 train/test split, and ranks judges by cost with optimal Empirical Bayes shrinkage applied.
python3 analysis.py -costrank -split 0.2 -d 100.0 -s 8.0 -optimizemapping -shrink opt
```
