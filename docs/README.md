__/\\\________/\\\__/\\\\\\\\\\\__/\\\\\\\\\\\\\____/\\\\\\\\\\\\\\\_____/\\\\\\\\\\\___        
 _\/\\\_______\/\\\_\/////\\\///__\/\\\/////////\\\_\/\\\///////////____/\\\/////////\\\_       
  _\//\\\______/\\\______\/\\\_____\/\\\_______\/\\\_\/\\\______________\//\\\______\///__      
   __\//\\\____/\\\_______\/\\\_____\/\\\\\\\\\\\\\\__\/\\\\\\\\\\\_______\////\\\_________     
    ___\//\\\__/\\\________\/\\\_____\/\\\/////////\\\_\/\\\///////___________\////\\\______    
     ____\//\\\/\\\_________\/\\\_____\/\\\_______\/\\\_\/\\\_____________________\////\\\___   
      _____\//\\\\\__________\/\\\_____\/\\\_______\/\\\_\/\\\______________/\\\______\//\\\__  
       ______\//\\\________/\\\\\\\\\\\_\/\\\\\\\\\\\\\/__\/\\\\\\\\\\\\\\\_\///\\\\\\\\\\\/___ 
        _______\///________\///////////__\/////////////____\///////////////____\///////////_____


# Vebo Data Profiler - Python Code Generation System

A comprehensive data profiling system that automatically generates and executes Python code based on rules to analyze tabular datasets.

## Features

- **Automatic Rule Selection**: Meta-rules determine which checks are relevant based on data characteristics
- **Multi-level Analysis**: Column-level, cross-column, and table-level profiling
- **Configurable Deepness**: Basic, standard, and deep analysis modes
- **Parallel Execution**: Efficient processing with configurable parallelization
- **Comprehensive Output**: Structured JSON results with detailed metrics
- **Error Resilience**: Continues execution even when individual checks fail

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
import pandas as pd
from vebo_profiler import VeboProfiler, ProfilingConfig

# Create or load your dataset
df = pd.read_csv('your_data.csv')

# Configure the profiler
config = ProfilingConfig(
    enable_cross_column=True,
    deepness_level="standard",
    max_workers=4
)

# Profile the dataset
profiler = VeboProfiler(config)
result = profiler.profile_dataframe(df, filename="your_data.csv")

# Save results
profiler.save_result(result, "profiling_results.json")
```

### Run the Example

```bash
python example_usage.py
```

## Rule Categories

### Column Attributes (Meta-Rules)
- **Type Categories**: numeric, textual, temporal, boolean, categorical, collection
- **Diversity Levels**: constant, binary, low, medium, high, distinctive
- **Nullability Levels**: empty, low, medium, high, full

### Column-Level Checks
- **Basic Statistics**: unique count, most common value, null analysis
- **Numeric Analysis**: statistics, outlier detection, distribution analysis
- **Text Analysis**: length statistics, pattern matching, format validation
- **Temporal Analysis**: date validation, temporal patterns

### Cross-Column Checks
- **Relationship Analysis**: identicality, correlation, mutual nullability
- **Dependency Analysis**: mutual indicativity, functional dependencies

## Configuration Options

```python
config = ProfilingConfig(
    enable_cross_column=True,      # Enable cross-column analysis
    deepness_level="standard",     # basic, standard, deep
    max_workers=4,                 # Parallel execution workers
    timeout_seconds=300,           # Check execution timeout
    sample_size=10000,             # Sample size for large datasets
    sampling_threshold=100000,     # Threshold for enabling sampling
    random_seed=42                 # Seed for reproducible sampling
)
```

## Output Format

The profiler generates structured JSON output with:

- **Metadata**: Dataset info, execution details, configuration
- **Summary**: Overall score, quality grade, issue counts
- **Column Analysis**: Per-column statistics and check results
- **Cross-Column Analysis**: Relationship analysis results
- **Table-Level Analysis**: Dataset-wide metrics
- **Errors**: Detailed error information

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Meta Rules    │───▶│   Rule Engine    │───▶│ Check Executor  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Column Analysis │    │  Rule Selection  │    │  Parallel Exec  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Development

### Project Structure

```
src/vebo_profiler/
├── __init__.py
└── core/
    ├── __init__.py
    ├── meta_rules.py      # Meta-rule detection
    ├── rule_engine.py     # Rule management
    ├── check_executor.py  # Check execution
    └── profiler.py        # Main profiler class
```

### Adding New Rules

1. Define the rule in `rule_engine.py`
2. Add the code template with the check logic
3. Specify which column types, diversity levels, and nullability levels it applies to
4. The system will automatically select and execute relevant rules

### Testing

```bash
python example_usage.py
```

## License

This project is part of the Vebo data profiling platform.

## Contributing

Please refer to the main Vebo project documentation for contribution guidelines.