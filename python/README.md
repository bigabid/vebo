# Vebo Python Profiler

This directory contains the Python implementation of the Vebo data profiling system.

## Installation

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Usage

```python
from vebo_profiler import VeboProfiler
from vebo_profiler.core.profiler import ProfilingConfig

# Configure profiler
config = ProfilingConfig(
    enable_cross_column=True,
    deepness_level="standard",
    max_workers=4
)

# Create profiler instance
profiler = VeboProfiler(config)

# Profile a DataFrame
result = profiler.profile_dataframe(df, filename="my_dataset.csv")

# Save results
profiler.save_result(result, "results.json")
```

## Examples

See the `examples/python/` directory for complete usage examples.

## API Documentation

The main classes and functions are:

- `VeboProfiler`: Main profiler class
- `ProfilingConfig`: Configuration options
- `ProfilingResult`: Results container
- `RuleEngine`: Rule management system
- `CheckExecutor`: Check execution engine
