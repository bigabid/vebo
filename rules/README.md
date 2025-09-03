# Vebo Rules

This directory contains data quality rules and configurations for the Vebo profiling system.

## Structure

- `data-quality/` - Rules for data quality assessment
- `data-types/` - Rules for data type validation
- `cross-column/` - Rules for cross-column relationship analysis
- `table-level/` - Rules for table-level analysis
- `vebo_rules.mdc` - Main rules configuration file

## Rule Categories

### Data Quality Rules
- Null value detection
- Duplicate identification
- Outlier detection
- Data completeness checks

### Data Type Rules
- Type validation
- Format consistency
- Range validation
- Pattern matching

### Cross-Column Rules
- Correlation analysis
- Dependency detection
- Constraint validation
- Relationship mapping

### Table-Level Rules
- Row count validation
- Schema consistency
- Performance metrics
- Statistical summaries

## Adding New Rules

1. Create rule files in the appropriate category directory
2. Follow the existing rule format and structure
3. Update the main rules configuration
4. Test with sample data
