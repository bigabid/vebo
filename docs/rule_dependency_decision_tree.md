# Rule Dependency Decision Tree

This document contains the decision tree for rule dependencies in the Vebo Python code generation system. This tree will be updated throughout the project as new rules and dependencies are discovered.

## Decision Tree Flow

```mermaid
graph TD
    A[Start: Dataset Input] --> B[Basic Data Analysis]
    B --> C{Data Size > Threshold?}
    C -->|Yes| D[Enable Sampling]
    C -->|No| E[Full Dataset Analysis]
    
    D --> F[Random Sample with Seed=42]
    E --> G[Column Type Detection]
    F --> G
    
    G --> H{Contains Numeric Columns?}
    H -->|Yes| I[Enable Numeric Rules]
    H -->|No| J[Skip Numeric Rules]
    
    G --> K{Contains String Columns?}
    K -->|Yes| L[Enable String Rules]
    K -->|No| M[Skip String Rules]
    
    G --> N{Contains DateTime Columns?}
    N -->|Yes| O[Enable DateTime Rules]
    N -->|No| P[Skip DateTime Rules]
    
    I --> Q{Cross-Column Checks Enabled?}
    J --> Q
    L --> Q
    M --> Q
    O --> Q
    P --> Q
    
    Q -->|Yes| R[Enable Cross-Column Analysis]
    Q -->|No| S[Skip Cross-Column Analysis]
    
    R --> T[Execute Dependent Rules]
    S --> U[Execute Independent Rules]
    
    T --> V[Generate Correlations]
    U --> W[Generate Individual Reports]
    V --> W
    
    W --> X[Combine Results]
    X --> Y[Generate Visualizations]
    Y --> Z[Final Report]
    
    style A fill:#e1f5fe
    style Z fill:#c8e6c9
    style D fill:#fff3e0
    style R fill:#f3e5f5
```

## Decision Points Explained

### 1. Data Size Threshold
- **Threshold**: TBD (e.g., 100,000 rows)
- **Decision**: If dataset exceeds threshold, enable sampling
- **Action**: Use random sampling with constant seed for reproducibility

### 2. Column Type Detection
- **Numeric Columns**: Enable statistical analysis, distribution checks, outlier detection
- **String Columns**: Enable pattern matching, length analysis, encoding checks
- **DateTime Columns**: Enable temporal analysis, date range validation, format checks

### 3. Cross-Column Analysis
- **Enabled**: Run correlation analysis, dependency checks, relationship validation
- **Disabled**: Skip cross-column rules for faster execution

### 4. Rule Dependencies
- **Independent Rules**: Can run in parallel
- **Dependent Rules**: Must wait for prerequisite rules to complete

## Rule Categories

### Basic Level (Always Run)
- Data type detection
- Null value analysis
- Basic statistics (count, unique values)
- File size and structure validation

### Standard Level (Data-Dependent)
- Distribution analysis (for numeric columns)
- Pattern analysis (for string columns)
- Date format validation (for datetime columns)
- Data quality metrics

### Deep Level (Cross-Column)
- Correlation analysis
- Relationship validation
- Complex data quality rules
- Advanced statistical tests

## Implementation Notes

- All random operations use seed=42 for reproducibility
- Parallel execution at both rule-level and category-level
- Error handling: skip failed rules, continue with remaining rules
- Results combined into structured JSON output
- Visualizations generated for key insights

## Updates Log

- **2024-12-XX**: Initial decision tree created
- **Future updates**: Will be added as new rules and dependencies are discovered
