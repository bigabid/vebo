# Rule Categories Analysis

Based on the provided rule ideas, here's the organized structure for the Vebo Python code generation system.

## Rule Categories

### 1. Column Attributes (Meta-Rules for Rule Selection)

#### 1.1 Type Category
- **numeric**: int, float, double, long
- **textual**: varchar, text, string
- **collection**: array, list, set
- **temporal**: date, datetime, timestamp
- **boolean**: true/false, 0/1, yes/no
- **categorical**: enum, category

#### 1.2 Type Specificity
- **int**: 8-bit, 16-bit, 32-bit, 64-bit
- **float**: single precision, double precision
- **varchar**: with length constraints
- **text**: unlimited length

#### 1.3 Diversity Levels
- **constant**: All values identical
- **binary**: Only 2 unique values
- **low**: 3-10 unique values
- **medium**: 11-100 unique values
- **high**: 101-1000 unique values
- **distinctive**: >1000 unique values

#### 1.4 Nullability Levels
- **empty**: No null values (0%)
- **low**: 1-5% null values
- **medium**: 6-25% null values
- **high**: 26-75% null values
- **full**: >75% null values

### 2. Column-Level Checks

#### 2.1 Basic Statistics
- Number of unique values
- Most common value and frequency
- Value distribution (histogram)
- Length statistics (for textual data)

#### 2.2 Pattern Matching
- Regex pattern matching
- Format validation (email, phone, etc.)
- Data type consistency

#### 2.3 Quality Checks
- Outlier detection (for numeric data)
- Duplicate detection
- Format consistency

### 3. Cross-Column Checks

#### 3.1 Relationship Analysis
- **Identicality**: Columns with identical values
- **Correlation**: Statistical correlation between numeric columns
- **Mutual nullability**: Columns that are null together
- **Mutual indicativity**: If X in column_1 → Y in column_2

#### 3.2 Dependency Analysis
- Functional dependencies
- Referential integrity
- Business rule validation

## Rule Selection Logic

### Meta-Rules for Rule Selection

1. **Data Type Detection**
   ```python
   def detect_column_type_category(series):
       if pd.api.types.is_numeric_dtype(series):
           return "numeric"
       elif pd.api.types.is_string_dtype(series):
           return "textual"
       elif pd.api.types.is_datetime64_any_dtype(series):
           return "temporal"
       elif pd.api.types.is_bool_dtype(series):
           return "boolean"
       else:
           return "unknown"
   ```

2. **Diversity Level Detection**
   ```python
   def detect_diversity_level(series):
       unique_count = series.nunique()
       total_count = len(series)
       unique_ratio = unique_count / total_count
       
       if unique_count == 1:
           return "constant"
       elif unique_count == 2:
           return "binary"
       elif unique_ratio < 0.01:
           return "low"
       elif unique_ratio < 0.1:
           return "medium"
       elif unique_ratio < 0.5:
           return "high"
       else:
           return "distinctive"
   ```

3. **Nullability Level Detection**
   ```python
   def detect_nullability_level(series):
       null_ratio = series.isnull().sum() / len(series)
       
       if null_ratio == 0:
           return "empty"
       elif null_ratio <= 0.05:
           return "low"
       elif null_ratio <= 0.25:
           return "medium"
       elif null_ratio <= 0.75:
           return "high"
       else:
           return "full"
   ```

## Implementation Priority

### Phase 1: Basic Column Analysis
1. Type category detection
2. Basic statistics (unique count, most common value)
3. Nullability analysis
4. Diversity level detection

### Phase 2: Advanced Column Checks
1. Value distribution analysis
2. Pattern matching (regex)
3. Length statistics
4. Outlier detection

### Phase 3: Cross-Column Analysis
1. Identicality detection
2. Correlation analysis
3. Mutual nullability
4. Mutual indicativity

## Rule Dependencies

### Dependency Graph
```
Basic Data Analysis
├── Type Detection
│   ├── Numeric Rules
│   ├── Textual Rules
│   └── Temporal Rules
├── Diversity Analysis
│   ├── Constant Value Rules
│   ├── Binary Value Rules
│   └── High Diversity Rules
└── Nullability Analysis
    ├── Empty Column Rules
    ├── Low Null Rules
    └── High Null Rules

Cross-Column Analysis (if enabled)
├── Correlation Analysis
├── Identicality Detection
└── Dependency Analysis
```

## Next Steps

1. Implement basic column attribute detection
2. Create rule templates for each category
3. Implement the meta-rule selection logic
4. Start with Phase 1 implementation
