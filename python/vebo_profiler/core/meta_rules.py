"""
Meta-rules for determining which rules are relevant based on data characteristics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class TypeCategory(Enum):
    """Column type categories."""
    NUMERIC = "numeric"
    TEXTUAL = "textual"
    TEMPORAL = "temporal"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    COLLECTION = "collection"
    UNKNOWN = "unknown"


class DiversityLevel(Enum):
    """Diversity levels for column values."""
    CONSTANT = "constant"
    BINARY = "binary"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    DISTINCTIVE = "distinctive"
    FULLY_UNIQUE = "fully_unique"


class NullabilityLevel(Enum):
    """Nullability levels for column values."""
    EMPTY = "empty"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    FULL = "full"


@dataclass
class ColumnAttributes:
    """Attributes of a column that determine which rules are relevant."""
    name: str
    type_category: TypeCategory
    diversity_level: DiversityLevel
    nullability_level: NullabilityLevel
    unique_count: int
    total_count: int
    null_count: int
    most_common_value: Any = None
    most_common_frequency: int = 0
    is_likely_index: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Compatibility helper for tests expecting a dict form."""
        return {
            "name": self.name,
            "type_category": self.type_category.value,
            "diversity_level": self.diversity_level.value,
            "nullability_level": self.nullability_level.value,
            "unique_count": int(self.unique_count),
            "total_count": int(self.total_count),
            "null_count": int(self.null_count),
            "most_common_value": self.most_common_value,
            "most_common_frequency": int(self.most_common_frequency),
            "is_likely_index": bool(self.is_likely_index),
        }


class MetaRuleDetector:
    """
    Detects column attributes to determine which rules are relevant.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the meta-rule detector.
        
        Args:
            seed: Random seed for reproducible sampling
        """
        self.seed = seed
        np.random.seed(seed)
    
    def detect_column_type_category(self, series: pd.Series) -> TypeCategory:
        """
        Detect the type category of a column.
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            TypeCategory enum value
        """
        # Check booleans before numeric to avoid misclassification
        if pd.api.types.is_bool_dtype(series):
            return TypeCategory.BOOLEAN
        elif pd.api.types.is_numeric_dtype(series):
            return TypeCategory.NUMERIC
        elif pd.api.types.is_string_dtype(series):
            return TypeCategory.TEXTUAL
        elif pd.api.types.is_datetime64_any_dtype(series):
            return TypeCategory.TEMPORAL
        elif pd.api.types.is_categorical_dtype(series):
            return TypeCategory.CATEGORICAL
        else:
            # Handle object dtype that contains strings
            if series.dtype == object:
                non_null = series.dropna()
                if len(non_null) == 0:
                    return TypeCategory.UNKNOWN
                sample = list(non_null.head(min(50, len(non_null))))
                if all(isinstance(x, str) for x in sample):
                    return TypeCategory.TEXTUAL
            return TypeCategory.UNKNOWN
    
    def detect_diversity_level(self, series: pd.Series) -> DiversityLevel:
        """
        Detect the diversity level of column values.
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            DiversityLevel enum value
        """
        unique_count = series.nunique(dropna=True)
        total_count = len(series.dropna())
        unique_ratio = unique_count / total_count if total_count > 0 else 0
        
        if unique_count == 1:
            return DiversityLevel.CONSTANT
        
        # Check for fully unique columns first (every non-null value is unique)
        if unique_ratio == 1.0 and total_count > 1:
            return DiversityLevel.FULLY_UNIQUE
            
        if unique_count == 2:
            # Treat boolean-like or 0/1 as BINARY, otherwise LOW
            non_null_vals = set(series.dropna().unique())
            if non_null_vals.issubset({0, 1}) or non_null_vals.issubset({True, False}):
                return DiversityLevel.BINARY
            return DiversityLevel.LOW
        if unique_ratio <= 0.2:
            return DiversityLevel.LOW
        if unique_ratio <= 0.5:
            return DiversityLevel.MEDIUM
        # Use count threshold for high vs distinctive when ratio is high
        if unique_ratio < 0.9:
            return DiversityLevel.HIGH
        elif unique_count < 75:  # High diversity but not too many unique values
            return DiversityLevel.HIGH
        else:
            return DiversityLevel.DISTINCTIVE
    
    def detect_nullability_level(self, series: pd.Series) -> NullabilityLevel:
        """
        Detect the nullability level of column values.
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            NullabilityLevel enum value
        """
        total = len(series)
        if total == 0:
            return NullabilityLevel.EMPTY
        nulls = series.isnull().sum()
        if nulls == total:
            return NullabilityLevel.EMPTY
        null_ratio = nulls / total
        if null_ratio == 0:
            return NullabilityLevel.LOW
        elif null_ratio <= 0.2:
            return NullabilityLevel.LOW
        elif null_ratio <= 0.5:
            return NullabilityLevel.MEDIUM
        elif null_ratio <= 0.75:
            return NullabilityLevel.HIGH
        else:
            return NullabilityLevel.FULL
    
    def analyze_column(self, series: pd.Series) -> ColumnAttributes:
        """
        Analyze a column and return its attributes.
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            ColumnAttributes object with all detected attributes
        """
        type_category = self.detect_column_type_category(series)
        diversity_level = self.detect_diversity_level(series)
        nullability_level = self.detect_nullability_level(series)
        
        # Use len(set()) for better performance, excluding NaN values like nunique()
        unique_count = len(set(series.dropna()))
        total_count = len(series)
        null_count = series.isnull().sum()
        
        # Get most common value
        value_counts = series.value_counts(dropna=False)
        if len(value_counts) > 0:
            most_common_value = value_counts.index[0]
            most_common_frequency = value_counts.iloc[0]
        else:
            most_common_value = None
            most_common_frequency = 0
        
        # Determine if this is likely an index/identifier column
        is_likely_index = self._detect_likely_index(series, type_category, diversity_level)
        
        return ColumnAttributes(
            name=series.name,
            type_category=type_category,
            diversity_level=diversity_level,
            nullability_level=nullability_level,
            unique_count=unique_count,
            total_count=total_count,
            null_count=null_count,
            most_common_value=most_common_value,
            most_common_frequency=most_common_frequency,
            is_likely_index=is_likely_index
        )
    
    def _detect_likely_index(self, series: pd.Series, type_category: TypeCategory, 
                            diversity_level: DiversityLevel) -> bool:
        """
        Detect if a column is likely an index/identifier column.
        
        Args:
            series: Pandas Series to analyze
            type_category: Already detected type category
            diversity_level: Already detected diversity level
            
        Returns:
            True if the column is likely an index/identifier
        """
        # Only consider fully unique columns as potential indices
        if diversity_level != DiversityLevel.FULLY_UNIQUE:
            return False
        
        # Skip continuous numeric columns (likely measurements, not indices)
        # Integer columns can still be indices (ID fields)
        if type_category == TypeCategory.NUMERIC:
            # Check if it's likely a continuous variable (floats) vs discrete IDs (integers)
            non_null = series.dropna()
            if len(non_null) == 0:
                return False
            
            # If all values are floats (not whole numbers), likely continuous
            if non_null.dtype in ['float32', 'float64']:
                # Check if values are actually whole numbers stored as floats
                try:
                    is_whole_numbers = (non_null % 1 == 0).all()
                    if not is_whole_numbers:
                        return False  # Continuous float values, not an index
                except:
                    return False
        
        # These types are good candidates for indices
        if type_category in [TypeCategory.TEXTUAL, TypeCategory.CATEGORICAL]:
            return True
        
        # Integer columns (including those stored as floats but are whole numbers)
        if type_category == TypeCategory.NUMERIC:
            return True
        
        # Temporal columns can be indices (timestamps)
        if type_category == TypeCategory.TEMPORAL:
            return True
        
        # Boolean columns with full uniqueness are unusual but possible
        if type_category == TypeCategory.BOOLEAN:
            return True
        
        return False

    # ----------------- Compatibility private-method aliases -----------------
    def _detect_column_type(self, series: pd.Series) -> TypeCategory:
        """Heuristic type detection used by tests (robust and predictable)."""
        # Empty series
        if len(series) == 0:
            return TypeCategory.UNKNOWN
        # Boolean first
        if pd.api.types.is_bool_dtype(series):
            return TypeCategory.BOOLEAN
        # Datetime second
        if pd.api.types.is_datetime64_any_dtype(series):
            return TypeCategory.TEMPORAL
        # String dtype
        if pd.api.types.is_string_dtype(series):
            non_null = series.dropna()
            if len(non_null) == 0:
                return TypeCategory.UNKNOWN
            nunique = len(set(non_null))
            # If duplicates exist and small set, categorical; else textual
            if nunique < len(non_null) and nunique <= 10:
                return TypeCategory.CATEGORICAL
            return TypeCategory.TEXTUAL
        # Object dtype handling
        if series.dtype == object:
            non_null = series.dropna()
            if len(non_null) == 0:
                return TypeCategory.UNKNOWN
            unique_vals = set(non_null.unique())
            if unique_vals.issubset({True, False}):
                return TypeCategory.BOOLEAN
            # If all observed are strings, textual unless very small set with duplicates
            sample = list(non_null.head(min(50, len(non_null))))
            if all(isinstance(x, str) for x in sample):
                nunique = len(set(non_null))
                if nunique < len(non_null) and nunique <= 10:
                    return TypeCategory.CATEGORICAL
                return TypeCategory.TEXTUAL
            # Small set of distinct values => categorical
            if len(set(non_null)) <= 10:
                return TypeCategory.CATEGORICAL
            return TypeCategory.UNKNOWN
        # Categorical dtype
        if pd.api.types.is_categorical_dtype(series):
            return TypeCategory.CATEGORICAL
        # Numeric last
        if pd.api.types.is_numeric_dtype(series):
            return TypeCategory.NUMERIC
        return TypeCategory.UNKNOWN

    def _detect_diversity_level(self, series: pd.Series) -> DiversityLevel:
        return self.detect_diversity_level(series)

    def _detect_nullability_level(self, series: pd.Series) -> NullabilityLevel:
        return self.detect_nullability_level(series)

    def _analyze_column(self, name: str, series: pd.Series) -> ColumnAttributes:
        s = series.copy()
        s.name = name
        return self.analyze_column(s)
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, ColumnAttributes]:
        """
        Analyze all columns in a dataframe.
        
        Args:
            df: Pandas DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to ColumnAttributes
        """
        return {
            col_name: self.analyze_column(df[col_name])
            for col_name in df.columns
        }
    
    def should_enable_sampling(self, df: pd.DataFrame, threshold: int = 100000) -> bool:
        """
        Determine if sampling should be enabled based on dataframe size.
        
        Args:
            df: Pandas DataFrame to analyze
            threshold: Row count threshold for enabling sampling
            
        Returns:
            True if sampling should be enabled
        """
        return len(df) > threshold
    
    def create_sample(self, df: pd.DataFrame, sample_size: int = 10000) -> pd.DataFrame:
        """
        Create a random sample of the dataframe.
        
        Args:
            df: Pandas DataFrame to sample
            sample_size: Size of the sample
            
        Returns:
            Sampled DataFrame
        """
        if len(df) <= sample_size:
            return df
        
        return df.sample(n=sample_size, random_state=self.seed)
    
    def get_optimal_sample_size(self, series: pd.Series, base_sample_size: int = 10000) -> int:
        """
        Determine optimal sample size based on column characteristics.
        
        Args:
            series: Pandas Series to analyze
            base_sample_size: Base sample size for normal columns
            
        Returns:
            Optimal sample size for this column
        """
        if len(series) <= base_sample_size:
            return len(series)
            
        # Calculate unique ratio efficiently using len(set())
        non_null = series.dropna()
        if len(non_null) == 0:
            return min(1000, len(series))  # Small sample for all-null columns
            
        unique_count = len(set(non_null))
        unique_ratio = unique_count / len(non_null)
        
        # For high-cardinality columns (nearly unique), use larger samples
        if unique_ratio > 0.8:
            return min(50000, len(series))
        # For low diversity columns, smaller sample is sufficient
        elif unique_ratio < 0.1:
            return min(5000, len(series))
        # Standard columns
        else:
            return min(base_sample_size, len(series))
    
    def get_relevant_rule_categories(self, attributes: ColumnAttributes) -> List[str]:
        """
        Determine which rule categories are relevant for a column.
        
        Args:
            attributes: ColumnAttributes object
            
        Returns:
            List of relevant rule category names
        """
        categories = []
        
        # Type-specific categories
        if attributes.type_category == TypeCategory.NUMERIC:
            categories.extend(["numeric_stats", "outlier_detection", "distribution_analysis"])
        elif attributes.type_category == TypeCategory.TEXTUAL:
            categories.extend(["text_patterns", "length_analysis", "format_validation"])
        elif attributes.type_category == TypeCategory.TEMPORAL:
            categories.extend(["date_validation", "temporal_patterns"])
        elif attributes.type_category == TypeCategory.BOOLEAN:
            categories.extend(["boolean_consistency"])
        
        # Diversity-specific categories
        if attributes.diversity_level == DiversityLevel.CONSTANT:
            categories.append("constant_value_checks")
        elif attributes.diversity_level == DiversityLevel.BINARY:
            categories.append("binary_value_checks")
        elif attributes.diversity_level in [DiversityLevel.HIGH, DiversityLevel.DISTINCTIVE]:
            categories.append("high_diversity_checks")
        
        # Nullability-specific categories
        if attributes.nullability_level != NullabilityLevel.EMPTY:
            categories.append("null_value_analysis")
        if attributes.nullability_level in [NullabilityLevel.HIGH, NullabilityLevel.FULL]:
            categories.append("high_null_analysis")
        
        # Always include basic categories
        categories.extend(["basic_stats", "uniqueness_analysis"])
        
        return list(set(categories))  # Remove duplicates
