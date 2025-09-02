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
    most_common_value: Any
    most_common_frequency: int


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
        if pd.api.types.is_numeric_dtype(series):
            return TypeCategory.NUMERIC
        elif pd.api.types.is_string_dtype(series):
            return TypeCategory.TEXTUAL
        elif pd.api.types.is_datetime64_any_dtype(series):
            return TypeCategory.TEMPORAL
        elif pd.api.types.is_bool_dtype(series):
            return TypeCategory.BOOLEAN
        elif pd.api.types.is_categorical_dtype(series):
            return TypeCategory.CATEGORICAL
        else:
            return TypeCategory.UNKNOWN
    
    def detect_diversity_level(self, series: pd.Series) -> DiversityLevel:
        """
        Detect the diversity level of column values.
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            DiversityLevel enum value
        """
        unique_count = series.nunique()
        total_count = len(series)
        unique_ratio = unique_count / total_count if total_count > 0 else 0
        
        if unique_count == 1:
            return DiversityLevel.CONSTANT
        elif unique_count == 2:
            return DiversityLevel.BINARY
        elif unique_ratio < 0.01:
            return DiversityLevel.LOW
        elif unique_ratio < 0.1:
            return DiversityLevel.MEDIUM
        elif unique_ratio < 0.5:
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
        null_ratio = series.isnull().sum() / len(series)
        
        if null_ratio == 0:
            return NullabilityLevel.EMPTY
        elif null_ratio <= 0.05:
            return NullabilityLevel.LOW
        elif null_ratio <= 0.25:
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
        
        unique_count = series.nunique()
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
        
        return ColumnAttributes(
            name=series.name,
            type_category=type_category,
            diversity_level=diversity_level,
            nullability_level=nullability_level,
            unique_count=unique_count,
            total_count=total_count,
            null_count=null_count,
            most_common_value=most_common_value,
            most_common_frequency=most_common_frequency
        )
    
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
