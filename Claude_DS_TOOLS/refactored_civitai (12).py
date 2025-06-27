# dataset_tools/vendored_sdpr/format/utility.py

__author__ = "receyuki"
__filename__ = "utility.py"
# MODIFIED by Ktiseos Nyx for Dataset-Tools: Enhanced with systematic engineering
__copyright__ = "Copyright 2023, Receyuki; Modified 2025, Ktiseos Nyx"
__email__ = "receyuki@gmail.com"

"""
Core utility functions for metadata parsing and data manipulation.

This module provides foundational string processing, data structure manipulation,
and merge operations used across all format parsers in the framework.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class QuoteStyle(Enum):
    """Enumeration of quote styles for string processing"""
    DOUBLE = '"'
    SINGLE = "'"
    SMART_DOUBLE = '""'
    SMART_SINGLE = "''"


@dataclass
class StringProcessingConfig:
    """Configuration for string processing operations"""
    
    # Quote detection patterns
    QUOTE_PATTERNS = {
        QuoteStyle.DOUBLE: ('"', '"'),
        QuoteStyle.SINGLE: ("'", "'"),
        QuoteStyle.SMART_DOUBLE: ('"', '"'),
        QuoteStyle.SMART_SINGLE: (''', '''),
    }
    
    # Default separators for concatenation
    DEFAULT_SEPARATORS = {
        "comma": ", ",
        "semicolon": "; ",
        "pipe": " | ",
        "dash": " - ",
        "space": " ",
        "newline": "\n",
    }
    
    # Whitespace normalization pattern
    WHITESPACE_NORMALIZE = re.compile(r'\s+')


class StringProcessor:
    """Advanced string processing utilities with comprehensive quote handling"""
    
    def __init__(self, config: Optional[StringProcessingConfig] = None):
        self.config = config or StringProcessingConfig()
        
    def remove_quotes(self, text: str, quote_styles: Optional[List[QuoteStyle]] = None) -> str:
        """
        Advanced quote removal with support for multiple quote styles.
        
        Args:
            text: Input string to process
            quote_styles: List of quote styles to check (default: all styles)
            
        Returns:
            String with outermost quotes removed if matched
        """
        if not isinstance(text, str):
            text = str(text)
            
        if not text:
            return text
            
        # Default to checking all quote styles
        if quote_styles is None:
            quote_styles = list(QuoteStyle)
            
        text = text.strip()
        
        for style in quote_styles:
            start_quote, end_quote = self.config.QUOTE_PATTERNS[style]
            if text.startswith(start_quote) and text.endswith(end_quote) and len(text) >= 2:
                return text[len(start_quote):-len(end_quote)]
                
        return text
        
    def add_quotes(self, text: str, style: QuoteStyle = QuoteStyle.DOUBLE) -> str:
        """
        Add quotes around a string with specified style.
        
        Args:
            text: Input string to quote
            style: Quote style to use
            
        Returns:
            Quoted string
        """
        if not isinstance(text, str):
            text = str(text)
            
        start_quote, end_quote = self.config.QUOTE_PATTERNS[style]
        return f"{start_quote}{text}{end_quote}"
        
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in a string.
        
        Args:
            text: Input string to normalize
            
        Returns:
            String with normalized whitespace
        """
        if not isinstance(text, str):
            text = str(text)
            
        # Replace multiple whitespace characters with single space
        normalized = self.config.WHITESPACE_NORMALIZE.sub(' ', text)
        return normalized.strip()
        
    def smart_concat(self, base: str, addition: str, 
                    separator: str = None, normalize: bool = True) -> str:
        """
        Intelligently concatenate strings with proper separator handling.
        
        Args:
            base: Base string
            addition: String to add
            separator: Separator to use (default: comma-space)
            normalize: Whether to normalize whitespace
            
        Returns:
            Concatenated string
        """
        if separator is None:
            separator = self.config.DEFAULT_SEPARATORS["comma"]
            
        base_str = str(base).strip() if base else ""
        addition_str = str(addition).strip() if addition else ""
        
        if not addition_str:
            return base_str
        if not base_str:
            return addition_str
            
        result = f"{base_str}{separator}{addition_str}"
        
        if normalize:
            result = self.normalize_whitespace(result)
            
        return result


class DataStructureProcessor:
    """Advanced data structure manipulation utilities"""
    
    @staticmethod
    def smart_merge_values(*items) -> Union[str, Tuple[Any, ...]]:
        """
        Intelligently merge multiple values into appropriate structure.
        
        Args:
            *items: Values to merge
            
        Returns:
            Single value if only one non-empty item, tuple if multiple
        """
        # Filter out None, empty strings, and empty collections
        filtered_items = [
            item for item in items 
            if item is not None and item != "" and item != [] and item != {}
        ]
        
        if not filtered_items:
            return ""
        elif len(filtered_items) == 1:
            return filtered_items[0]
        else:
            return tuple(filtered_items)
            
    @staticmethod
    def merge_to_tuple(item1: Any, item2: Any) -> Tuple[Any, ...]:
        """
        Merge items into a tuple, handling existing tuples intelligently.
        
        Args:
            item1: First item
            item2: Second item
            
        Returns:
            Tuple containing both items
        """
        # Convert single items to tuples
        tuple1 = item1 if isinstance(item1, tuple) else (item1,)
        tuple2 = item2 if isinstance(item2, tuple) else (item2,)
        
        return tuple1 + tuple2
        
    @staticmethod
    def deep_merge_dict(base_dict: Dict[str, Any], 
                       update_dict: Dict[str, Any],
                       merge_strategy: str = "tuple") -> Dict[str, Any]:
        """
        Deep merge two dictionaries with configurable merge strategy.
        
        Args:
            base_dict: Base dictionary
            update_dict: Dictionary to merge in
            merge_strategy: How to handle conflicts ("tuple", "replace", "list")
            
        Returns:
            Merged dictionary
        """
        result = base_dict.copy()
        
        for key, value in update_dict.items():
            if key in result:
                if merge_strategy == "tuple":
                    result[key] = DataStructureProcessor.merge_to_tuple(result[key], value)
                elif merge_strategy == "list":
                    # Convert to list and combine
                    existing = result[key] if isinstance(result[key], list) else [result[key]]
                    new_value = value if isinstance(value, list) else [value]
                    result[key] = existing + new_value
                elif merge_strategy == "replace":
                    result[key] = value
                else:
                    raise ValueError(f"Unknown merge strategy: {merge_strategy}")
            else:
                result[key] = value
                
        return result


class ValidationUtilities:
    """Utilities for data validation and sanitization"""
    
    @staticmethod
    def is_valid_string(value: Any, min_length: int = 0, max_length: Optional[int] = None) -> bool:
        """
        Validate if a value is a valid string within specified constraints.
        
        Args:
            value: Value to validate
            min_length: Minimum required length
            max_length: Maximum allowed length (None for no limit)
            
        Returns:
            True if valid string
        """
        if not isinstance(value, str):
            return False
            
        length = len(value.strip())
        if length < min_length:
            return False
            
        if max_length is not None and length > max_length:
            return False
            
        return True
        
    @staticmethod
    def sanitize_key_name(key: str) -> str:
        """
        Sanitize a key name for safe usage.
        
        Args:
            key: Key name to sanitize
            
        Returns:
            Sanitized key name
        """
        if not isinstance(key, str):
            key = str(key)
            
        # Replace problematic characters with underscores
        sanitized = re.sub(r'[^\w\-]', '_', key)
        
        # Remove multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = "unknown_key"
            
        return sanitized
        
    @staticmethod
    def safe_convert_to_number(value: Any) -> Optional[Union[int, float]]:
        """
        Safely convert a value to a number.
        
        Args:
            value: Value to convert
            
        Returns:
            Converted number or None if conversion fails
        """
        if isinstance(value, (int, float)):
            return value
            
        if not isinstance(value, str):
            value = str(value)
            
        # Clean the string
        cleaned = value.strip()
        
        # Try integer conversion first
        try:
            return int(cleaned)
        except ValueError:
            pass
            
        # Try float conversion
        try:
            return float(cleaned)
        except ValueError:
            return None


# Global processor instances for convenience
_string_processor = StringProcessor()
_data_processor = DataStructureProcessor()


# --- Backwards Compatibility Functions ---
# Maintain API compatibility with existing parsers

def remove_quotes(string: str) -> str:
    """
    Legacy function for quote removal.
    Maintained for backwards compatibility.
    """
    return _string_processor.remove_quotes(string)


def add_quotes(string: str) -> str:
    """
    Legacy function for adding quotes.
    Maintained for backwards compatibility.
    """
    return _string_processor.add_quotes(string)


def concat_strings(base: str, addition: str, separator: str = ", ") -> str:
    """
    Legacy function for string concatenation.
    Maintained for backwards compatibility.
    """
    return _string_processor.smart_concat(base, addition, separator)


def merge_str_to_tuple(item1: Any, item2: Any) -> Tuple[Any, ...]:
    """
    Legacy function for tuple merging.
    Maintained for backwards compatibility.
    """
    return _data_processor.merge_to_tuple(item1, item2)


def merge_dict(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Legacy function for dictionary merging.
    Maintained for backwards compatibility.
    """
    return _data_processor.deep_merge_dict(dict1, dict2, "tuple")


# --- Enhanced Utility Functions ---
# New functions with advanced capabilities

def smart_string_split(text: str, delimiters: Union[str, List[str]], 
                      max_splits: Optional[int] = None) -> List[str]:
    """
    Smart string splitting with multiple delimiter support.
    
    Args:
        text: Text to split
        delimiters: Single delimiter or list of delimiters
        max_splits: Maximum number of splits (None for unlimited)
        
    Returns:
        List of split strings
    """
    if not isinstance(text, str):
        text = str(text)
        
    if isinstance(delimiters, str):
        delimiters = [delimiters]
        
    # Create regex pattern from delimiters
    escaped_delims = [re.escape(d) for d in delimiters]
    pattern = '|'.join(escaped_delims)
    
    # Split with optional limit
    if max_splits is not None:
        parts = re.split(pattern, text, maxsplit=max_splits)
    else:
        parts = re.split(pattern, text)
        
    # Clean up results
    return [part.strip() for part in parts if part.strip()]


def format_key_for_display(key: str, style: str = "title") -> str:
    """
    Format a key name for display purposes.
    
    Args:
        key: Key to format
        style: Format style ("title", "sentence", "upper", "lower")
        
    Returns:
        Formatted key name
    """
    if not isinstance(key, str):
        key = str(key)
        
    # Replace underscores and hyphens with spaces
    formatted = re.sub(r'[_-]', ' ', key)
    
    # Apply style
    if style == "title":
        formatted = formatted.title()
    elif style == "sentence":
        formatted = formatted.capitalize()
    elif style == "upper":
        formatted = formatted.upper()
    elif style == "lower":
        formatted = formatted.lower()
        
    return formatted


def build_settings_string(items: Dict[str, Any], 
                         separator: str = ", ",
                         key_value_sep: str = ": ",
                         sort_keys: bool = True,
                         exclude_empty: bool = True) -> str:
    """
    Build a formatted settings string from a dictionary.
    
    Args:
        items: Dictionary of settings
        separator: Separator between items
        key_value_sep: Separator between key and value
        sort_keys: Whether to sort keys alphabetically
        exclude_empty: Whether to exclude empty values
        
    Returns:
        Formatted settings string
    """
    if not items:
        return ""
        
    formatted_items = []
    
    keys = sorted(items.keys()) if sort_keys else items.keys()
    
    for key in keys:
        value = items[key]
        
        if exclude_empty and (value is None or value == ""):
            continue
            
        # Format key for display
        display_key = format_key_for_display(key)
        
        # Process value
        if isinstance(value, (list, tuple)):
            value_str = ", ".join(str(v) for v in value)
        else:
            value_str = str(value)
            
        formatted_items.append(f"{display_key}{key_value_sep}{value_str}")
        
    return separator.join(formatted_items)


# Export list for explicit imports
__all__ = [
    # Legacy compatibility functions
    "add_quotes",
    "concat_strings", 
    "merge_dict",
    "merge_str_to_tuple",
    "remove_quotes",
    
    # Enhanced utility functions
    "smart_string_split",
    "format_key_for_display",
    "build_settings_string",
    
    # Advanced processor classes
    "StringProcessor",
    "DataStructureProcessor",
    "ValidationUtilities",
    
    # Configuration classes
    "StringProcessingConfig",
    "QuoteStyle",
]