# dataset_tools/correct_types.py

# Copyright (c) 2025 [KTISEOS NYX / 0FTH3N1GHT / EARTH & DUSK MEDIA]
# SPDX-License-Identifier: GPL-3.0

"""
Type definitions, validation utilities, and constants for Dataset Tools.

This module provides:
- Enum definitions for UI field organization
- File extension categorization
- ComfyUI-specific node constants
- Type validation utilities
- Pydantic models for data validation
"""

import sys
from enum import Enum
from typing import Any, ClassVar, Union

from pydantic import BaseModel, TypeAdapter, field_validator

from dataset_tools import LOG_LEVEL

# ============================================================================
# PYTHON VERSION COMPATIBILITY
# ============================================================================

# Handle TypedDict import for Pydantic V2 compatibility
if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Parsing and display limits
MAX_REMAINING_UNPARSED_SPLIT_LEN = 5
MAX_RAW_METADATA_DISPLAY_LEN = 500

# Debugging configuration based on log level
DEBUG_MODES = {"DEBUG", "TRACE", "NOTSET", "ALL"}
EXC_INFO: bool = LOG_LEVEL.strip().upper() in DEBUG_MODES

# ============================================================================
# UI FIELD ENUMERATIONS
# ============================================================================

class EmptyField(Enum):
    """
    Placeholder and empty field state constants.
    
    Used as keys in metadata dictionaries and as UI placeholder text sources.
    """
    
    # Internal placeholders
    PLACEHOLDER = "_dt_internal_placeholder_"
    EMPTY = "_dt_internal_empty_value_"
    
    # User-facing placeholders
    PLACEHOLDER_POSITIVE = "Positive prompt will appear here."
    PLACEHOLDER_NEGATIVE = "Negative prompt will appear here."
    PLACEHOLDER_DETAILS = "Generation details and other metadata will appear here."


class UpField(Enum):
    """
    Define sections for the upper display area in the UI.
    
    The string values serve as keys in the metadata dictionary
    and determine the order of display in the upper panel.
    """
    
    METADATA = "metadata_info_section"
    PROMPT = "prompt_data_section"
    TAGS = "tags_and_keywords_section"
    TEXT_DATA = "text_file_content_section"
    
    @classmethod
    def get_ordered_labels(cls) -> list["UpField"]:
        """
        Return ordered list of UpField members for UI iteration.
        
        Returns:
            List of UpField enum members in display order
        """
        return [cls.PROMPT, cls.TAGS, cls.METADATA, cls.TEXT_DATA]
    
    @classmethod
    def get_section_keys(cls) -> list[str]:
        """
        Get the string values of all sections.
        
        Returns:
            List of section key strings
        """
        return [field.value for field in cls.get_ordered_labels()]


class DownField(Enum):
    """
    Define sections for the lower display area in the UI.
    
    The string values serve as keys in the metadata dictionary
    and determine the order of display in the lower panel.
    """
    
    GENERATION_DATA = "generation_parameters_section"
    RAW_DATA = "raw_tool_specific_data_section"
    EXIF = "standard_exif_data_section"
    JSON_DATA = "json_file_content_section"
    TOML_DATA = "toml_file_content_section"
    
    @classmethod
    def get_ordered_labels(cls) -> list["DownField"]:
        """
        Return ordered list of DownField members for UI iteration.
        
        Returns:
            List of DownField enum members in display order
        """
        return [
            cls.GENERATION_DATA,
            cls.RAW_DATA,
            cls.EXIF,
            cls.JSON_DATA,
            cls.TOML_DATA,
        ]
    
    @classmethod
    def get_section_keys(cls) -> list[str]:
        """
        Get the string values of all sections.
        
        Returns:
            List of section key strings
        """
        return [field.value for field in cls.get_ordered_labels()]


# ============================================================================
# FILE TYPE DEFINITIONS
# ============================================================================

class ExtensionType:
    """
    File extension categorization for processing different file types.
    
    This class organizes file extensions into logical groups for
    different processing pipelines and validation.
    """
    
    # Image formats
    PNG_: ClassVar[set[str]] = {".png"}
    JPEG: ClassVar[set[str]] = {".jpg", ".jpeg"}
    WEBP: ClassVar[set[str]] = {".webp"}
    
    # Data formats
    JSON: ClassVar[set[str]] = {".json"}
    TOML: ClassVar[set[str]] = {".toml"}
    TEXT: ClassVar[set[str]] = {".txt", ".text"}
    HTML: ClassVar[set[str]] = {".html", ".htm"}
    XML_: ClassVar[set[str]] = {".xml"}
    
    # Model formats
    GGUF: ClassVar[set[str]] = {".gguf"}
    SAFE: ClassVar[set[str]] = {".safetensors", ".sft"}
    PICK: ClassVar[set[str]] = {".pt", ".pth", ".ckpt", ".pickletensor"}
    
    # Grouped categories for processing
    IMAGE: ClassVar[list[set[str]]] = [PNG_, JPEG, WEBP]
    EXIF_CAPABLE: ClassVar[list[set[str]]] = [JPEG, WEBP]
    SCHEMA_FILES: ClassVar[list[set[str]]] = [JSON, TOML]
    PLAIN_TEXT_LIKE: ClassVar[list[set[str]]] = [TEXT, XML_, HTML]
    MODEL_FILES: ClassVar[list[set[str]]] = [SAFE, GGUF, PICK]
    
    # Files to ignore during processing
    IGNORE: ClassVar[list[str]] = [
        "Thumbs.db",     # Windows thumbnail cache
        "desktop.ini",   # Windows folder settings
        ".DS_Store",     # macOS folder metadata
        ".fseventsd",    # macOS file system events
        "._*",           # macOS resource forks
        "~$*",           # Office temporary files
        "~$*.tmp",       # Office temporary files
        "*.tmp",         # Generic temporary files
    ]
    
    @classmethod
    def get_all_supported_extensions(cls) -> set[str]:
        """
        Get all supported file extensions across all categories.
        
        Returns:
            Set of all supported file extensions
        """
        extensions = set()
        
        # Add individual format extensions
        for attr_name in dir(cls):
            attr_value = getattr(cls, attr_name)
            if isinstance(attr_value, set) and not attr_name.startswith('_'):
                extensions.update(attr_value)
        
        return extensions
    
    @classmethod
    def get_image_extensions(cls) -> set[str]:
        """Get all image file extensions."""
        extensions = set()
        for ext_set in cls.IMAGE:
            extensions.update(ext_set)
        return extensions
    
    @classmethod
    def get_model_extensions(cls) -> set[str]:
        """Get all model file extensions."""
        extensions = set()
        for ext_set in cls.MODEL_FILES:
            extensions.update(ext_set)
        return extensions
    
    @classmethod
    def is_image_file(cls, extension: str) -> bool:
        """Check if extension is an image format."""
        extension = extension.lower()
        return any(extension in ext_set for ext_set in cls.IMAGE)
    
    @classmethod
    def is_model_file(cls, extension: str) -> bool:
        """Check if extension is a model format."""
        extension = extension.lower()
        return any(extension in ext_set for ext_set in cls.MODEL_FILES)
    
    @classmethod
    def should_ignore_file(cls, filename: str) -> bool:
        """Check if a file should be ignored during processing."""
        filename_lower = filename.lower()
        return any(
            filename_lower == pattern.lower() or 
            (pattern.endswith('*') and filename_lower.startswith(pattern[:-1].lower()))
            for pattern in cls.IGNORE
        )


# ============================================================================
# COMFYUI NODE DEFINITIONS
# ============================================================================

class NodeNames:
    """
    Constants for ComfyUI node identification and data extraction.
    
    This class contains sets and mappings used to identify different
    types of nodes in ComfyUI workflows and extract relevant data.
    """
    
    # Text encoding nodes
    ENCODERS: ClassVar[set[str]] = {
        "CLIPTextEncodeFlux",
        "CLIPTextEncodeSD3",
        "CLIPTextEncodeSDXL",
        "CLIPTextEncodeHunyuanDiT",
        "CLIPTextEncodePixArtAlpha",
        "CLIPTextEncodeSDXLRefiner",
        "ImpactWildcardEncodeCLIPTextEncode",
        "BNK_CLIPTextEncodeAdvanced",
        "BNK_CLIPTextEncodeSDXLAdvanced",
        "WildcardEncode //Inspire",
        "TSC_EfficientLoader",
        "TSC_EfficientLoaderSDXL",
        "RgthreePowerPrompt",
        "RgthreePowerPromptSimple",
        "RgthreeSDXLPowerPromptPositive",
        "RgthreeSDXLPowerPromptSimple",
        "AdvancedCLIPTextEncode",
        "AdvancedCLIPTextEncodeWithBreak",
        "Text2Prompt",
        "smZ CLIPTextEncode",
        "CLIPTextEncode",
    }
    
    # String input/processing nodes
    STRING_INPUT: ClassVar[set[str]] = {
        "RecourseStrings",
        "StringSelector",
        "ImpactWildcardProcessor",
        "CText",
        "CTextML",
        "CListString",
        "CSwitchString",
        "CR_PromptText",
        "StringLiteral",
        "CR_CombinePromptSDParameterGenerator",
        "WidgetToString",
        "Show Text 🐍",
    }
    
    # UI labels for prompt identification
    PROMPT_LABELS: ClassVar[list[str]] = [
        "Positive prompt",
        "Negative prompt",
        "Prompt",
    ]
    
    # Keys to ignore during node processing
    IGNORE_KEYS: ClassVar[set[str]] = {
        "type", "link", "shape", "id", "pos", "size", 
        "node_id", "empty_padding"
    }
    
    # Mapping for data extraction paths
    DATA_KEYS: ClassVar[dict[str, str]] = {
        "class_type": "inputs",
        "nodes": "widget_values",
    }
    
    # Fields that typically contain prompt text
    PROMPT_NODE_FIELDS: ClassVar[set[str]] = {
        "text", "t5xxl", "clip-l", "clip-g", "mt5", "mt5xl",
        "bert", "clip-h", "wildcard", "string", "positive", 
        "negative", "text_g", "text_l", "wildcard_text", 
        "populated_text"
    }
    
    @classmethod
    def is_encoder_node(cls, class_type: str) -> bool:
        """Check if a node type is a text encoder."""
        return class_type in cls.ENCODERS
    
    @classmethod
    def is_string_input_node(cls, class_type: str) -> bool:
        """Check if a node type handles string input."""
        return class_type in cls.STRING_INPUT
    
    @classmethod
    def is_prompt_field(cls, field_name: str) -> bool:
        """Check if a field name typically contains prompt text."""
        return field_name in cls.PROMPT_NODE_FIELDS
    
    @classmethod
    def should_ignore_key(cls, key_name: str) -> bool:
        """Check if a key should be ignored during processing."""
        return key_name in cls.IGNORE_KEYS


# ============================================================================
# TYPE DEFINITIONS FOR COMFYUI DATA
# ============================================================================

class NodeDataMap(TypedDict):
    """Type definition for ComfyUI node data structure."""
    class_type: str
    inputs: Union[dict[str, Any], float, str, list[Any], None]


class NodeWorkflow(TypedDict):
    """Type definition for ComfyUI workflow structure."""
    last_node_id: int
    last_link_id: Union[int, dict[str, Any], None]
    nodes: list[NodeDataMap]
    links: list[Any]
    groups: list[Any]
    config: dict[str, Any]
    extra: dict[str, Any]
    version: float


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def bracket_check(maybe_brackets: Union[str, dict[str, Any]]) -> Union[str, dict[str, Any]]:
    """
    Ensure a string is bracket-enclosed for JSON parsing, pass through dicts.
    
    Args:
        maybe_brackets: String to check/correct or dict to pass through
        
    Returns:
        Dictionary unchanged, or string with proper bracket enclosure
        
    Raises:
        TypeError: If input is neither string nor dict
        
    Examples:
        >>> bracket_check('{"key": "value"}')
        '{"key": "value"}'
        >>> bracket_check('"key": "value"')
        '{"key": "value"}'
        >>> bracket_check({"key": "value"})
        {"key": "value"}
    """
    if isinstance(maybe_brackets, dict):
        return maybe_brackets
    
    if isinstance(maybe_brackets, str):
        corrected_str = maybe_brackets.strip()
        
        if not corrected_str.startswith("{"):
            corrected_str = "{" + corrected_str
        
        if not corrected_str.endswith("}"):
            corrected_str = corrected_str + "}"
        
        return corrected_str
    
    raise TypeError(
        f"Input must be a string or dictionary, got {type(maybe_brackets).__name__}"
    )


def validate_extension(extension: str) -> str:
    """
    Normalize and validate a file extension.
    
    Args:
        extension: File extension to validate
        
    Returns:
        Normalized extension with leading dot
        
    Examples:
        >>> validate_extension("png")
        ".png"
        >>> validate_extension(".PNG")
        ".png"
    """
    if not extension:
        raise ValueError("Extension cannot be empty")
    
    # Add leading dot if missing
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    # Normalize to lowercase
    return extension.lower()


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class BracketedDict(BaseModel):
    """
    Base model for validating bracket-enclosed dictionary data.
    
    This serves as a foundation for models that need to validate
    JSON-like string data that may need bracket correction.
    """
    
    @classmethod
    def from_string_or_dict(cls, data: Union[str, dict[str, Any]]) -> "BracketedDict":
        """
        Create instance from string or dict, applying bracket correction.
        
        Args:
            data: String (potentially needing brackets) or dict
            
        Returns:
            Validated model instance
        """
        corrected_data = bracket_check(data)
        return cls.model_validate(corrected_data)


class ListOfDelineatedStr(BaseModel):
    """
    Model for processing lists that may contain tuples needing extraction.
    
    This handles the common case where data comes in as [(value,), ...]
    and needs to be flattened to [value, ...].
    """
    
    convert: list[Any]
    
    @field_validator("convert")
    @classmethod
    def extract_from_tuples(cls, v: list[Any]) -> list[Any]:
        """
        Extract values from tuple-wrapped list items.
        
        Args:
            v: List that may contain tuples as first elements
            
        Returns:
            List with tuple contents extracted
        """
        if not v:
            return v
        
        # Check if first element is a tuple and extract its first value
        if isinstance(v[0], tuple) and v[0]:
            first_element = v[0][0]
            return [first_element] if first_element is not None else []
        
        return v


# ============================================================================
# TYPE ADAPTERS FOR VALIDATION
# ============================================================================

class ValidationAdapters:
    """
    Type adapters for validating ComfyUI JSON data structures.
    
    These adapters provide runtime validation for complex nested
    data structures commonly found in ComfyUI workflows.
    """
    
    node_data: TypeAdapter[NodeDataMap] = TypeAdapter(NodeDataMap)
    workflow: TypeAdapter[NodeWorkflow] = TypeAdapter(NodeWorkflow)
    
    @classmethod
    def validate_node_data(cls, data: Any) -> NodeDataMap:
        """Validate and return typed node data."""
        return cls.node_data.validate_python(data)
    
    @classmethod
    def validate_workflow(cls, data: Any) -> NodeWorkflow:
        """Validate and return typed workflow data."""
        return cls.workflow.validate_python(data)


# Backwards compatibility alias
IsThisNode = ValidationAdapters

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_field_placeholder(field_type: str) -> str:
    """
    Get appropriate placeholder text for a UI field type.
    
    Args:
        field_type: Type of field ("positive", "negative", "details")
        
    Returns:
        Appropriate placeholder text
    """
    placeholders = {
        "positive": EmptyField.PLACEHOLDER_POSITIVE.value,
        "negative": EmptyField.PLACEHOLDER_NEGATIVE.value,
        "details": EmptyField.PLACEHOLDER_DETAILS.value,
    }
    
    return placeholders.get(field_type, EmptyField.PLACEHOLDER.value)


def is_debug_mode() -> bool:
    """Check if application is running in debug mode."""
    return EXC_INFO