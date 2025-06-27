# dataset_tools/vendored_sdpr/format/drawthings.py

import json
import logging
from typing import Any, Dict, Optional, Set
from dataclasses import dataclass, field

from .base_format import BaseFormat


@dataclass
class DrawThingsConfig:
    """Configuration for DrawThings format detection and parsing"""
    
    # Parameter mapping for standardization
    PARAMETER_MAPPINGS: Dict[str, str] = field(default_factory=lambda: {
        "model": "model",
        "sampler": "sampler_name", 
        "seed": "seed",
        "scale": "cfg_scale",
        "steps": "steps",
        "scheduler": "scheduler",
        "strength": "denoising_strength",
        "guidance": "cfg_scale",  # Alternative key
    })
    
    # Required keys for format detection (at least one must be present)
    REQUIRED_KEYS: Set[str] = field(default_factory=lambda: {
        "c",        # Positive prompt
        "sampler",  # Sampler name
        "model",    # Model name
    })
    
    # Characteristic keys that strongly indicate DrawThings format
    CHARACTERISTIC_KEYS: Set[str] = field(default_factory=lambda: {
        "c",     # Positive prompt (DrawThings specific)
        "uc",    # Negative prompt (DrawThings specific) 
        "size",  # Dimensions in "WxH" format
    })
    
    # Keys to exclude from settings string
    HANDLED_KEYS: Set[str] = field(default_factory=lambda: {
        "c", "uc", "size",  # Core prompt and dimension keys
    })


class DrawThingsValidator:
    """Handles validation and format detection for DrawThings JSON"""
    
    def __init__(self, config: DrawThingsConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def validate_json_structure(self, raw_data: str) -> Optional[Dict[str, Any]]:
        """Validate and parse JSON structure"""
        if not raw_data or not isinstance(raw_data, str):
            self.logger.debug("DrawThings: No raw data provided")
            return None
            
        try:
            parsed_data = json.loads(raw_data)
        except json.JSONDecodeError as e:
            self.logger.debug(f"DrawThings: Invalid JSON - {e}")
            return None
            
        if not isinstance(parsed_data, dict):
            self.logger.debug("DrawThings: Parsed data is not a dictionary")
            return None
            
        return parsed_data
        
    def is_drawthings_format(self, data: Dict[str, Any]) -> bool:
        """Determine if the data matches DrawThings format"""
        if not data:
            return False
            
        # Check for required keys (at least one must be present)
        has_required = any(key in data for key in self.config.REQUIRED_KEYS)
        if not has_required:
            self.logger.debug("DrawThings: No required keys found")
            return False
            
        # Check for characteristic DrawThings patterns
        has_characteristic = any(key in data for key in self.config.CHARACTERISTIC_KEYS)
        if not has_characteristic:
            self.logger.debug("DrawThings: No characteristic DrawThings keys found")
            return False
            
        # Additional validation: if 'c' exists, it should be a string (prompt)
        if "c" in data and not isinstance(data["c"], str):
            self.logger.debug("DrawThings: 'c' key exists but is not a string")
            return False
            
        # If 'size' exists, it should match expected format
        if "size" in data:
            size_str = str(data["size"])
            if not self._is_valid_size_format(size_str):
                self.logger.debug(f"DrawThings: Invalid size format: {size_str}")
                return False
                
        self.logger.debug("DrawThings: Format validation passed")
        return True
        
    def _is_valid_size_format(self, size_str: str) -> bool:
        """Validate DrawThings size format (e.g., '512x512')"""
        if not size_str:
            return False
            
        # Check for WxH pattern
        if 'x' not in size_str.lower():
            return False
            
        try:
            parts = size_str.lower().split('x')
            if len(parts) != 2:
                return False
                
            # Try to parse as integers
            int(parts[0].strip())
            int(parts[1].strip())
            return True
            
        except ValueError:
            return False


class DrawThingsExtractor:
    """Handles extraction of data from validated DrawThings JSON"""
    
    def __init__(self, config: DrawThingsConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def extract_prompts(self, data: Dict[str, Any]) -> tuple[str, str]:
        """Extract positive and negative prompts"""
        positive = str(data.get("c", "")).strip()
        negative = str(data.get("uc", "")).strip()
        
        self.logger.debug(f"DrawThings: Extracted prompts - pos: {len(positive)} chars, neg: {len(negative)} chars")
        return positive, negative
        
    def extract_parameters(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Extract and standardize parameters"""
        parameters = {}
        
        for dt_key, standard_key in self.config.PARAMETER_MAPPINGS.items():
            if dt_key in data and data[dt_key] is not None:
                value = self._clean_parameter_value(data[dt_key])
                if value:  # Only add non-empty values
                    parameters[standard_key] = value
                    
        self.logger.debug(f"DrawThings: Extracted {len(parameters)} parameters")
        return parameters
        
    def extract_dimensions(self, data: Dict[str, Any]) -> tuple[str, str]:
        """Extract width and height from size string"""
        size_str = str(data.get("size", "0x0"))
        
        if not size_str or 'x' not in size_str.lower():
            return "0", "0"
            
        try:
            parts = size_str.lower().split('x')
            if len(parts) == 2:
                width = str(int(parts[0].strip()))
                height = str(int(parts[1].strip()))
                self.logger.debug(f"DrawThings: Extracted dimensions - {width}x{height}")
                return width, height
        except ValueError as e:
            self.logger.warning(f"DrawThings: Failed to parse dimensions from '{size_str}': {e}")
            
        return "0", "0"
        
    def _clean_parameter_value(self, value: Any) -> str:
        """Clean and standardize parameter values"""
        if value is None:
            return ""
            
        # Convert to string and strip whitespace
        cleaned = str(value).strip()
        
        # Remove surrounding quotes if present
        if len(cleaned) >= 2:
            if (cleaned.startswith('"') and cleaned.endswith('"')) or \
               (cleaned.startswith("'") and cleaned.endswith("'")):
                cleaned = cleaned[1:-1]
                
        return cleaned


class DrawThings(BaseFormat):
    """
    Enhanced DrawThings format parser with robust validation and extraction.
    
    DrawThings stores metadata as JSON in XMP exif:UserComment with specific structure:
    - 'c': positive prompt
    - 'uc': negative prompt (optional)
    - 'size': dimensions as 'WxH' string
    - Various generation parameters
    """
    
    tool = "Draw Things"
    
    def __init__(
        self,
        info: Optional[Dict[str, Any]] = None,
        raw: str = "",
        width: Any = 0,
        height: Any = 0,
        logger_obj: Optional[logging.Logger] = None,
        **kwargs: Any,
    ):
        super().__init__(
            info=info,
            raw=raw, 
            width=width,
            height=height,
            logger_obj=logger_obj,
            **kwargs,
        )
        
        # Initialize configuration and processors
        self.config = DrawThingsConfig()
        self.validator = DrawThingsValidator(self.config, self._logger)
        self.extractor = DrawThingsExtractor(self.config, self._logger)

    def _process(self) -> None:
        """Main processing pipeline for DrawThings format"""
        self._logger.debug(f"{self.tool}: Starting DrawThings format processing")
        
        # Validate input data
        if not self._raw:
            self._logger.debug(f"{self.tool}: No raw data provided")
            self.status = self.Status.MISSING_INFO
            self._error = "No raw data provided for DrawThings parsing"
            return
            
        # Parse and validate JSON structure
        parsed_data = self.validator.validate_json_structure(self._raw)
        if parsed_data is None:
            self.status = self.Status.FORMAT_DETECTION_ERROR
            self._error = "Raw data is not valid JSON"
            return
            
        # Validate DrawThings format
        if not self.validator.is_drawthings_format(parsed_data):
            self.status = self.Status.FORMAT_DETECTION_ERROR
            self._error = "JSON does not match DrawThings format"
            return
            
        # Extract data components
        success = self._extract_all_data(parsed_data)
        if not success:
            return  # Error already set by _extract_all_data
            
        self._logger.info(f"{self.tool}: Successfully parsed DrawThings metadata")

    def _extract_all_data(self, data: Dict[str, Any]) -> bool:
        """Extract all data components from validated DrawThings JSON"""
        try:
            # Extract prompts
            self._positive, self._negative = self.extractor.extract_prompts(data)
            
            # Extract and apply parameters
            parameters = self.extractor.extract_parameters(data)
            self._parameter.update(parameters)
            
            # Extract dimensions
            width, height = self.extractor.extract_dimensions(data)
            if width != "0":
                self._width = width
                self._parameter["width"] = width
            if height != "0":
                self._height = height
                self._parameter["height"] = height
            if width != "0" and height != "0":
                self._parameter["size"] = f"{width}x{height}"
                
            # Build settings string for remaining data
            self._build_settings_from_remaining_data(data)
            
            # Validate extraction success
            if not self._has_meaningful_data():
                self._logger.warning(f"{self.tool}: No meaningful data extracted")
                self.status = self.Status.FORMAT_ERROR
                self._error = "No meaningful data found in DrawThings JSON"
                return False
                
            return True
            
        except Exception as e:
            self._logger.error(f"{self.tool}: Unexpected error during extraction: {e}")
            self.status = self.Status.FORMAT_ERROR
            self._error = f"Extraction failed: {e}"
            return False

    def _build_settings_from_remaining_data(self, data: Dict[str, Any]) -> None:
        """Build settings string from data not handled by standard parameters"""
        # Determine which keys have been handled
        handled_keys = set(self.config.HANDLED_KEYS)
        handled_keys.update(self.config.PARAMETER_MAPPINGS.keys())
        
        # Add size to handled keys since we processed it for dimensions
        if "size" in data:
            handled_keys.add("size")
            
        # Build settings string from remaining data
        self._setting = self._build_settings_string(
            include_standard_params=False,  # Already in self._parameter
            custom_settings_dict=None,
            remaining_data_dict=data,
            remaining_handled_keys=handled_keys,
            sort_parts=True,
        )

    def _has_meaningful_data(self) -> bool:
        """Check if extraction yielded meaningful data"""
        has_prompts = bool(self._positive.strip())
        has_parameters = self._parameter_has_data()
        has_dimensions = self._width != "0" or self._height != "0"
        
        return has_prompts or has_parameters or has_dimensions

    def get_format_info(self) -> Dict[str, Any]:
        """Get detailed information about the parsed DrawThings data"""
        return {
            "format_name": self.tool,
            "has_positive_prompt": bool(self._positive),
            "has_negative_prompt": bool(self._negative),
            "parameter_count": len([v for v in self._parameter.values() if v and v != self.DEFAULT_PARAMETER_PLACEHOLDER]),
            "has_dimensions": self._width != "0" or self._height != "0",
            "dimensions": f"{self._width}x{self._height}" if self._width != "0" and self._height != "0" else None,
            "settings_items": len(self._setting.split(", ")) if self._setting else 0,
        }

    def validate_drawthings_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate DrawThings data structure and return validation results.
        Useful for debugging and format verification.
        """
        validation_result = {
            "is_valid": False,
            "has_required_keys": False,
            "has_characteristic_keys": False,
            "valid_size_format": None,
            "found_keys": list(data.keys()) if data else [],
            "issues": []
        }
        
        if not data:
            validation_result["issues"].append("No data provided")
            return validation_result
            
        # Check required keys
        required_found = [key for key in self.config.REQUIRED_KEYS if key in data]
        validation_result["has_required_keys"] = bool(required_found)
        validation_result["required_keys_found"] = required_found
        
        # Check characteristic keys
        characteristic_found = [key for key in self.config.CHARACTERISTIC_KEYS if key in data]
        validation_result["has_characteristic_keys"] = bool(characteristic_found)
        validation_result["characteristic_keys_found"] = characteristic_found
        
        # Validate size format if present
        if "size" in data:
            size_str = str(data["size"])
            validation_result["valid_size_format"] = self.validator._is_valid_size_format(size_str)
            if not validation_result["valid_size_format"]:
                validation_result["issues"].append(f"Invalid size format: {size_str}")
                
        # Overall validation
        validation_result["is_valid"] = (
            validation_result["has_required_keys"] and 
            validation_result["has_characteristic_keys"] and
            (validation_result["valid_size_format"] is not False)  # None or True
        )
        
        return validation_result