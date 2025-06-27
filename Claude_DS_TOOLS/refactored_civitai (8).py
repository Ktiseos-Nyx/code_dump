# dataset_tools/vendored_sdpr/format/mochi_diffusion.py

import re
import logging
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple
from dataclasses import dataclass, field

from .base_format import BaseFormat


@dataclass
class MochiDiffusionConfig:
    """Configuration for Mochi Diffusion format parsing - systematically organized"""
    
    # IPTC parameter mapping to standard names
    IPTC_PARAMETER_MAPPINGS: Dict[str, str] = field(default_factory=lambda: {
        "Guidance Scale": "cfg_scale",
        "Steps": "steps", 
        "Model": "model",
        "Seed": "seed",
        "Scheduler": "sampler_name",  # Mochi's scheduler is their sampler concept
        "Upscaler": "upscaler",
        "Date": "generation_date",
        "ML Compute Unit": "ml_compute_unit",
        "Generator": "generator_version",
        "Width": "width",
        "Height": "height",
        "Safety Checker": "safety_checker",
        "Reduce Memory Usage": "reduce_memory",
    })
    
    # Special handling keys (not included in standard parameter mapping)
    SPECIAL_HANDLING_KEYS: Set[str] = field(default_factory=lambda: {
        "Include in Image",    # Positive prompt
        "Exclude from Image",  # Negative prompt  
        "Size",               # Combined dimensions
        "Generator",          # Tool version info
    })
    
    # Mochi Diffusion identification patterns
    IDENTIFICATION_PATTERNS: Dict[str, Pattern[str]] = field(default_factory=lambda: {
        "originating_program": re.compile(r"Mochi\s+Diffusion", re.IGNORECASE),
        "generator_field": re.compile(r"Generator:\s*Mochi\s+Diffusion", re.IGNORECASE),
        "iptc_structure": re.compile(r"[^:]+:\s*[^;]+(?:;\s*[^:]+:\s*[^;]+)*", re.IGNORECASE),
    })
    
    # IPTC parsing patterns
    IPTC_PARSING_PATTERNS: Dict[str, Pattern[str]] = field(default_factory=lambda: {
        "key_value_pair": re.compile(r"([^:]+):\s*([^;]+)(?:;|$)"),
        "size_pattern": re.compile(r"(\d+)\s*[x×]\s*(\d+)", re.IGNORECASE),
        "version_pattern": re.compile(r"Mochi\s+Diffusion\s+([\d.]+)", re.IGNORECASE),
    })


class MochiDiffusionValidator:
    """Validates Mochi Diffusion format structure and content"""
    
    def __init__(self, config: MochiDiffusionConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def validate_mochi_structure(self, info_data: Dict[str, Any], raw_data: str) -> Dict[str, Any]:
        """
        Validate that the data structure matches Mochi Diffusion format.
        Returns detailed validation results.
        """
        validation_result = {
            "is_mochi": False,
            "confidence_score": 0.0,
            "identification_methods": [],
            "iptc_structure_valid": False,
            "originating_program_match": False,
            "generator_field_match": False,
            "parsed_key_count": 0,
        }
        
        # Check IPTC Originating Program
        originating_program = str(info_data.get("iptc_originating_program", "")).strip()
        if self.config.IDENTIFICATION_PATTERNS["originating_program"].search(originating_program):
            validation_result["originating_program_match"] = True
            validation_result["identification_methods"].append("iptc_originating_program")
            validation_result["confidence_score"] += 0.7
            
        # Check Generator field in raw data
        if raw_data and self.config.IDENTIFICATION_PATTERNS["generator_field"].search(raw_data):
            validation_result["generator_field_match"] = True
            validation_result["identification_methods"].append("generator_field")
            validation_result["confidence_score"] += 0.6
            
        # Validate IPTC structure
        if raw_data:
            structure_valid, key_count = self._validate_iptc_structure(raw_data)
            validation_result["iptc_structure_valid"] = structure_valid
            validation_result["parsed_key_count"] = key_count
            
            if structure_valid:
                validation_result["identification_methods"].append("iptc_structure")
                validation_result["confidence_score"] += 0.4
                
        # Determine if this is definitively Mochi
        validation_result["is_mochi"] = self._is_definitive_mochi(validation_result)
        
        self.logger.debug(f"Mochi validation: confidence={validation_result['confidence_score']:.2f}, "
                         f"methods={validation_result['identification_methods']}")
        
        return validation_result
        
    def _validate_iptc_structure(self, raw_data: str) -> Tuple[bool, int]:
        """Validate the IPTC structure and return (is_valid, key_count)"""
        if not raw_data:
            return False, 0
            
        # Clean the data
        cleaned_data = raw_data.replace("\n", " ").strip()
        
        # Check overall structure pattern
        if not self.config.IDENTIFICATION_PATTERNS["iptc_structure"].match(cleaned_data):
            return False, 0
            
        # Count valid key-value pairs
        key_value_matches = self.config.IPTC_PARSING_PATTERNS["key_value_pair"].findall(cleaned_data)
        valid_pairs = [(k.strip(), v.strip()) for k, v in key_value_matches if k.strip() and v.strip()]
        
        # Must have at least 2 valid pairs to be considered valid structure
        return len(valid_pairs) >= 2, len(valid_pairs)
        
    def _is_definitive_mochi(self, validation_result: Dict[str, Any]) -> bool:
        """Determine if we have definitive proof this is Mochi Diffusion"""
        # Strong identification: Originating Program + valid structure
        if (validation_result["originating_program_match"] and 
            validation_result["iptc_structure_valid"]):
            return True
            
        # Medium identification: Generator field + multiple keys
        if (validation_result["generator_field_match"] and 
            validation_result["parsed_key_count"] >= 3):
            return True
            
        # High confidence with multiple methods
        if (validation_result["confidence_score"] >= 0.8 and 
            len(validation_result["identification_methods"]) >= 2):
            return True
            
        return False


class MochiDiffusionParser:
    """Handles parsing of Mochi Diffusion IPTC metadata"""
    
    def __init__(self, config: MochiDiffusionConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def parse_iptc_metadata(self, raw_data: str) -> Dict[str, Any]:
        """
        Parse Mochi Diffusion IPTC metadata string.
        Returns structured extraction results.
        """
        if not raw_data:
            return self._empty_parse_result()
            
        result = {
            "positive": "",
            "negative": "",
            "parameters": {},
            "tool_info": {},
            "parsed_pairs": {},
            "handled_keys": set(),
            "parse_errors": [],
        }
        
        try:
            # Clean and normalize the data
            cleaned_data = self._clean_iptc_data(raw_data)
            
            # Parse key-value pairs
            parsed_pairs = self._parse_key_value_pairs(cleaned_data)
            result["parsed_pairs"] = parsed_pairs
            
            if not parsed_pairs:
                result["parse_errors"].append("No valid key-value pairs found")
                return result
                
            # Extract prompts
            result["positive"] = parsed_pairs.get("Include in Image", "").strip()
            result["negative"] = parsed_pairs.get("Exclude from Image", "").strip()
            result["handled_keys"].update(["Include in Image", "Exclude from Image"])
            
            # Extract tool information
            result["tool_info"] = self._extract_tool_information(parsed_pairs)
            result["handled_keys"].add("Generator")
            
            # Extract and standardize parameters
            parameters = self._extract_parameters(parsed_pairs, result["handled_keys"])
            result["parameters"] = parameters
            
            # Handle dimensions
            dimensions = self._extract_dimensions(parsed_pairs)
            if dimensions:
                result["parameters"].update(dimensions)
                result["handled_keys"].add("Size")
                
            self.logger.debug(f"Mochi: Parsed {len(parsed_pairs)} pairs, "
                             f"extracted {len(result['parameters'])} parameters")
            
        except Exception as e:
            self.logger.error(f"Mochi: Error during IPTC parsing: {e}")
            result["parse_errors"].append(f"Parsing exception: {e}")
            
        return result
        
    def _clean_iptc_data(self, raw_data: str) -> str:
        """Clean and normalize IPTC data for parsing"""
        # Replace newlines with spaces
        cleaned = raw_data.replace("\n", " ").replace("\r", " ")
        
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        
        # Ensure proper semicolon separation
        cleaned = re.sub(r";\s*;", ";", cleaned)  # Remove double semicolons
        cleaned = cleaned.rstrip(";")  # Remove trailing semicolon
        
        return cleaned
        
    def _parse_key_value_pairs(self, data: str) -> Dict[str, str]:
        """Parse key-value pairs from cleaned IPTC data"""
        pairs = {}
        
        # Use regex to find all key-value pairs
        matches = self.config.IPTC_PARSING_PATTERNS["key_value_pair"].findall(data)
        
        for key, value in matches:
            key = key.strip()
            value = value.strip()
            
            if key and value:
                pairs[key] = value
            else:
                self.logger.debug(f"Mochi: Skipping empty key-value pair: '{key}': '{value}'")
                
        return pairs
        
    def _extract_tool_information(self, parsed_pairs: Dict[str, str]) -> Dict[str, str]:
        """Extract tool version and related information"""
        tool_info = {}
        
        generator = parsed_pairs.get("Generator", "")
        if generator:
            tool_info["full_name"] = generator
            
            # Extract version number
            version_match = self.config.IPTC_PARSING_PATTERNS["version_pattern"].search(generator)
            if version_match:
                tool_info["version"] = version_match.group(1)
            else:
                tool_info["version"] = "unknown"
                
        return tool_info
        
    def _extract_parameters(self, parsed_pairs: Dict[str, str], handled_keys: Set[str]) -> Dict[str, str]:
        """Extract and standardize parameters"""
        parameters = {}
        
        for iptc_key, standard_key in self.config.IPTC_PARAMETER_MAPPINGS.items():
            if iptc_key in parsed_pairs:
                value = parsed_pairs[iptc_key]
                processed_value = self._process_parameter_value(iptc_key, value)
                
                if processed_value:
                    parameters[standard_key] = processed_value
                    handled_keys.add(iptc_key)
                    
        return parameters
        
    def _process_parameter_value(self, key: str, value: str) -> str:
        """Process parameter values with key-specific logic"""
        if not value:
            return ""
            
        # Handle boolean-like values
        if value.lower() in ["true", "false", "yes", "no", "on", "off"]:
            return value.lower()
            
        # Handle numeric values
        if key in ["Steps", "Guidance Scale", "Seed"]:
            # Ensure numeric values are clean
            cleaned = re.sub(r"[^\d.-]", "", value)
            return cleaned if cleaned else value
            
        # Default: return cleaned string
        return value.strip()
        
    def _extract_dimensions(self, parsed_pairs: Dict[str, str]) -> Dict[str, str]:
        """Extract width and height from Size field or individual fields"""
        dimensions = {}
        
        # Try Size field first (e.g., "512x768")
        size_value = parsed_pairs.get("Size", "")
        if size_value:
            size_match = self.config.IPTC_PARSING_PATTERNS["size_pattern"].search(size_value)
            if size_match:
                width = size_match.group(1)
                height = size_match.group(2)
                dimensions["width"] = width
                dimensions["height"] = height
                dimensions["size"] = f"{width}x{height}"
                return dimensions
                
        # Try individual Width/Height fields
        width = parsed_pairs.get("Width", "")
        height = parsed_pairs.get("Height", "")
        
        if width and height:
            # Clean numeric values
            width_clean = re.sub(r"[^\d]", "", width)
            height_clean = re.sub(r"[^\d]", "", height)
            
            if width_clean and height_clean:
                dimensions["width"] = width_clean
                dimensions["height"] = height_clean
                dimensions["size"] = f"{width_clean}x{height_clean}"
                
        return dimensions
        
    def _empty_parse_result(self) -> Dict[str, Any]:
        """Return empty parse result structure"""
        return {
            "positive": "",
            "negative": "",
            "parameters": {},
            "tool_info": {},
            "parsed_pairs": {},
            "handled_keys": set(),
            "parse_errors": ["No data to parse"],
        }


class MochiDiffusionFormat(BaseFormat):
    """
    Enhanced Mochi Diffusion format parser with robust IPTC metadata handling.
    
    Mochi Diffusion stores metadata in IPTC fields with a specific semicolon-separated
    key-value format. This parser handles identification via IPTC Originating Program
    and comprehensive parsing of generation parameters.
    """
    
    tool = "Mochi Diffusion"
    
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
        
        # Initialize components
        self.config = MochiDiffusionConfig()
        self.validator = MochiDiffusionValidator(self.config, self._logger)
        self.parser = MochiDiffusionParser(self.config, self._logger)
        
        # Store validation and parsing results
        self._validation_result: Optional[Dict[str, Any]] = None
        self._parse_result: Optional[Dict[str, Any]] = None

    def _process(self) -> None:
        """Main processing pipeline for Mochi Diffusion format"""
        self._logger.debug(f"{self.tool}: Starting Mochi Diffusion format processing")
        
        # Validate input data
        if not self._info and not self._raw:
            self._logger.debug(f"{self.tool}: No info or raw data provided")
            self.status = self.Status.MISSING_INFO
            self._error = "No metadata provided for Mochi Diffusion analysis"
            return
            
        # Validate Mochi Diffusion format
        self._validation_result = self.validator.validate_mochi_structure(self._info or {}, self._raw)
        
        if not self._validation_result["is_mochi"]:
            confidence = self._validation_result["confidence_score"]
            methods = self._validation_result["identification_methods"]
            self._logger.debug(f"{self.tool}: Not identified as Mochi (confidence: {confidence:.2f}, methods: {methods})")
            self.status = self.Status.FORMAT_DETECTION_ERROR
            self._error = "No definitive Mochi Diffusion signatures found"
            return
            
        # Parse IPTC metadata
        self._parse_result = self.parser.parse_iptc_metadata(self._raw)
        
        if self._parse_result["parse_errors"]:
            self._logger.warning(f"{self.tool}: Parse errors: {self._parse_result['parse_errors']}")
            if not self._parse_result["parsed_pairs"]:
                self.status = self.Status.FORMAT_ERROR
                self._error = f"IPTC parsing failed: {'; '.join(self._parse_result['parse_errors'])}"
                return
                
        # Apply parsed data
        self._apply_parse_results()
        
        # Validate extraction success
        if not self._has_meaningful_extraction():
            self._logger.warning(f"{self.tool}: No meaningful data extracted")
            self.status = self.Status.FORMAT_ERROR
            self._error = "Mochi Diffusion parsing yielded no meaningful data"
            return
            
        self._logger.info(f"{self.tool}: Successfully parsed with {self._validation_result['confidence_score']:.2f} confidence")

    def _apply_parse_results(self) -> None:
        """Apply parsing results to instance variables"""
        if not self._parse_result:
            return
            
        # Apply prompts
        self._positive = self._parse_result["positive"]
        self._negative = self._parse_result["negative"]
        
        # Apply parameters
        parameters = self._parse_result["parameters"]
        self._parameter.update(parameters)
        
        # Update tool name with version info
        tool_info = self._parse_result["tool_info"]
        if tool_info.get("full_name"):
            self.tool = tool_info["full_name"]
        elif tool_info.get("version"):
            self.tool = f"Mochi Diffusion {tool_info['version']}"
            
        # Apply dimensions
        if "width" in parameters:
            self._width = parameters["width"]
        if "height" in parameters:
            self._height = parameters["height"]
            
        # Build settings string from unhandled pairs
        self._build_mochi_settings()
        
        # Add metadata about detection
        self._parameter["mochi_confidence"] = f"{self._validation_result['confidence_score']:.2f}"
        self._parameter["detection_methods"] = str(len(self._validation_result['identification_methods']))

    def _build_mochi_settings(self) -> None:
        """Build settings string from unhandled IPTC pairs"""
        if not self._parse_result:
            return
            
        parsed_pairs = self._parse_result["parsed_pairs"]
        handled_keys = self._parse_result["handled_keys"]
        
        # Build settings from remaining unhandled keys
        self._setting = self._build_settings_string(
            remaining_data_dict=parsed_pairs,
            remaining_handled_keys=handled_keys,
            kv_separator=": ",
            pair_separator="; ",
            sort_parts=True,
        )

    def _has_meaningful_extraction(self) -> bool:
        """Check if extraction yielded meaningful data"""
        has_prompts = bool(self._positive.strip())
        has_parameters = self._parameter_has_data()
        has_dimensions = self._width != "0" or self._height != "0"
        
        return has_prompts or has_parameters or has_dimensions

    def get_format_info(self) -> Dict[str, Any]:
        """Get detailed information about the parsed Mochi Diffusion data"""
        return {
            "format_name": self.tool,
            "validation_result": self._validation_result,
            "has_positive_prompt": bool(self._positive),
            "has_negative_prompt": bool(self._negative),
            "parameter_count": len([v for v in self._parameter.values() if v and v != self.DEFAULT_PARAMETER_PLACEHOLDER]),
            "has_dimensions": self._width != "0" or self._height != "0",
            "dimensions": f"{self._width}x{self._height}" if self._width != "0" and self._height != "0" else None,
            "mochi_features": self._analyze_mochi_features(),
        }

    def _analyze_mochi_features(self) -> Dict[str, Any]:
        """Analyze Mochi Diffusion-specific features detected"""
        features = {
            "has_ml_compute_info": False,
            "has_safety_checker": False,
            "has_upscaler": False,
            "has_memory_optimization": False,
            "feature_count": 0,
        }
        
        # Check for Mochi-specific features
        features["has_ml_compute_info"] = "ml_compute_unit" in self._parameter
        features["has_safety_checker"] = "safety_checker" in self._parameter
        features["has_upscaler"] = "upscaler" in self._parameter
        features["has_memory_optimization"] = "reduce_memory" in self._parameter
        
        # Count active features
        features["feature_count"] = sum([
            features["has_ml_compute_info"],
            features["has_safety_checker"],
            features["has_upscaler"],
            features["has_memory_optimization"],
        ])
        
        return features

    def debug_mochi_parsing(self) -> Dict[str, Any]:
        """Get comprehensive debugging information about Mochi parsing"""
        debug_info = {
            "input_data": {
                "has_info": bool(self._info),
                "info_keys": list(self._info.keys()) if self._info else [],
                "has_raw": bool(self._raw),
                "raw_length": len(self._raw) if self._raw else 0,
                "raw_preview": self._raw[:200] if self._raw else None,
            },
            "validation_details": self._validation_result,
            "parsing_details": self._parse_result,
            "extraction_summary": {
                "tool_name": self.tool,
                "parameter_count": len(self._parameter),
                "has_prompts": bool(self._positive or self._negative),
                "mochi_specific_params": [
                    key for key in self._parameter.keys() 
                    if key in self.config.IPTC_PARAMETER_MAPPINGS.values()
                ],
            },
            "config_info": {
                "total_parameter_mappings": len(self.config.IPTC_PARAMETER_MAPPINGS),
                "special_handling_keys": list(self.config.SPECIAL_HANDLING_KEYS),
                "identification_patterns": list(self.config.IDENTIFICATION_PATTERNS.keys()),
            }
        }
        
        return debug_info