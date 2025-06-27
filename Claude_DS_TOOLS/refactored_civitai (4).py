# dataset_tools/vendored_sdpr/format/fooocus.py

__author__ = "receyuki"
__filename__ = "fooocus.py"
# MODIFIED by Ktiseos Nyx for Dataset-Tools
__copyright__ = "Copyright 2023, Receyuki; Modified 2025, Ktiseos Nyx"
__email__ = "receyuki@gmail.com"

import logging
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field

from .base_format import BaseFormat


@dataclass
class FooocusConfig:
    """Configuration for Fooocus format parsing - systematically organized"""
    
    # Parameter mapping for standardization
    PARAMETER_MAPPINGS: Dict[str, Union[str, List[str]]] = field(default_factory=lambda: {
        "sampler_name": "sampler_name",
        "seed": "seed", 
        "guidance_scale": "cfg_scale",
        "steps": "steps",
        "base_model_name": "model",
        "base_model_hash": "model_hash",
        "lora_loras": "loras",
        "scheduler": "scheduler",
        "sharpness": "sharpness",
        "adm_scaler_positive": "adm_scaler_positive",
        "adm_scaler_negative": "adm_scaler_negative",
        "adm_scaler_end": "adm_scaler_end",
        "refiner_model_name": "refiner_model",
        "refiner_switch": "refiner_switch",
        "inpaint_engine": "inpaint_engine",
        "style_selections": "styles",
    })
    
    # Prompt key variations (Fooocus is generally consistent, but let's be safe)
    PROMPT_KEYS: Dict[str, Set[str]] = field(default_factory=lambda: {
        "positive": {"prompt", "positive_prompt"},
        "negative": {"negative_prompt", "negative"}
    })
    
    # Dimension keys that Fooocus uses
    DIMENSION_KEYS: Set[str] = field(default_factory=lambda: {
        "width", "height", "aspect_ratio"
    })
    
    # Keys that should always be handled and not included in settings
    CORE_HANDLED_KEYS: Set[str] = field(default_factory=lambda: {
        "prompt", "positive_prompt", "negative_prompt", "negative", 
        "width", "height", "aspect_ratio"
    })
    
    # Fooocus-specific keys that identify the format
    FOOOCUS_IDENTIFIER_KEYS: Set[str] = field(default_factory=lambda: {
        "base_model_name", "base_model_hash", "lora_loras", 
        "adm_scaler_positive", "inpaint_engine", "style_selections"
    })


class FooocusValidator:
    """Validates Fooocus format structure and content"""
    
    def __init__(self, config: FooocusConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def validate_fooocus_structure(self, data: Any) -> bool:
        """Validate that the data structure matches Fooocus format"""
        if not isinstance(data, dict):
            self.logger.debug("Fooocus: Data is not a dictionary")
            return False
            
        if not data:
            self.logger.debug("Fooocus: Data dictionary is empty")
            return False
            
        # Check for Fooocus-specific identifier keys
        has_fooocus_identifiers = any(
            key in data for key in self.config.FOOOCUS_IDENTIFIER_KEYS
        )
        
        # Check for basic generation parameters
        has_basic_params = any(
            key in data for key in ["prompt", "sampler_name", "seed", "steps"]
        )
        
        # Must have either Fooocus-specific identifiers or basic params
        is_valid = has_fooocus_identifiers or has_basic_params
        
        if not is_valid:
            self.logger.debug("Fooocus: No Fooocus identifiers or basic parameters found")
            
        return is_valid
        
    def check_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality and completeness of Fooocus data"""
        quality_info = {
            "has_prompts": False,
            "has_model_info": False,
            "has_generation_params": False,
            "has_dimensions": False,
            "has_advanced_features": False,
            "completeness_score": 0,
        }
        
        # Check for prompts
        prompt_keys = set(data.keys()) & {key for key_set in self.config.PROMPT_KEYS.values() for key in key_set}
        quality_info["has_prompts"] = bool(prompt_keys)
        
        # Check for model information
        model_keys = {"base_model_name", "base_model_hash", "refiner_model_name"}
        quality_info["has_model_info"] = bool(set(data.keys()) & model_keys)
        
        # Check for generation parameters
        gen_param_keys = {"sampler_name", "seed", "steps", "guidance_scale", "scheduler"}
        quality_info["has_generation_params"] = bool(set(data.keys()) & gen_param_keys)
        
        # Check for dimensions
        quality_info["has_dimensions"] = bool(set(data.keys()) & self.config.DIMENSION_KEYS)
        
        # Check for advanced Fooocus features
        advanced_keys = {"lora_loras", "style_selections", "inpaint_engine", "adm_scaler_positive"}
        quality_info["has_advanced_features"] = bool(set(data.keys()) & advanced_keys)
        
        # Calculate completeness score
        quality_info["completeness_score"] = sum([
            quality_info["has_prompts"],
            quality_info["has_model_info"], 
            quality_info["has_generation_params"],
            quality_info["has_dimensions"],
            quality_info["has_advanced_features"]
        ])
        
        return quality_info


class FooocusExtractor:
    """Handles extraction of data from validated Fooocus dictionaries"""
    
    def __init__(self, config: FooocusConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def extract_prompts(self, data: Dict[str, Any]) -> tuple[str, str]:
        """Extract positive and negative prompts"""
        positive = self._extract_by_key_variations(data, self.config.PROMPT_KEYS["positive"])
        negative = self._extract_by_key_variations(data, self.config.PROMPT_KEYS["negative"])
        
        self.logger.debug(f"Fooocus: Extracted prompts - positive: {len(positive)} chars, negative: {len(negative)} chars")
        return positive, negative
        
    def _extract_by_key_variations(self, data: Dict[str, Any], key_variations: Set[str]) -> str:
        """Extract value using multiple possible key names"""
        for key in key_variations:
            if key in data and data[key] is not None:
                return str(data[key]).strip()
        return ""
        
    def extract_parameters(self, data: Dict[str, Any]) -> tuple[Dict[str, str], Set[str]]:
        """Extract and standardize parameters, returning (parameters, handled_keys)"""
        parameters = {}
        handled_keys = set()
        
        for fooocus_key, canonical_target in self.config.PARAMETER_MAPPINGS.items():
            if fooocus_key not in data or data[fooocus_key] is None:
                continue
                
            # Process the value
            raw_value = data[fooocus_key]
            processed_value = self._process_parameter_value(fooocus_key, raw_value)
            
            if not processed_value:
                continue
                
            # Store in parameters
            if isinstance(canonical_target, list):
                # Use first target key if multiple options
                if canonical_target:
                    parameters[canonical_target[0]] = processed_value
            else:
                parameters[canonical_target] = processed_value
                
            handled_keys.add(fooocus_key)
            
        self.logger.debug(f"Fooocus: Extracted {len(parameters)} parameters")
        return parameters, handled_keys
        
    def _process_parameter_value(self, key: str, value: Any) -> str:
        """Process parameter values with key-specific logic"""
        if value is None:
            return ""
            
        # Handle special cases
        if key == "lora_loras" and isinstance(value, list):
            # Convert LoRA list to string representation
            return self._format_lora_list(value)
        elif key == "style_selections" and isinstance(value, list):
            # Convert style list to comma-separated string
            return ", ".join(str(item) for item in value if item)
        else:
            # Standard string conversion with cleaning
            return self._clean_parameter_value(value)
            
    def _format_lora_list(self, lora_list: List[Any]) -> str:
        """Format LoRA list into readable string"""
        if not lora_list:
            return ""
            
        formatted_loras = []
        for lora in lora_list:
            if isinstance(lora, dict):
                name = lora.get("model_name", "unknown")
                weight = lora.get("weight", 1.0)
                formatted_loras.append(f"{name}:{weight}")
            else:
                formatted_loras.append(str(lora))
                
        return ", ".join(formatted_loras)
        
    def _clean_parameter_value(self, value: Any) -> str:
        """Clean and standardize parameter values"""
        if value is None:
            return ""
            
        cleaned = str(value).strip()
        
        # Remove surrounding quotes
        if len(cleaned) >= 2:
            if (cleaned.startswith('"') and cleaned.endswith('"')) or \
               (cleaned.startswith("'") and cleaned.endswith("'")):
                cleaned = cleaned[1:-1]
                
        return cleaned
        
    def extract_dimensions(self, data: Dict[str, Any]) -> tuple[str, str, Set[str]]:
        """Extract width and height, returning (width, height, handled_keys)"""
        width = "0"
        height = "0" 
        handled_keys = set()
        
        # Direct width/height extraction
        if "width" in data and data["width"] is not None:
            try:
                width = str(int(data["width"]))
                handled_keys.add("width")
            except (ValueError, TypeError):
                pass
                
        if "height" in data and data["height"] is not None:
            try:
                height = str(int(data["height"]))
                handled_keys.add("height")
            except (ValueError, TypeError):
                pass
                
        # Handle aspect ratio if no direct dimensions
        if width == "0" and height == "0" and "aspect_ratio" in data:
            width, height = self._parse_aspect_ratio(data["aspect_ratio"])
            if width != "0" or height != "0":
                handled_keys.add("aspect_ratio")
                
        self.logger.debug(f"Fooocus: Extracted dimensions - {width}x{height}")
        return width, height, handled_keys
        
    def _parse_aspect_ratio(self, aspect_ratio: Any) -> tuple[str, str]:
        """Parse aspect ratio string into approximate dimensions"""
        if not aspect_ratio:
            return "0", "0"
            
        ratio_str = str(aspect_ratio).strip()
        
        # Common aspect ratios mapping
        ratio_map = {
            "1:1": ("512", "512"),
            "4:3": ("512", "384"), 
            "3:4": ("384", "512"),
            "16:9": ("512", "288"),
            "9:16": ("288", "512"),
            "3:2": ("512", "341"),
            "2:3": ("341", "512"),
        }
        
        if ratio_str in ratio_map:
            return ratio_map[ratio_str]
            
        # Try to parse custom ratio
        if ":" in ratio_str:
            try:
                w_ratio, h_ratio = ratio_str.split(":", 1)
                w_ratio = float(w_ratio.strip())
                h_ratio = float(h_ratio.strip())
                
                # Scale to reasonable dimensions (base 512)
                if w_ratio >= h_ratio:
                    width = "512"
                    height = str(int(512 * h_ratio / w_ratio))
                else:
                    height = "512"
                    width = str(int(512 * w_ratio / h_ratio))
                    
                return width, height
            except (ValueError, ZeroDivisionError):
                pass
                
        return "0", "0"


class Fooocus(BaseFormat):
    """
    Enhanced Fooocus format parser with robust validation and comprehensive extraction.
    
    Fooocus stores metadata as structured dictionaries with specific parameter names.
    This parser handles all Fooocus-specific features while maintaining compatibility.
    """
    
    tool = "Fooocus"
    
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
        self.config = FooocusConfig()
        self.validator = FooocusValidator(self.config, self._logger)
        self.extractor = FooocusExtractor(self.config, self._logger)

    def _process(self) -> None:
        """Main processing pipeline for Fooocus format"""
        self._logger.debug(f"{self.tool}: Starting Fooocus format processing")
        
        # Validate input data structure
        if not self._validate_input_data():
            return
            
        # Use info dict as data source (Fooocus standard)
        data = self._info
        
        # Validate Fooocus format
        if not self.validator.validate_fooocus_structure(data):
            self._logger.debug(f"{self.tool}: Data does not match Fooocus format")
            self.status = self.Status.FORMAT_DETECTION_ERROR
            self._error = "Data structure does not match Fooocus format"
            return
            
        # Extract all data components
        success = self._extract_all_components(data)
        if not success:
            return  # Error already set
            
        # Build settings string from remaining data
        self._build_comprehensive_settings(data)
        
        # Set raw data if needed
        self._set_raw_from_info_if_empty()
        
        # Validate extraction success
        if not self._has_meaningful_extraction():
            self._logger.warning(f"{self.tool}: No meaningful data extracted")
            self.status = self.Status.FORMAT_ERROR
            self._error = "Fooocus parsing yielded no meaningful data"
            return
            
        self._logger.info(f"{self.tool}: Successfully parsed Fooocus metadata")

    def _validate_input_data(self) -> bool:
        """Validate that we have usable input data"""
        if not self._info:
            self._logger.warning(f"{self.tool}: No info data provided")
            self.status = self.Status.MISSING_INFO
            self._error = "No Fooocus metadata provided"
            return False
            
        if not isinstance(self._info, dict):
            self._logger.warning(f"{self.tool}: Info data is not a dictionary")
            self.status = self.Status.FORMAT_ERROR
            self._error = "Fooocus metadata is not a dictionary"
            return False
            
        return True

    def _extract_all_components(self, data: Dict[str, Any]) -> bool:
        """Extract all data components from validated Fooocus dictionary"""
        try:
            # Track all handled keys
            all_handled_keys = set(self.config.CORE_HANDLED_KEYS)
            
            # Extract prompts
            self._positive, self._negative = self.extractor.extract_prompts(data)
            
            # Extract parameters
            parameters, param_handled_keys = self.extractor.extract_parameters(data)
            self._parameter.update(parameters)
            all_handled_keys.update(param_handled_keys)
            
            # Extract dimensions
            width, height, dim_handled_keys = self.extractor.extract_dimensions(data)
            all_handled_keys.update(dim_handled_keys)
            
            # Apply dimensions
            if width != "0":
                self._width = width
                self._parameter["width"] = width
            if height != "0":
                self._height = height
                self._parameter["height"] = height
            if width != "0" and height != "0":
                self._parameter["size"] = f"{width}x{height}"
                
            # Store handled keys for settings building
            self._handled_keys = all_handled_keys
            
            return True
            
        except Exception as e:
            self._logger.error(f"{self.tool}: Unexpected error during extraction: {e}")
            self.status = self.Status.FORMAT_ERROR
            self._error = f"Fooocus extraction failed: {e}"
            return False

    def _build_comprehensive_settings(self, data: Dict[str, Any]) -> None:
        """Build settings string from all remaining unhandled data"""
        self._setting = self._build_settings_string(
            include_standard_params=False,  # Already in self._parameter
            custom_settings_dict=None,
            remaining_data_dict=data,
            remaining_handled_keys=self._handled_keys,
            sort_parts=True,
        )

    def _has_meaningful_extraction(self) -> bool:
        """Check if we extracted meaningful data"""
        has_prompts = bool(self._positive.strip())
        has_parameters = self._parameter_has_data()
        has_dimensions = self._width != "0" or self._height != "0"
        
        return has_prompts or has_parameters or has_dimensions

    def get_format_info(self) -> Dict[str, Any]:
        """Get detailed information about the parsed Fooocus data"""
        return {
            "format_name": self.tool,
            "has_positive_prompt": bool(self._positive),
            "has_negative_prompt": bool(self._negative),
            "parameter_count": len([v for v in self._parameter.values() if v and v != self.DEFAULT_PARAMETER_PLACEHOLDER]),
            "has_dimensions": self._width != "0" or self._height != "0",
            "dimensions": f"{self._width}x{self._height}" if self._width != "0" and self._height != "0" else None,
            "settings_items": len(self._setting.split(", ")) if self._setting else 0,
            "fooocus_features": self._get_fooocus_features(),
        }

    def _get_fooocus_features(self) -> Dict[str, bool]:
        """Get information about Fooocus-specific features detected"""
        return {
            "has_loras": "loras" in self._parameter,
            "has_styles": "styles" in self._parameter,
            "has_refiner": "refiner_model" in self._parameter,
            "has_inpainting": "inpaint_engine" in self._parameter,
            "has_adm_scalers": any(key.startswith("adm_scaler") for key in self._parameter),
        }

    def analyze_fooocus_data(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of Fooocus data structure and content.
        Useful for debugging and format verification.
        """
        analysis_data = data or self._info or {}
        
        analysis = {
            "is_valid_fooocus": False,
            "data_quality": {},
            "detected_features": {},
            "parameter_analysis": {},
            "recommendations": [],
        }
        
        if not analysis_data:
            analysis["recommendations"].append("No data provided for analysis")
            return analysis
            
        # Validate structure
        analysis["is_valid_fooocus"] = self.validator.validate_fooocus_structure(analysis_data)
        
        # Assess data quality
        analysis["data_quality"] = self.validator.check_data_quality(analysis_data)
        
        # Detect Fooocus features
        fooocus_feature_keys = set(analysis_data.keys()) & self.config.FOOOCUS_IDENTIFIER_KEYS
        analysis["detected_features"] = {
            "fooocus_specific_keys": list(fooocus_feature_keys),
            "has_advanced_features": bool(fooocus_feature_keys),
            "total_keys": len(analysis_data),
        }
        
        # Parameter analysis
        param_keys = set(analysis_data.keys()) & set(self.config.PARAMETER_MAPPINGS.keys())
        analysis["parameter_analysis"] = {
            "recognized_parameters": list(param_keys),
            "parameter_coverage": len(param_keys) / len(self.config.PARAMETER_MAPPINGS),
            "has_core_params": bool(param_keys & {"seed", "steps", "sampler_name"}),
        }
        
        # Generate recommendations
        if analysis["data_quality"]["completeness_score"] < 3:
            analysis["recommendations"].append("Consider including more metadata for better compatibility")
        if not analysis["detected_features"]["has_advanced_features"]:
            analysis["recommendations"].append("No Fooocus-specific features detected")
        if not analysis["parameter_analysis"]["has_core_params"]:
            analysis["recommendations"].append("Missing core generation parameters")
            
        return analysis