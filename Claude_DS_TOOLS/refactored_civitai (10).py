# dataset_tools/vendored_sdpr/format/ruinedfooocus.py

import json
import logging
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field

from .base_format import BaseFormat


@dataclass
class RuinedFooocusConfig:
    """Configuration for RuinedFooocus format parsing - systematically organized"""
    
    # Software identification
    SOFTWARE_IDENTIFIER: str = "RuinedFooocus"
    
    # Parameter mapping for standardization
    PARAMETER_MAPPINGS: Dict[str, Union[str, List[str]]] = field(default_factory=lambda: {
        "base_model_name": "model",
        "sampler_name": ["sampler_name", "sampler"],
        "seed": "seed",
        "cfg": ["cfg_scale", "cfg"],
        "steps": "steps",
        "scheduler": "scheduler",
        "base_model_hash": "model_hash",
        "loras": ["loras", "loras_str"],
        "start_step": "start_step",
        "denoise": "denoise",
        "width": "width",
        "height": "height",
    })
    
    # Core fields handled separately from parameters
    CORE_FIELDS: Set[str] = field(default_factory=lambda: {
        "Prompt", "Negative", "software", "width", "height"
    })
    
    # Fields that should be in custom settings if present
    CUSTOM_SETTINGS_FIELDS: Dict[str, str] = field(default_factory=lambda: {
        "scheduler": "Scheduler",
        "base_model_hash": "Model hash", 
        "loras": "Loras",
        "start_step": "Start step",
        "denoise": "Denoise",
        "refiner_model_name": "Refiner model",
        "refiner_switch": "Refiner switch",
        "inpaint_engine": "Inpaint engine",
        "control_lora": "Control LoRA",
        "style_selections": "Style selections",
    })
    
    # RuinedFooocus-specific feature keys
    RUINEDFOOOCUS_FEATURES: Set[str] = field(default_factory=lambda: {
        "mixing_image_prompt_and_vary_upscale",
        "mixing_image_prompt_and_inpaint", 
        "debugging_cn_preprocessor",
        "skipping_cn_preprocessor",
        "control_lora_canny",
        "controlnet_softness",
        "freeu_enabled",
        "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2",
    })


class RuinedFooocusValidator:
    """Validates RuinedFooocus format structure and content"""
    
    def __init__(self, config: RuinedFooocusConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def validate_ruinedfooocus_format(self, raw_data: str) -> Dict[str, Any]:
        """
        Validate RuinedFooocus format and return detailed analysis.
        Returns validation results with confidence scoring.
        """
        validation_result = {
            "is_ruinedfooocus": False,
            "confidence_score": 0.0,
            "json_valid": False,
            "has_software_tag": False,
            "software_value": None,
            "has_prompts": False,
            "has_parameters": False,
            "validation_errors": [],
        }
        
        if not raw_data:
            validation_result["validation_errors"].append("No raw data provided")
            return validation_result
            
        # Try to parse JSON
        try:
            data = json.loads(raw_data)
            validation_result["json_valid"] = True
            validation_result["confidence_score"] += 0.3
            
            if not isinstance(data, dict):
                validation_result["validation_errors"].append("JSON is not an object")
                return validation_result
                
        except json.JSONDecodeError as e:
            validation_result["validation_errors"].append(f"Invalid JSON: {e}")
            return validation_result
            
        # Check software identifier
        software_value = data.get("software")
        if software_value:
            validation_result["has_software_tag"] = True
            validation_result["software_value"] = str(software_value)
            
            if software_value == self.config.SOFTWARE_IDENTIFIER:
                validation_result["confidence_score"] += 0.6  # Strong indicator
            elif "fooocus" in str(software_value).lower():
                validation_result["confidence_score"] += 0.3  # Weak indicator
            else:
                validation_result["validation_errors"].append(f"Software tag mismatch: {software_value}")
                
        # Check for prompts
        has_prompts = bool(data.get("Prompt") or data.get("Negative"))
        validation_result["has_prompts"] = has_prompts
        if has_prompts:
            validation_result["confidence_score"] += 0.2
            
        # Check for generation parameters
        param_keys = set(data.keys()) & set(self.config.PARAMETER_MAPPINGS.keys())
        validation_result["has_parameters"] = bool(param_keys)
        if param_keys:
            validation_result["confidence_score"] += 0.2
            
        # Check for RuinedFooocus-specific features
        ruined_features = set(data.keys()) & self.config.RUINEDFOOOCUS_FEATURES
        if ruined_features:
            validation_result["confidence_score"] += 0.3
            validation_result["ruined_features_found"] = list(ruined_features)
            
        # Determine if this is definitively RuinedFooocus
        validation_result["is_ruinedfooocus"] = (
            validation_result["confidence_score"] >= 0.8 and
            validation_result["has_software_tag"] and
            software_value == self.config.SOFTWARE_IDENTIFIER
        )
        
        self.logger.debug(f"RuinedFooocus validation: confidence={validation_result['confidence_score']:.2f}, "
                         f"valid={validation_result['is_ruinedfooocus']}")
        
        return validation_result


class RuinedFooocusExtractor:
    """Handles extraction of data from validated RuinedFooocus JSON"""
    
    def __init__(self, config: RuinedFooocusConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def extract_ruinedfooocus_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract RuinedFooocus data from validated JSON.
        Returns structured extraction results.
        """
        result = {
            "positive": "",
            "negative": "",
            "parameters": {},
            "custom_settings": {},
            "handled_keys": set(),
            "extraction_errors": [],
        }
        
        try:
            # Extract prompts
            result["positive"] = str(data.get("Prompt", "")).strip()
            result["negative"] = str(data.get("Negative", "")).strip()
            result["handled_keys"].update(["Prompt", "Negative", "software"])
            
            # Extract standard parameters
            parameters, param_handled = self._extract_standard_parameters(data)
            result["parameters"] = parameters
            result["handled_keys"].update(param_handled)
            
            # Extract custom settings
            custom_settings, custom_handled = self._extract_custom_settings(data, result["handled_keys"])
            result["custom_settings"] = custom_settings
            result["handled_keys"].update(custom_handled)
            
            # Extract dimensions
            dimensions, dim_handled = self._extract_dimensions(data)
            result["parameters"].update(dimensions)
            result["handled_keys"].update(dim_handled)
            
            self.logger.debug(f"RuinedFooocus: Extracted {len(result['parameters'])} parameters, "
                             f"{len(result['custom_settings'])} custom settings")
            
        except Exception as e:
            self.logger.error(f"RuinedFooocus: Extraction error: {e}")
            result["extraction_errors"].append(f"Extraction failed: {e}")
            
        return result
        
    def _extract_standard_parameters(self, data: Dict[str, Any]) -> Tuple[Dict[str, str], Set[str]]:
        """Extract and standardize parameters"""
        parameters = {}
        handled_keys = set()
        
        for rf_key, standard_target in self.config.PARAMETER_MAPPINGS.items():
            if rf_key in data and data[rf_key] is not None:
                value = data[rf_key]
                processed_value = self._process_parameter_value(rf_key, value)
                
                if processed_value:
                    # Handle list targets (use first one)
                    if isinstance(standard_target, list):
                        target_key = standard_target[0] if standard_target else rf_key
                    else:
                        target_key = standard_target
                        
                    parameters[target_key] = processed_value
                    handled_keys.add(rf_key)
                    
        return parameters, handled_keys
        
    def _extract_custom_settings(self, data: Dict[str, Any], already_handled: Set[str]) -> Tuple[Dict[str, str], Set[str]]:
        """Extract custom settings for display"""
        custom_settings = {}
        handled_keys = set()
        
        # Process known custom settings fields
        for rf_key, display_name in self.config.CUSTOM_SETTINGS_FIELDS.items():
            if rf_key in data and rf_key not in already_handled:
                value = data[rf_key]
                if value is not None:
                    processed_value = self._process_parameter_value(rf_key, value)
                    if processed_value:
                        custom_settings[display_name] = processed_value
                        handled_keys.add(rf_key)
                        
        # Process any remaining RuinedFooocus-specific features
        for rf_key in self.config.RUINEDFOOOCUS_FEATURES:
            if rf_key in data and rf_key not in already_handled and rf_key not in handled_keys:
                value = data[rf_key]
                if value is not None:
                    processed_value = self._process_parameter_value(rf_key, value)
                    if processed_value:
                        display_name = self._format_key_for_display(rf_key)
                        custom_settings[display_name] = processed_value
                        handled_keys.add(rf_key)
                        
        return custom_settings, handled_keys
        
    def _extract_dimensions(self, data: Dict[str, Any]) -> Tuple[Dict[str, str], Set[str]]:
        """Extract width and height dimensions"""
        dimensions = {}
        handled_keys = set()
        
        for dim_key in ["width", "height"]:
            if dim_key in data and data[dim_key] is not None:
                try:
                    dim_value = str(int(data[dim_key]))
                    dimensions[dim_key] = dim_value
                    handled_keys.add(dim_key)
                except (ValueError, TypeError):
                    self.logger.debug(f"RuinedFooocus: Invalid {dim_key} value: {data[dim_key]}")
                    
        # Add size if both dimensions present
        if "width" in dimensions and "height" in dimensions:
            dimensions["size"] = f"{dimensions['width']}x{dimensions['height']}"
            
        return dimensions, handled_keys
        
    def _process_parameter_value(self, key: str, value: Any) -> str:
        """Process parameter values with key-specific logic"""
        if value is None:
            return ""
            
        # Handle boolean values
        if isinstance(value, bool):
            return str(value).lower()
            
        # Handle list values (like LoRAs)
        if isinstance(value, list):
            if key == "loras":
                return self._format_loras_list(value)
            else:
                return ", ".join(str(item) for item in value if item)
                
        # Handle dictionary values
        if isinstance(value, dict):
            return str(value)  # JSON representation for complex objects
            
        # Default string conversion
        return str(value).strip()
        
    def _format_loras_list(self, loras: List[Any]) -> str:
        """Format LoRAs list into readable string"""
        if not loras:
            return ""
            
        formatted_loras = []
        for lora in loras:
            if isinstance(lora, dict):
                name = lora.get("model_name", lora.get("name", "unknown"))
                weight = lora.get("weight", 1.0)
                formatted_loras.append(f"{name}:{weight}")
            else:
                formatted_loras.append(str(lora))
                
        return ", ".join(formatted_loras)
        
    def _format_key_for_display(self, key: str) -> str:
        """Format key name for display"""
        # Convert snake_case to Title Case
        return key.replace("_", " ").title()


class RuinedFooocusFormat(BaseFormat):
    """
    Enhanced RuinedFooocus format parser with comprehensive validation and extraction.
    
    RuinedFooocus is a fork of Fooocus with additional features and modifications.
    This parser handles the JSON metadata format with proper validation.
    """
    
    tool = "RuinedFooocus"
    
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
        self.config = RuinedFooocusConfig()
        self.validator = RuinedFooocusValidator(self.config, self._logger)
        self.extractor = RuinedFooocusExtractor(self.config, self._logger)
        
        # Store validation and extraction results
        self._validation_result: Optional[Dict[str, Any]] = None
        self._extraction_result: Optional[Dict[str, Any]] = None

    def _process(self) -> None:
        """Main processing pipeline for RuinedFooocus format"""
        self._logger.debug(f"{self.tool}: Starting RuinedFooocus format processing")
        
        # Validate input data
        if not self._raw:
            self._logger.warning(f"{self.tool}: No raw data provided")
            self.status = self.Status.MISSING_INFO
            self._error = "No raw data provided for RuinedFooocus parsing"
            return
            
        # Validate RuinedFooocus format
        self._validation_result = self.validator.validate_ruinedfooocus_format(self._raw)
        
        if not self._validation_result["is_ruinedfooocus"]:
            confidence = self._validation_result["confidence_score"]
            errors = self._validation_result["validation_errors"]
            self._logger.debug(f"{self.tool}: Not identified as RuinedFooocus (confidence: {confidence:.2f}, errors: {errors})")
            self.status = self.Status.FORMAT_DETECTION_ERROR
            self._error = f"Not RuinedFooocus format: {'; '.join(errors)}"
            return
            
        # Parse JSON data
        try:
            data = json.loads(self._raw)
            if not isinstance(data, dict):
                self.status = self.Status.FORMAT_ERROR
                self._error = "RuinedFooocus JSON is not an object"
                return
                
        except json.JSONDecodeError as e:
            self.status = self.Status.FORMAT_ERROR
            self._error = f"Invalid JSON in RuinedFooocus data: {e}"
            return
            
        # Extract data
        self._extraction_result = self.extractor.extract_ruinedfooocus_data(data)
        
        if self._extraction_result["extraction_errors"]:
            errors = self._extraction_result["extraction_errors"]
            self._logger.warning(f"{self.tool}: Extraction errors: {errors}")
            if not self._extraction_result["parameters"] and not self._extraction_result["positive"]:
                self.status = self.Status.FORMAT_ERROR
                self._error = f"RuinedFooocus extraction failed: {'; '.join(errors)}"
                return
                
        # Apply extraction results
        self._apply_extraction_results()
        
        # Validate extraction success
        if not self._has_meaningful_extraction():
            self._logger.warning(f"{self.tool}: No meaningful data extracted")
            self.status = self.Status.FORMAT_ERROR
            self._error = "RuinedFooocus parsing yielded no meaningful data"
            return
            
        self._logger.info(f"{self.tool}: Successfully parsed with {self._validation_result['confidence_score']:.2f} confidence")

    def _apply_extraction_results(self) -> None:
        """Apply extraction results to instance variables"""
        if not self._extraction_result:
            return
            
        # Apply prompts
        self._positive = self._extraction_result["positive"]
        self._negative = self._extraction_result["negative"]
        
        # Apply parameters
        parameters = self._extraction_result["parameters"]
        self._parameter.update(parameters)
        
        # Apply dimensions
        if "width" in parameters:
            self._width = parameters["width"]
        if "height" in parameters:
            self._height = parameters["height"]
            
        # Build settings string
        custom_settings = self._extraction_result["custom_settings"]
        self._setting = self._build_settings_string(
            custom_settings_dict=custom_settings,
            include_standard_params=True,
            sort_parts=True,
        )

    def _has_meaningful_extraction(self) -> bool:
        """Check if extraction yielded meaningful data"""
        has_prompts = bool(self._positive.strip())
        has_parameters = self._parameter_has_data()
        has_dimensions = self._width != "0" or self._height != "0"
        
        return has_prompts or has_parameters or has_dimensions

    def get_format_info(self) -> Dict[str, Any]:
        """Get detailed information about the parsed RuinedFooocus data"""
        return {
            "format_name": self.tool,
            "validation_result": self._validation_result,
            "has_positive_prompt": bool(self._positive),
            "has_negative_prompt": bool(self._negative),
            "parameter_count": len([v for v in self._parameter.values() if v and v != self.DEFAULT_PARAMETER_PLACEHOLDER]),
            "has_dimensions": self._width != "0" or self._height != "0",
            "dimensions": f"{self._width}x{self._height}" if self._width != "0" and self._height != "0" else None,
            "ruinedfooocus_features": self._analyze_ruinedfooocus_features(),
        }

    def _analyze_ruinedfooocus_features(self) -> Dict[str, Any]:
        """Analyze RuinedFooocus-specific features detected"""
        features = {
            "has_loras": False,
            "has_refiner": False,
            "has_inpainting": False,
            "has_controlnet": False,
            "has_freeu": False,
            "advanced_features": [],
            "feature_count": 0,
        }
        
        if not self._extraction_result:
            return features
            
        # Check custom settings for features
        custom_settings = self._extraction_result.get("custom_settings", {})
        
        # LoRA detection
        features["has_loras"] = "loras" in self._parameter or any("lora" in key.lower() for key in custom_settings)
        
        # Refiner detection
        features["has_refiner"] = any("refiner" in key.lower() for key in custom_settings)
        
        # Inpainting detection
        features["has_inpainting"] = any("inpaint" in key.lower() for key in custom_settings)
        
        # ControlNet detection
        features["has_controlnet"] = any("control" in key.lower() for key in custom_settings)
        
        # FreeU detection
        features["has_freeu"] = any("freeu" in key.lower() for key in custom_settings)
        
        # Build advanced features list
        if features["has_loras"]:
            features["advanced_features"].append("loras")
        if features["has_refiner"]:
            features["advanced_features"].append("refiner")
        if features["has_inpainting"]:
            features["advanced_features"].append("inpainting")
        if features["has_controlnet"]:
            features["advanced_features"].append("controlnet")
        if features["has_freeu"]:
            features["advanced_features"].append("freeu")
            
        features["feature_count"] = len(features["advanced_features"])
        
        return features

    def debug_ruinedfooocus_parsing(self) -> Dict[str, Any]:
        """Get comprehensive debugging information about RuinedFooocus parsing"""
        return {
            "input_data": {
                "has_raw": bool(self._raw),
                "raw_length": len(self._raw) if self._raw else 0,
                "raw_preview": self._raw[:200] if self._raw else None,
            },
            "validation_details": self._validation_result,
            "extraction_details": self._extraction_result,
            "parsing_summary": {
                "tool_name": self.tool,
                "parameter_count": len(self._parameter),
                "has_prompts": bool(self._positive or self._negative),
                "ruinedfooocus_specific_params": [
                    key for key in self._parameter.keys() 
                    if key in self.config.PARAMETER_MAPPINGS.values()
                ],
            },
            "config_info": {
                "software_identifier": self.config.SOFTWARE_IDENTIFIER,
                "parameter_mappings": len(self.config.PARAMETER_MAPPINGS),
                "custom_settings_fields": len(self.config.CUSTOM_SETTINGS_FIELDS),
                "ruinedfooocus_features": len(self.config.RUINEDFOOOCUS_FEATURES),
            }
        }