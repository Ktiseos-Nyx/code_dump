# dataset_tools/vendored_sdpr/format/swarmui.py

__author__ = "receyuki"
__filename__ = "swarmui.py"
# MODIFIED by Ktiseos Nyx for Dataset-Tools
__copyright__ = "Copyright 2023, Receyuki; Modified 2025, Ktiseos Nyx"
__email__ = "receyuki@gmail.com"

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

from .base_format import BaseFormat


@dataclass
class SwarmUIConfig:
    """Configuration for SwarmUI format parsing - systematically organized"""
    
    # Parameter mapping for SwarmUI to standard names
    PARAMETER_MAPPINGS: Dict[str, Union[str, List[str]]] = field(default_factory=lambda: {
        "model": "model",
        "seed": "seed",
        "cfgscale": "cfg_scale",
        "steps": "steps",
        "width": "width",
        "height": "height",
        "batchsize": "batch_size",
        "refinermultiplier": "refiner_multiplier",
        "refinermodel": "refiner_model",
        "refinerstart": "refiner_start",
        "refinersteps": "refiner_steps",
        "vae": "vae_model",
        "clipskip": "clip_skip",
        "loras": "loras",
        "embeddings": "embeddings",
        "controlnets": "controlnets",
    })
    
    # Sampler field variations in SwarmUI
    SAMPLER_FIELDS: List[str] = field(default_factory=lambda: [
        "comfyuisampler", "autowebuisampler", "sampler", "scheduler"
    ])
    
    # Prompt field variations
    PROMPT_FIELDS: Dict[str, List[str]] = field(default_factory=lambda: {
        "positive": ["prompt", "positive_prompt", "positiveprompt"],
        "negative": ["negativeprompt", "negative_prompt", "negprompt"]
    })
    
    # SwarmUI-specific data structure keys
    SWARMUI_STRUCTURE_KEYS: Set[str] = field(default_factory=lambda: {
        "sui_image_params", "swarmui_params", "metadata", "generation_params"
    })
    
    # Core fields that should not appear in settings
    CORE_HANDLED_FIELDS: Set[str] = field(default_factory=lambda: {
        "prompt", "negativeprompt", "positive_prompt", "negative_prompt",
        "width", "height", "comfyuisampler", "autowebuisampler"
    })
    
    # SwarmUI feature indicators
    SWARMUI_FEATURES: Set[str] = field(default_factory=lambda: {
        "wildcards", "regional_prompting", "prompt_editing", 
        "dynamic_prompts", "extension_scripts", "swarm_"
    })


class SwarmUIDataLocator:
    """Locates SwarmUI data from various possible sources and structures"""
    
    def __init__(self, config: SwarmUIConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def locate_swarmui_data(self, info_data: Optional[Dict[str, Any]], 
                           raw_data: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Locate SwarmUI data from available sources.
        Returns (data_dict, source_description)
        """
        # Priority order for data sources
        sources = [
            (self._extract_from_nested_structure, info_data, "nested structure"),
            (self._extract_from_info_direct, info_data, "info data"),
            (self._extract_from_raw_json, raw_data, "raw JSON"),
        ]
        
        for extractor_func, source_data, description in sources:
            if source_data:
                result = extractor_func(source_data)
                if result:
                    self.logger.debug(f"SwarmUI: Located data from {description}")
                    return result, description
                    
        self.logger.warning("SwarmUI: No valid data source found")
        return None, "no source"
        
    def _extract_from_nested_structure(self, info_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract from nested SwarmUI structure keys"""
        if not isinstance(info_data, dict):
            return None
            
        for structure_key in self.config.SWARMUI_STRUCTURE_KEYS:
            if structure_key in info_data:
                nested_data = info_data[structure_key]
                if isinstance(nested_data, dict):
                    self.logger.debug(f"SwarmUI: Found nested data in '{structure_key}'")
                    return nested_data
                elif isinstance(nested_data, str):
                    # Try to parse as JSON
                    try:
                        parsed = json.loads(nested_data)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        continue
                        
        return None
        
    def _extract_from_info_direct(self, info_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract directly from info data if it contains SwarmUI fields"""
        if not isinstance(info_data, dict):
            return None
            
        # Check if info_data directly contains SwarmUI-like fields
        swarmui_indicators = (
            set(info_data.keys()) & 
            (set(self.config.PARAMETER_MAPPINGS.keys()) | 
             set(self.config.SAMPLER_FIELDS) |
             {key for keys in self.config.PROMPT_FIELDS.values() for key in keys})
        )
        
        if swarmui_indicators:
            self.logger.debug(f"SwarmUI: Direct info data contains {len(swarmui_indicators)} SwarmUI fields")
            return info_data
            
        return None
        
    def _extract_from_raw_json(self, raw_data: str) -> Optional[Dict[str, Any]]:
        """Extract from raw JSON string"""
        if not raw_data:
            return None
            
        try:
            parsed = json.loads(raw_data)
            if isinstance(parsed, dict):
                # Check for nested structure first
                nested = self._extract_from_nested_structure(parsed)
                if nested:
                    return nested
                    
                # Otherwise use direct
                return self._extract_from_info_direct(parsed) or parsed
                
        except json.JSONDecodeError as e:
            self.logger.debug(f"SwarmUI: Raw data is not valid JSON: {e}")
            
        return None


class SwarmUIValidator:
    """Validates SwarmUI format structure and content"""
    
    def __init__(self, config: SwarmUIConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def validate_swarmui_format(self, data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        Validate SwarmUI format and return detailed analysis.
        Returns validation results with confidence scoring.
        """
        validation_result = {
            "is_swarmui": False,
            "confidence_score": 0.0,
            "data_source": source,
            "has_prompts": False,
            "has_parameters": False,
            "has_swarmui_features": False,
            "swarmui_indicators": [],
            "validation_errors": [],
        }
        
        if not data or not isinstance(data, dict):
            validation_result["validation_errors"].append("Data is not a valid dictionary")
            return validation_result
            
        # Check for prompts
        prompt_found = self._check_prompts(data)
        validation_result["has_prompts"] = prompt_found
        if prompt_found:
            validation_result["confidence_score"] += 0.3
            validation_result["swarmui_indicators"].append("prompts")
            
        # Check for generation parameters
        param_indicators = set(data.keys()) & set(self.config.PARAMETER_MAPPINGS.keys())
        validation_result["has_parameters"] = bool(param_indicators)
        if param_indicators:
            validation_result["confidence_score"] += 0.4
            validation_result["swarmui_indicators"].extend(list(param_indicators))
            
        # Check for sampler fields
        sampler_indicators = set(data.keys()) & set(self.config.SAMPLER_FIELDS)
        if sampler_indicators:
            validation_result["confidence_score"] += 0.3
            validation_result["swarmui_indicators"].extend(list(sampler_indicators))
            
        # Check for SwarmUI-specific features
        feature_indicators = []
        for feature in self.config.SWARMUI_FEATURES:
            if any(feature in str(key).lower() for key in data.keys()):
                feature_indicators.append(feature)
                
        validation_result["has_swarmui_features"] = bool(feature_indicators)
        if feature_indicators:
            validation_result["confidence_score"] += 0.2
            validation_result["swarmui_indicators"].extend(feature_indicators)
            
        # Determine if this is definitively SwarmUI
        validation_result["is_swarmui"] = (
            validation_result["confidence_score"] >= 0.6 and
            (validation_result["has_prompts"] or validation_result["has_parameters"])
        )
        
        self.logger.debug(f"SwarmUI validation: confidence={validation_result['confidence_score']:.2f}, "
                         f"valid={validation_result['is_swarmui']}, "
                         f"indicators={len(validation_result['swarmui_indicators'])}")
        
        return validation_result
        
    def _check_prompts(self, data: Dict[str, Any]) -> bool:
        """Check if data contains prompt fields"""
        for prompt_keys in self.config.PROMPT_FIELDS.values():
            for key in prompt_keys:
                if key in data and data[key]:
                    return True
        return False


class SwarmUIExtractor:
    """Handles extraction of data from validated SwarmUI dictionaries"""
    
    def __init__(self, config: SwarmUIConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def extract_swarmui_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract SwarmUI data from validated dictionary.
        Returns structured extraction results.
        """
        result = {
            "positive": "",
            "negative": "",
            "parameters": {},
            "handled_keys": set(),
            "extraction_errors": [],
        }
        
        try:
            # Extract prompts
            positive, negative, prompt_handled = self._extract_prompts(data)
            result["positive"] = positive
            result["negative"] = negative
            result["handled_keys"].update(prompt_handled)
            
            # Extract standard parameters
            parameters, param_handled = self._extract_parameters(data)
            result["parameters"] = parameters
            result["handled_keys"].update(param_handled)
            
            # Extract sampler information
            sampler_info, sampler_handled = self._extract_sampler_info(data)
            result["parameters"].update(sampler_info)
            result["handled_keys"].update(sampler_handled)
            
            # Extract dimensions
            dimensions, dim_handled = self._extract_dimensions(data)
            result["parameters"].update(dimensions)
            result["handled_keys"].update(dim_handled)
            
            self.logger.debug(f"SwarmUI: Extracted {len(result['parameters'])} parameters")
            
        except Exception as e:
            self.logger.error(f"SwarmUI: Extraction error: {e}")
            result["extraction_errors"].append(f"Extraction failed: {e}")
            
        return result
        
    def _extract_prompts(self, data: Dict[str, Any]) -> Tuple[str, str, Set[str]]:
        """Extract positive and negative prompts"""
        positive = ""
        negative = ""
        handled_keys = set()
        
        # Extract positive prompt
        for key in self.config.PROMPT_FIELDS["positive"]:
            if key in data:
                value = data[key]
                if value:
                    positive = str(value).strip()
                    handled_keys.add(key)
                    break
                    
        # Extract negative prompt
        for key in self.config.PROMPT_FIELDS["negative"]:
            if key in data:
                value = data[key]
                if value:
                    negative = str(value).strip()
                    handled_keys.add(key)
                    break
                    
        return positive, negative, handled_keys
        
    def _extract_parameters(self, data: Dict[str, Any]) -> Tuple[Dict[str, str], Set[str]]:
        """Extract and standardize parameters"""
        parameters = {}
        handled_keys = set()
        
        for swarm_key, standard_target in self.config.PARAMETER_MAPPINGS.items():
            if swarm_key in data and data[swarm_key] is not None:
                value = data[swarm_key]
                processed_value = self._process_parameter_value(swarm_key, value)
                
                if processed_value:
                    # Handle list targets
                    if isinstance(standard_target, list):
                        target_key = standard_target[0] if standard_target else swarm_key
                    else:
                        target_key = standard_target
                        
                    parameters[target_key] = processed_value
                    handled_keys.add(swarm_key)
                    
        return parameters, handled_keys
        
    def _extract_sampler_info(self, data: Dict[str, Any]) -> Tuple[Dict[str, str], Set[str]]:
        """Extract sampler information from various possible fields"""
        sampler_info = {}
        handled_keys = set()
        
        # Priority order for sampler fields
        for field in self.config.SAMPLER_FIELDS:
            if field in data and data[field] is not None:
                value = str(data[field]).strip()
                if value:
                    sampler_info["sampler_name"] = value
                    handled_keys.add(field)
                    break
                    
        return sampler_info, handled_keys
        
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
                    self.logger.debug(f"SwarmUI: Invalid {dim_key} value: {data[dim_key]}")
                    
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
            
        # Handle list values (LoRAs, ControlNets, etc.)
        if isinstance(value, list):
            if key in ["loras", "controlnets", "embeddings"]:
                return self._format_list_parameter(value, key)
            else:
                return ", ".join(str(item) for item in value if item)
                
        # Handle dictionary values
        if isinstance(value, dict):
            return self._format_dict_parameter(value, key)
            
        # Default string conversion
        return str(value).strip()
        
    def _format_list_parameter(self, value_list: List[Any], key: str) -> str:
        """Format list parameters like LoRAs and ControlNets"""
        if not value_list:
            return ""
            
        formatted_items = []
        for item in value_list:
            if isinstance(item, dict):
                if key == "loras":
                    name = item.get("model", item.get("name", "unknown"))
                    weight = item.get("weight", item.get("strength", 1.0))
                    formatted_items.append(f"{name}:{weight}")
                elif key == "controlnets":
                    model = item.get("model", "unknown")
                    strength = item.get("strength", 1.0)
                    formatted_items.append(f"{model}:{strength}")
                else:
                    formatted_items.append(str(item))
            else:
                formatted_items.append(str(item))
                
        return ", ".join(formatted_items)
        
    def _format_dict_parameter(self, value_dict: Dict[str, Any], key: str) -> str:
        """Format dictionary parameters"""
        if not value_dict:
            return ""
            
        # For complex objects, create a readable representation
        formatted_pairs = []
        for k, v in value_dict.items():
            if v is not None:
                formatted_pairs.append(f"{k}: {v}")
                
        return "; ".join(formatted_pairs)


class SwarmUI(BaseFormat):
    """
    Enhanced SwarmUI format parser with intelligent data location and robust extraction.
    
    SwarmUI (StableSwarmUI) stores metadata in various structures and formats.
    This parser handles nested structures, multiple field variations, and comprehensive
    parameter extraction with proper validation.
    """
    
    tool = "StableSwarmUI"
    
    def __init__(
        self,
        info: Optional[Dict[str, Any]] = None,
        raw: str = "",
        width: Any = 0,
        height: Any = 0,
        logger_obj: Optional[logging.Logger] = None,
        **kwargs: Any,
    ):
        # Pre-process raw data to potentially populate info
        processed_info = self._preprocess_input_data(info, raw)
        
        super().__init__(
            info=processed_info,
            raw=raw,
            width=width,
            height=height,
            logger_obj=logger_obj,
            **kwargs,
        )
        
        # Initialize components
        self.config = SwarmUIConfig()
        self.data_locator = SwarmUIDataLocator(self.config, self._logger)
        self.validator = SwarmUIValidator(self.config, self._logger)
        self.extractor = SwarmUIExtractor(self.config, self._logger)
        
        # Store processing results
        self._located_data: Optional[Dict[str, Any]] = None
        self._data_source: str = ""
        self._validation_result: Optional[Dict[str, Any]] = None
        self._extraction_result: Optional[Dict[str, Any]] = None

    def _preprocess_input_data(self, info: Optional[Dict[str, Any]], raw: str) -> Optional[Dict[str, Any]]:
        """Preprocess input data to handle raw JSON strings"""
        if info:
            return info
            
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    self._logger.debug("SwarmUI: Preprocessed raw JSON into info dict")
                    return parsed
            except json.JSONDecodeError:
                pass  # Will be handled in main processing
                
        return info

    def _process(self) -> None:
        """Main processing pipeline for SwarmUI format"""
        self._logger.debug(f"{self.tool}: Starting SwarmUI format processing")
        
        # Locate SwarmUI data from available sources
        self._located_data, self._data_source = self.data_locator.locate_swarmui_data(self._info, self._raw)
        
        if not self._located_data:
            self._logger.debug(f"{self.tool}: No SwarmUI data located")
            self.status = self.Status.MISSING_INFO
            self._error = "No SwarmUI data found in available sources"
            return
            
        # Validate SwarmUI format
        self._validation_result = self.validator.validate_swarmui_format(self._located_data, self._data_source)
        
        if not self._validation_result["is_swarmui"]:
            confidence = self._validation_result["confidence_score"]
            errors = self._validation_result["validation_errors"]
            self._logger.debug(f"{self.tool}: Not identified as SwarmUI (confidence: {confidence:.2f}, errors: {errors})")
            self.status = self.Status.FORMAT_DETECTION_ERROR
            self._error = f"Not SwarmUI format: {'; '.join(errors) if errors else 'insufficient confidence'}"
            return
            
        # Extract data
        self._extraction_result = self.extractor.extract_swarmui_data(self._located_data)
        
        if self._extraction_result["extraction_errors"]:
            errors = self._extraction_result["extraction_errors"]
            self._logger.warning(f"{self.tool}: Extraction errors: {errors}")
            if not self._extraction_result["parameters"] and not self._extraction_result["positive"]:
                self.status = self.Status.FORMAT_ERROR
                self._error = f"SwarmUI extraction failed: {'; '.join(errors)}"
                return
                
        # Apply extraction results
        self._apply_extraction_results()
        
        # Ensure raw data is set
        self._ensure_raw_data()
        
        # Validate extraction success
        if not self._has_meaningful_extraction():
            self._logger.warning(f"{self.tool}: No meaningful data extracted")
            self.status = self.Status.FORMAT_ERROR
            self._error = "SwarmUI parsing yielded no meaningful data"
            return
            
        self._logger.info(f"{self.tool}: Successfully parsed from {self._data_source} with {self._validation_result['confidence_score']:.2f} confidence")

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
            
        # Build settings string from unhandled data
        self._build_swarmui_settings()

    def _build_swarmui_settings(self) -> None:
        """Build settings string from unhandled SwarmUI data"""
        if not self._located_data or not self._extraction_result:
            return
            
        handled_keys = self._extraction_result["handled_keys"]
        handled_keys.update(self.config.CORE_HANDLED_FIELDS)
        
        self._setting = self._build_settings_string(
            include_standard_params=False,
            remaining_data_dict=self._located_data,
            remaining_handled_keys=handled_keys,
            sort_parts=True,
        )

    def _ensure_raw_data(self) -> None:
        """Ensure raw data is populated if needed"""
        if not self._raw and self._located_data:
            try:
                self._raw = json.dumps(self._located_data, indent=2)
            except (TypeError, ValueError):
                self._raw = str(self._located_data)

    def _has_meaningful_extraction(self) -> bool:
        """Check if extraction yielded meaningful data"""
        has_prompts = bool(self._positive.strip())
        has_parameters = self._parameter_has_data()
        has_dimensions = self._width != "0" or self._height != "0"
        
        return has_prompts or has_parameters or has_dimensions

    def get_format_info(self) -> Dict[str, Any]:
        """Get detailed information about the parsed SwarmUI data"""
        return {
            "format_name": self.tool,
            "data_source": self._data_source,
            "validation_result": self._validation_result,
            "has_positive_prompt": bool(self._positive),
            "has_negative_prompt": bool(self._negative),
            "parameter_count": len([v for v in self._parameter.values() if v and v != self.DEFAULT_PARAMETER_PLACEHOLDER]),
            "has_dimensions": self._width != "0" or self._height != "0",
            "dimensions": f"{self._width}x{self._height}" if self._width != "0" and self._height != "0" else None,
            "swarmui_features": self._analyze_swarmui_features(),
        }

    def _analyze_swarmui_features(self) -> Dict[str, Any]:
        """Analyze SwarmUI-specific features detected"""
        features = {
            "has_loras": False,
            "has_controlnets": False,
            "has_embeddings": False,
            "has_refiner": False,
            "has_vae": False,
            "advanced_features": [],
            "feature_count": 0,
        }
        
        # Check parameters for features
        features["has_loras"] = "loras" in self._parameter
        features["has_controlnets"] = "controlnets" in self._parameter
        features["has_embeddings"] = "embeddings" in self._parameter
        features["has_refiner"] = any("refiner" in key for key in self._parameter.keys())
        features["has_vae"] = "vae_model" in self._parameter
        
        # Build advanced features list
        if features["has_loras"]:
            features["advanced_features"].append("loras")
        if features["has_controlnets"]:
            features["advanced_features"].append("controlnets")
        if features["has_embeddings"]:
            features["advanced_features"].append("embeddings")
        if features["has_refiner"]:
            features["advanced_features"].append("refiner")
        if features["has_vae"]:
            features["advanced_features"].append("vae")
            
        features["feature_count"] = len(features["advanced_features"])
        
        return features

    def debug_swarmui_processing(self) -> Dict[str, Any]:
        """Get comprehensive debugging information about SwarmUI processing"""
        return {
            "input_data": {
                "has_info": bool(self._info),
                "info_keys": list(self._info.keys()) if self._info else [],
                "has_raw": bool(self._raw),
                "raw_length": len(self._raw) if self._raw else 0,
                "raw_preview": self._raw[:200] if self._raw else None,
            },
            "data_location": {
                "data_source": self._data_source,
                "located_data_keys": list(self._located_data.keys()) if self._located_data else [],
                "nested_structures_found": [
                    key for key in (self._info.keys() if self._info else [])
                    if key in self.config.SWARMUI_STRUCTURE_KEYS
                ],
            },
            "validation_details": self._validation_result,
            "extraction_details": self._extraction_result,
            "processing_summary": {
                "tool_name": self.tool,
                "parameter_count": len(self._parameter),
                "has_prompts": bool(self._positive or self._negative),
                "swarmui_specific_params": [
                    key for key in self._parameter.keys() 
                    if key in self.config.PARAMETER_MAPPINGS.values()
                ],
            },
            "config_info": {
                "parameter_mappings": len(self.config.PARAMETER_MAPPINGS),
                "sampler_fields": len(self.config.SAMPLER_FIELDS),
                "structure_keys": len(self.config.SWARMUI_STRUCTURE_KEYS),
                "feature_indicators": len(self.config.SWARMUI_FEATURES),
            }
        }