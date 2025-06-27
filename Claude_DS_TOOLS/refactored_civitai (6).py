# dataset_tools/vendored_sdpr/format/invokeai.py

__author__ = "receyuki"
__filename__ = "invokeai.py"
# MODIFIED by Ktiseos Nyx for Dataset-Tools
__copyright__ = "Copyright 2023, Receyuki; Modified 2025, Ktiseos Nyx"
__email__ = "receyuki@gmail.com"

import json
import logging
import re
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from .base_format import BaseFormat


class InvokeAIFormat(Enum):
    """Enumeration of InvokeAI metadata formats"""
    INVOKEAI_METADATA = "invokeai_metadata"
    SD_METADATA = "sd-metadata"
    DREAM = "Dream"
    UNKNOWN = "unknown"


@dataclass
class InvokeAIConfig:
    """Configuration for InvokeAI format parsing - systematically organized"""
    
    # Format detection keys and their priorities
    FORMAT_DETECTION_KEYS: Dict[InvokeAIFormat, Set[str]] = field(default_factory=lambda: {
        InvokeAIFormat.INVOKEAI_METADATA: {"invokeai_metadata"},
        InvokeAIFormat.SD_METADATA: {"sd-metadata"},
        InvokeAIFormat.DREAM: {"Dream"},
    })
    
    # Parameter mappings for each format
    INVOKEAI_METADATA_MAPPINGS: Dict[str, Union[str, List[str]]] = field(default_factory=lambda: {
        "seed": "seed",
        "steps": "steps",
        "cfg_scale": "cfg_scale",
        "scheduler": "scheduler",
        "refiner_steps": "refiner_steps",
        "refiner_cfg_scale": "refiner_cfg_scale",
        "refiner_scheduler": "refiner_scheduler",
        "refiner_positive_aesthetic_score": "refiner_positive_aesthetic_score",
        "refiner_negative_aesthetic_score": "refiner_negative_aesthetic_score",
        "refiner_start": "refiner_start",
        "controlnet_guidance_start": "controlnet_guidance_start",
        "controlnet_guidance_end": "controlnet_guidance_end",
        "ip_adapter_scale": "ip_adapter_scale",
    })
    
    SD_METADATA_MAPPINGS: Dict[str, Union[str, List[str]]] = field(default_factory=lambda: {
        "sampler": "sampler_name",
        "seed": "seed",
        "cfg_scale": "cfg_scale",
        "steps": "steps",
        "strength": "denoising_strength",
        "fit": "fit_mode",
    })
    
    DREAM_FORMAT_MAPPINGS: Dict[str, Union[str, List[str]]] = field(default_factory=lambda: {
        "s": "steps",
        "S": "seed",
        "C": "cfg_scale",
        "A": "sampler_name",
        "W": "width",
        "H": "height",
        "f": "init_strength",
        "U": "upscaling_factor",
        "G": "gfpgan_strength",
    })
    
    # Dream format display mapping
    DREAM_DISPLAY_MAPPING: Dict[str, str] = field(default_factory=lambda: {
        "s": "Steps",
        "S": "Seed", 
        "C": "CFG scale",
        "A": "Sampler",
        "W": "Width",
        "H": "Height",
        "f": "Init strength",
        "U": "Upscaling factor",
        "G": "GFPGAN strength",
    })
    
    # Prompt handling patterns
    PROMPT_PATTERNS: Dict[str, Pattern[str]] = field(default_factory=lambda: {
        "dream_main": re.compile(r'"(.*?)"\s*(-\S.*)?$'),
        "dream_options": re.compile(r"-(\w+)\s+([\w.-]+)"),
        "negative_brackets": re.compile(r"^(.*?)(?:\s*\[(.*?)\])?$"),
    })


class InvokeAIFormatDetector:
    """Detects which InvokeAI format is present in the data"""
    
    def __init__(self, config: InvokeAIConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def detect_format(self, info_data: Dict[str, Any]) -> Tuple[InvokeAIFormat, Dict[str, Any]]:
        """
        Detect InvokeAI format and return format type with analysis.
        Returns (format_type, detection_info)
        """
        if not info_data:
            return InvokeAIFormat.UNKNOWN, {"error": "No info data provided"}
            
        detection_info = {
            "detected_format": InvokeAIFormat.UNKNOWN,
            "available_keys": list(info_data.keys()),
            "format_scores": {},
            "confidence": 0.0,
        }
        
        # Check each format in priority order
        for format_type, required_keys in self.config.FORMAT_DETECTION_KEYS.items():
            score = self._calculate_format_score(info_data, required_keys)
            detection_info["format_scores"][format_type.value] = score
            
            if score > 0:
                detection_info["detected_format"] = format_type
                detection_info["confidence"] = score
                self.logger.debug(f"InvokeAI: Detected {format_type.value} format with score {score}")
                return format_type, detection_info
                
        self.logger.debug("InvokeAI: No recognized format detected")
        return InvokeAIFormat.UNKNOWN, detection_info
        
    def _calculate_format_score(self, data: Dict[str, Any], required_keys: Set[str]) -> float:
        """Calculate confidence score for a specific format"""
        if not required_keys:
            return 0.0
            
        found_keys = set(data.keys()) & required_keys
        score = len(found_keys) / len(required_keys)
        
        # Bonus for non-empty values
        for key in found_keys:
            if data.get(key):  # Non-empty value
                score += 0.1
                
        return min(score, 1.0)


class InvokeAIPromptProcessor:
    """Handles InvokeAI-specific prompt processing"""
    
    def __init__(self, config: InvokeAIConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def split_invokeai_prompt(self, prompt: str) -> Tuple[str, str]:
        """Split InvokeAI prompt into positive and negative components"""
        if not prompt:
            return "", ""
            
        # Use regex pattern from config
        pattern = self.config.PROMPT_PATTERNS["negative_brackets"]
        match = pattern.fullmatch(prompt.strip())
        
        if match:
            positive = match.group(1).strip()
            negative = (match.group(2) or "").strip()
        else:
            positive = prompt.strip()
            negative = ""
            
        self.logger.debug(f"InvokeAI: Split prompt - positive: {len(positive)} chars, negative: {len(negative)} chars")
        return positive, negative
        
    def extract_dream_prompts_and_options(self, dream_string: str) -> Tuple[str, str, Dict[str, str]]:
        """Extract prompts and options from Dream format string"""
        if not dream_string:
            return "", "", {}
            
        # Use regex pattern from config
        main_pattern = self.config.PROMPT_PATTERNS["dream_main"]
        match = main_pattern.search(dream_string)
        
        if not match:
            self.logger.warning(f"InvokeAI: Could not parse Dream string structure: {dream_string[:100]}")
            return "", "", {}
            
        full_prompt = match.group(1).strip('" ')
        options_str = (match.group(2) or "").strip()
        
        # Split the prompt into positive/negative
        positive, negative = self.split_invokeai_prompt(full_prompt)
        
        # Parse options
        option_pattern = self.config.PROMPT_PATTERNS["dream_options"]
        options_dict = dict(option_pattern.findall(options_str))
        
        self.logger.debug(f"InvokeAI: Extracted Dream format - {len(options_dict)} options")
        return positive, negative, options_dict


class InvokeAIExtractor:
    """Handles extraction from different InvokeAI formats"""
    
    def __init__(self, config: InvokeAIConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.prompt_processor = InvokeAIPromptProcessor(config, logger)
        
    def extract_invokeai_metadata(self, raw_json: str) -> Dict[str, Any]:
        """Extract data from invokeai_metadata format"""
        try:
            data = json.loads(raw_json)
            if not isinstance(data, dict):
                raise ValueError("Metadata is not a JSON dictionary")
                
            result = {
                "positive": str(data.get("positive_prompt", "")).strip(),
                "negative": str(data.get("negative_prompt", "")).strip(),
                "positive_sdxl": {},
                "negative_sdxl": {},
                "is_sdxl": False,
                "parameters": {},
                "handled_keys": set(),
                "raw_data": raw_json,
            }
            
            # Handle SDXL style prompts
            if data.get("positive_style_prompt"):
                result["positive_sdxl"]["style"] = str(data.get("positive_style_prompt", "")).strip()
            if data.get("negative_style_prompt"):
                result["negative_sdxl"]["style"] = str(data.get("negative_style_prompt", "")).strip()
                
            if result["positive_sdxl"] or result["negative_sdxl"]:
                result["is_sdxl"] = True
                
            # Mark handled keys
            result["handled_keys"].update([
                "positive_prompt", "negative_prompt", 
                "positive_style_prompt", "negative_style_prompt"
            ])
            
            # Handle model information
            model_info = data.get("model")
            if isinstance(model_info, dict):
                if model_info.get("model_name"):
                    result["parameters"]["model"] = str(model_info["model_name"])
                if model_info.get("hash"):
                    result["parameters"]["model_hash"] = str(model_info["hash"])
                result["handled_keys"].add("model")
                
            # Extract standard parameters
            for invoke_key, standard_key in self.config.INVOKEAI_METADATA_MAPPINGS.items():
                if invoke_key in data and data[invoke_key] is not None:
                    if isinstance(standard_key, list):
                        target_key = standard_key[0] if standard_key else invoke_key
                    else:
                        target_key = standard_key
                    result["parameters"][target_key] = str(data[invoke_key])
                    result["handled_keys"].add(invoke_key)
                    
            self.logger.debug(f"InvokeAI: Extracted invokeai_metadata with {len(result['parameters'])} parameters")
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"InvokeAI: Error parsing invokeai_metadata: {e}")
            raise
            
    def extract_sd_metadata(self, raw_json: str) -> Dict[str, Any]:
        """Extract data from sd-metadata format"""
        try:
            data = json.loads(raw_json)
            if not isinstance(data, dict):
                raise ValueError("sd-metadata is not a JSON dictionary")
                
            image_data = data.get("image")
            if not isinstance(image_data, dict):
                raise ValueError("'image' field missing or not a dict in sd-metadata")
                
            result = {
                "positive": "",
                "negative": "",
                "positive_sdxl": {},
                "negative_sdxl": {},
                "is_sdxl": False,
                "parameters": {},
                "handled_keys": set(),
                "raw_data": raw_json,
            }
            
            # Extract prompt
            prompt_field = image_data.get("prompt")
            prompt_text = ""
            
            if isinstance(prompt_field, list) and prompt_field:
                prompt_entry = prompt_field[0]
                if isinstance(prompt_entry, dict):
                    prompt_text = str(prompt_entry.get("prompt", ""))
            elif isinstance(prompt_field, str):
                prompt_text = prompt_field
                
            result["positive"], result["negative"] = self.prompt_processor.split_invokeai_prompt(prompt_text)
            result["handled_keys"].add("prompt")
            
            # Handle model weights
            if "model_weights" in data:
                result["parameters"]["model"] = str(data["model_weights"])
                result["handled_keys"].add("model_weights")
                
            # Extract image parameters
            for sd_key, standard_key in self.config.SD_METADATA_MAPPINGS.items():
                if sd_key in image_data and image_data[sd_key] is not None:
                    if isinstance(standard_key, list):
                        target_key = standard_key[0] if standard_key else sd_key
                    else:
                        target_key = standard_key
                    result["parameters"][target_key] = str(image_data[sd_key])
                    result["handled_keys"].add(sd_key)
                    
            self.logger.debug(f"InvokeAI: Extracted sd-metadata with {len(result['parameters'])} parameters")
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"InvokeAI: Error parsing sd-metadata: {e}")
            raise
            
    def extract_dream_format(self, dream_string: str) -> Dict[str, Any]:
        """Extract data from Dream format string"""
        try:
            positive, negative, options = self.prompt_processor.extract_dream_prompts_and_options(dream_string)
            
            result = {
                "positive": positive,
                "negative": negative,
                "positive_sdxl": {},
                "negative_sdxl": {},
                "is_sdxl": False,
                "parameters": {},
                "handled_keys": set(),
                "raw_data": dream_string,
                "dream_options": options,
            }
            
            # Extract parameters from options
            for short_key, standard_key in self.config.DREAM_FORMAT_MAPPINGS.items():
                if short_key in options:
                    if isinstance(standard_key, list):
                        target_key = standard_key[0] if standard_key else short_key
                    else:
                        target_key = standard_key
                    result["parameters"][target_key] = options[short_key]
                    result["handled_keys"].add(short_key)
                    
            self.logger.debug(f"InvokeAI: Extracted Dream format with {len(result['parameters'])} parameters")
            return result
            
        except Exception as e:
            self.logger.error(f"InvokeAI: Error parsing Dream format: {e}")
            raise


class InvokeAI(BaseFormat):
    """
    Enhanced InvokeAI format parser with multi-format support and robust extraction.
    
    Supports three InvokeAI metadata formats:
    1. invokeai_metadata (modern JSON format)
    2. sd-metadata (legacy JSON format)
    3. Dream (command-line string format)
    """
    
    tool = "InvokeAI"
    
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
        self.config = InvokeAIConfig()
        self.format_detector = InvokeAIFormatDetector(self.config, self._logger)
        self.extractor = InvokeAIExtractor(self.config, self._logger)
        
        # Store detection and extraction results
        self._detected_format: InvokeAIFormat = InvokeAIFormat.UNKNOWN
        self._detection_info: Dict[str, Any] = {}
        self._extraction_result: Dict[str, Any] = {}

    def _process(self) -> None:
        """Main processing pipeline for InvokeAI formats"""
        self._logger.debug(f"{self.tool}: Starting InvokeAI format processing")
        
        # Validate input data
        if not self._info:
            self._logger.warning(f"{self.tool}: No info data provided")
            self.status = self.Status.MISSING_INFO
            self._error = "No InvokeAI metadata provided"
            return
            
        # Detect format
        self._detected_format, self._detection_info = self.format_detector.detect_format(self._info)
        
        if self._detected_format == InvokeAIFormat.UNKNOWN:
            self._logger.debug(f"{self.tool}: No recognized InvokeAI format detected")
            self.status = self.Status.FORMAT_DETECTION_ERROR
            self._error = "No recognized InvokeAI format found"
            return
            
        # Extract data based on detected format
        try:
            success = self._extract_by_format()
            if not success:
                return  # Error already set
                
        except Exception as e:
            self._logger.error(f"{self.tool}: Extraction failed: {e}")
            self.status = self.Status.FORMAT_ERROR
            self._error = f"InvokeAI extraction failed: {e}"
            return
            
        # Apply extraction results
        self._apply_extraction_results()
        
        # Validate extraction success
        if not self._has_meaningful_extraction():
            self._logger.warning(f"{self.tool}: No meaningful data extracted")
            self.status = self.Status.FORMAT_ERROR
            self._error = "InvokeAI parsing yielded no meaningful data"
            return
            
        self._logger.info(f"{self.tool}: Successfully parsed {self._detected_format.value} format")

    def _extract_by_format(self) -> bool:
        """Extract data based on detected format"""
        format_key = list(self.config.FORMAT_DETECTION_KEYS[self._detected_format])[0]
        raw_data = self._info.get(format_key, "")
        
        if self._detected_format == InvokeAIFormat.INVOKEAI_METADATA:
            self._extraction_result = self.extractor.extract_invokeai_metadata(raw_data)
            
        elif self._detected_format == InvokeAIFormat.SD_METADATA:
            self._extraction_result = self.extractor.extract_sd_metadata(raw_data)
            
        elif self._detected_format == InvokeAIFormat.DREAM:
            self._extraction_result = self.extractor.extract_dream_format(raw_data)
            
        else:
            self._logger.error(f"{self.tool}: Unknown format for extraction: {self._detected_format}")
            self.status = self.Status.FORMAT_ERROR
            self._error = f"Unknown InvokeAI format: {self._detected_format.value}"
            return False
            
        return True

    def _apply_extraction_results(self) -> None:
        """Apply extraction results to instance variables"""
        if not self._extraction_result:
            return
            
        # Apply prompts
        self._positive = self._extraction_result.get("positive", "")
        self._negative = self._extraction_result.get("negative", "")
        self._positive_sdxl = self._extraction_result.get("positive_sdxl", {})
        self._negative_sdxl = self._extraction_result.get("negative_sdxl", {})
        self._is_sdxl = self._extraction_result.get("is_sdxl", False)
        
        # Apply parameters
        parameters = self._extraction_result.get("parameters", {})
        self._parameter.update(parameters)
        
        # Handle dimensions (extract from parameters or original data)
        self._extract_dimensions_from_results()
        
        # Set raw data
        if not self._raw and "raw_data" in self._extraction_result:
            self._raw = self._extraction_result["raw_data"]
            
        # Build settings string
        self._build_format_specific_settings()

    def _extract_dimensions_from_results(self) -> None:
        """Extract and set dimensions from extraction results"""
        # Check if dimensions are in parameters
        if "width" in self._parameter:
            try:
                self._width = str(int(self._parameter["width"]))
            except (ValueError, TypeError):
                pass
                
        if "height" in self._parameter:
            try:
                self._height = str(int(self._parameter["height"]))
            except (ValueError, TypeError):
                pass
                
        # Also check original data source for dimensions
        if self._detected_format == InvokeAIFormat.INVOKEAI_METADATA:
            self._extract_invokeai_dimensions()
        elif self._detected_format == InvokeAIFormat.SD_METADATA:
            self._extract_sd_metadata_dimensions()
            
        # Update parameter dict with final dimensions
        if self._width != "0":
            self._parameter["width"] = self._width
        if self._height != "0":
            self._parameter["height"] = self._height
        if self._width != "0" and self._height != "0":
            self._parameter["size"] = f"{self._width}x{self._height}"

    def _extract_invokeai_dimensions(self) -> None:
        """Extract dimensions from invokeai_metadata format"""
        format_key = list(self.config.FORMAT_DETECTION_KEYS[InvokeAIFormat.INVOKEAI_METADATA])[0]
        raw_data = self._info.get(format_key, "{}")
        
        try:
            data = json.loads(raw_data)
            if isinstance(data, dict):
                if "width" in data and self._width == "0":
                    self._width = str(int(data["width"]))
                if "height" in data and self._height == "0":
                    self._height = str(int(data["height"]))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    def _extract_sd_metadata_dimensions(self) -> None:
        """Extract dimensions from sd-metadata format"""
        format_key = list(self.config.FORMAT_DETECTION_KEYS[InvokeAIFormat.SD_METADATA])[0]
        raw_data = self._info.get(format_key, "{}")
        
        try:
            data = json.loads(raw_data)
            if isinstance(data, dict):
                image_data = data.get("image", {})
                if isinstance(image_data, dict):
                    if "width" in image_data and self._width == "0":
                        self._width = str(int(image_data["width"]))
                    if "height" in image_data and self._height == "0":
                        self._height = str(int(image_data["height"]))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    def _build_format_specific_settings(self) -> None:
        """Build settings string specific to the detected format"""
        if self._detected_format == InvokeAIFormat.DREAM:
            self._build_dream_settings()
        else:
            self._build_standard_settings()

    def _build_dream_settings(self) -> None:
        """Build settings string for Dream format with proper key formatting"""
        if "dream_options" not in self._extraction_result:
            return
            
        options = self._extraction_result["dream_options"]
        handled_keys = self._extraction_result.get("handled_keys", set())
        
        # Create display mapping for remaining keys
        def dream_key_formatter(short_key: str) -> str:
            return self.config.DREAM_DISPLAY_MAPPING.get(short_key, short_key.capitalize())
            
        self._setting = self._build_settings_string(
            remaining_data_dict=options,
            remaining_handled_keys=handled_keys,
            remaining_key_formatter=dream_key_formatter,
            include_standard_params=False,
        )

    def _build_standard_settings(self) -> None:
        """Build settings string for standard JSON formats"""
        # Get the original data
        format_key = list(self.config.FORMAT_DETECTION_KEYS[self._detected_format])[0]
        raw_data = self._info.get(format_key, "{}")
        
        try:
            data = json.loads(raw_data)
            if isinstance(data, dict):
                handled_keys = self._extraction_result.get("handled_keys", set())
                
                # For sd-metadata, also include image-level handled keys
                if self._detected_format == InvokeAIFormat.SD_METADATA:
                    handled_keys.add("image")  # Don't include the whole image dict
                    
                self._setting = self._build_settings_string(
                    remaining_data_dict=data,
                    remaining_handled_keys=handled_keys,
                    include_standard_params=False,
                )
        except (json.JSONDecodeError, ValueError):
            self._setting = ""

    def _has_meaningful_extraction(self) -> bool:
        """Check if extraction yielded meaningful data"""
        has_prompts = bool(self._positive.strip())
        has_parameters = self._parameter_has_data()
        has_dimensions = self._width != "0" or self._height != "0"
        
        return has_prompts or has_parameters or has_dimensions

    def get_format_info(self) -> Dict[str, Any]:
        """Get detailed information about the detected and parsed format"""
        return {
            "detected_format": self._detected_format.value,
            "detection_info": self._detection_info,
            "has_positive_prompt": bool(self._positive),
            "has_negative_prompt": bool(self._negative),
            "is_sdxl": self._is_sdxl,
            "parameter_count": len([v for v in self._parameter.values() if v and v != self.DEFAULT_PARAMETER_PLACEHOLDER]),
            "has_dimensions": self._width != "0" or self._height != "0",
            "dimensions": f"{self._width}x{self._height}" if self._width != "0" and self._height != "0" else None,
            "invokeai_features": self._analyze_invokeai_features(),
        }

    def _analyze_invokeai_features(self) -> Dict[str, Any]:
        """Analyze InvokeAI-specific features detected"""
        features = {
            "has_refiner": False,
            "has_controlnet": False,
            "has_ip_adapter": False,
            "has_model_info": False,
            "advanced_features": [],
        }
        
        # Check for refiner
        refiner_keys = [k for k in self._parameter.keys() if k.startswith("refiner_")]
        features["has_refiner"] = bool(refiner_keys)
        if refiner_keys:
            features["advanced_features"].append("refiner")
            
        # Check for ControlNet
        controlnet_keys = [k for k in self._parameter.keys() if "controlnet" in k.lower()]
        features["has_controlnet"] = bool(controlnet_keys)
        if controlnet_keys:
            features["advanced_features"].append("controlnet")
            
        # Check for IP Adapter
        ip_adapter_keys = [k for k in self._parameter.keys() if "ip_adapter" in k.lower()]
        features["has_ip_adapter"] = bool(ip_adapter_keys)
        if ip_adapter_keys:
            features["advanced_features"].append("ip_adapter")
            
        # Check for model info
        features["has_model_info"] = "model" in self._parameter or "model_hash" in self._parameter
        
        return features

    def debug_format_detection(self) -> Dict[str, Any]:
        """Get detailed debugging information about format detection"""
        return {
            "input_info_keys": list(self._info.keys()) if self._info else [],
            "detection_result": self._detection_info,
            "extraction_summary": {
                "format_used": self._detected_format.value,
                "extraction_success": bool(self._extraction_result),
                "parameters_extracted": len(self._extraction_result.get("parameters", {})),
                "prompts_extracted": bool(self._extraction_result.get("positive") or self._extraction_result.get("negative")),
            } if self._extraction_result else {},
            "config_info": {
                "supported_formats": [f.value for f in InvokeAIFormat if f != InvokeAIFormat.UNKNOWN],
                "total_parameter_mappings": sum([
                    len(self.config.INVOKEAI_METADATA_MAPPINGS),
                    len(self.config.SD_METADATA_MAPPINGS), 
                    len(self.config.DREAM_FORMAT_MAPPINGS)
                ]),
            }
        }