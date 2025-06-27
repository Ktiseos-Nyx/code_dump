# dataset_tools/vendored_sdpr/format/easydiffusion.py

__author__ = "receyuki"
__filename__ = "easydiffusion.py"
# MODIFIED by Ktiseos Nyx for Dataset-Tools
__copyright__ = "Copyright 2023, Receyuki; Modified 2025, Ktiseos Nyx"
__email__ = "receyuki@gmail.com"

import json
import logging
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Dict, Optional, Set, Union, Callable
from dataclasses import dataclass, field

from .base_format import BaseFormat


@dataclass
class EasyDiffusionConfig:
    """Configuration for EasyDiffusion format parsing - systematically organized"""
    
    # Parameter mapping with flexible key handling
    PARAMETER_MAPPINGS: Dict[str, Union[str, list[str]]] = field(default_factory=lambda: {
        "seed": "seed",
        "use_stable_diffusion_model": "model",
        "clip_skip": "clip_skip",
        "use_vae_model": "vae_model", 
        "sampler_name": "sampler_name",
        "num_inference_steps": "steps",
        "guidance_scale": "cfg_scale",
        "scheduler": "scheduler",
        "denoising_strength": "denoising_strength",
        "use_face_correction": "face_correction",
        "use_upscale": "upscaling_method",
    })
    
    # Prompt key variations (EasyDiffusion isn't always consistent)
    PROMPT_KEY_VARIATIONS: Dict[str, Set[str]] = field(default_factory=lambda: {
        "positive": {"prompt", "Prompt", "positive_prompt", "Positive Prompt"},
        "negative": {"negative_prompt", "Negative Prompt", "negative", "Negative"}
    })
    
    # Dimension key variations
    DIMENSION_KEYS: Set[str] = field(default_factory=lambda: {
        "width", "height", "Width", "Height", "image_width", "image_height"
    })
    
    # Keys that should be processed with special value processors
    SPECIAL_PROCESSING_KEYS: Dict[str, str] = field(default_factory=lambda: {
        "model": "path_processor",
        "vae_model": "path_processor",
        "use_stable_diffusion_model": "path_processor",
        "use_vae_model": "path_processor",
    })


class EasyDiffusionPathProcessor:
    """Handles path processing for model files - because EasyDiffusion loves full paths"""
    
    @staticmethod
    def extract_filename_from_path(value: Any) -> str:
        """Extract just the filename from a full path"""
        if not value or not isinstance(value, str):
            return str(value) if value is not None else ""
            
        value_str = str(value).strip()
        if not value_str:
            return ""
            
        # Handle Windows paths with drive letters
        if PureWindowsPath(value_str).drive and len(PureWindowsPath(value_str).parts) > 1:
            return PureWindowsPath(value_str).name
            
        # Handle POSIX paths with multiple components
        if not PureWindowsPath(value_str).drive and len(PurePosixPath(value_str).parts) > 1:
            return PurePosixPath(value_str).name
            
        # If it's just a filename already, return as-is
        return value_str


class EasyDiffusionValidator:
    """Validates EasyDiffusion JSON structure and content"""
    
    def __init__(self, config: EasyDiffusionConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def validate_json_source(self, raw_data: str, info_data: Any) -> Optional[Dict[str, Any]]:
        """Validate and extract JSON from available sources"""
        # Priority: raw_data -> info_data (if dict) -> info_data (if string)
        json_source, source_desc = self._select_json_source(raw_data, info_data)
        
        if not json_source:
            self.logger.warning("EasyDiffusion: No JSON source data available")
            return None
            
        # If source is already a dict, validate and return
        if isinstance(json_source, dict):
            if self._is_valid_easydiffusion_structure(json_source):
                self.logger.debug(f"EasyDiffusion: Using pre-parsed dict from {source_desc}")
                return json_source
            else:
                self.logger.debug(f"EasyDiffusion: Dict from {source_desc} doesn't match expected structure")
                return None
                
        # Parse JSON string
        try:
            parsed_data = json.loads(json_source)
            if not isinstance(parsed_data, dict):
                self.logger.debug("EasyDiffusion: Parsed JSON is not a dictionary")
                return None
                
            if self._is_valid_easydiffusion_structure(parsed_data):
                self.logger.debug(f"EasyDiffusion: Successfully parsed JSON from {source_desc}")
                return parsed_data
            else:
                self.logger.debug("EasyDiffusion: Parsed JSON doesn't match EasyDiffusion structure")
                return None
                
        except json.JSONDecodeError as e:
            self.logger.debug(f"EasyDiffusion: JSON decode failed from {source_desc}: {e}")
            return None
            
    def _select_json_source(self, raw_data: str, info_data: Any) -> tuple[Union[str, Dict[str, Any]], str]:
        """Select the best JSON source from available data"""
        if raw_data and raw_data.strip():
            return raw_data, "raw data"
        elif isinstance(info_data, dict):
            return info_data, "info (pre-parsed dict)"
        elif isinstance(info_data, str) and info_data.strip():
            return info_data, "info (JSON string)"
        else:
            return "", "no source"
            
    def _is_valid_easydiffusion_structure(self, data: Dict[str, Any]) -> bool:
        """Check if the data structure matches EasyDiffusion format"""
        if not data:
            return False
            
        # Look for characteristic EasyDiffusion keys
        has_prompt_keys = any(
            key in data for key_set in self.config.PROMPT_KEY_VARIATIONS.values() 
            for key in key_set
        )
        
        has_ed_specific_keys = any(
            key in data for key in [
                "use_stable_diffusion_model", "num_inference_steps", 
                "guidance_scale", "sampler_name", "seed"
            ]
        )
        
        # Must have either prompts or ED-specific generation params
        return has_prompt_keys or has_ed_specific_keys


class EasyDiffusionExtractor:
    """Handles extraction of data from validated EasyDiffusion JSON"""
    
    def __init__(self, config: EasyDiffusionConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.path_processor = EasyDiffusionPathProcessor()
        self._value_processors: Dict[str, Callable] = {
            "path_processor": self.path_processor.extract_filename_from_path,
        }

    def debug_json_parsing(self, raw_data: str, info_data: Any = None) -> Dict[str, Any]:
        """
        Debug utility for troubleshooting JSON parsing issues.
        Returns detailed information about parsing attempts.
        """
        debug_info = {
            "raw_data_info": {
                "has_raw_data": bool(raw_data),
                "raw_data_type": type(raw_data).__name__,
                "raw_data_length": len(raw_data) if raw_data else 0,
                "raw_data_preview": raw_data[:100] if raw_data else None,
            },
            "info_data_info": {
                "has_info_data": info_data is not None,
                "info_data_type": type(info_data).__name__,
                "info_data_preview": str(info_data)[:100] if info_data else None,
            },
            "parsing_attempts": []
        }
        
        # Attempt to parse each source
        sources = [
            ("raw_data", raw_data),
            ("info_data", info_data if isinstance(info_data, str) else None),
        ]
        
        for source_name, source_data in sources:
            if not source_data:
                continue
                
            attempt = {
                "source": source_name,
                "success": False,
                "error": None,
                "result_type": None,
                "is_dict": False,
                "has_ed_structure": False,
            }
            
            try:
                if isinstance(source_data, str):
                    parsed = json.loads(source_data)
                    attempt["success"] = True
                    attempt["result_type"] = type(parsed).__name__
                    attempt["is_dict"] = isinstance(parsed, dict)
                    
                    if isinstance(parsed, dict):
                        attempt["has_ed_structure"] = self.validator._is_valid_easydiffusion_structure(parsed)
                        attempt["keys_found"] = list(parsed.keys())[:10]  # First 10 keys
                        
            except json.JSONDecodeError as e:
                attempt["error"] = str(e)
            except Exception as e:
                attempt["error"] = f"Unexpected error: {e}"
                
            debug_info["parsing_attempts"].append(attempt)
            
        return debug_info
        
    def extract_prompts(self, data: Dict[str, Any]) -> tuple[str, str]:
        """Extract positive and negative prompts with flexible key matching"""
        positive = self._extract_prompt_by_variations(data, self.config.PROMPT_KEY_VARIATIONS["positive"])
        negative = self._extract_prompt_by_variations(data, self.config.PROMPT_KEY_VARIATIONS["negative"])
        
        self.logger.debug(f"EasyDiffusion: Extracted prompts - positive: {len(positive)} chars, negative: {len(negative)} chars")
        return positive, negative
        
    def _extract_prompt_by_variations(self, data: Dict[str, Any], key_variations: Set[str]) -> str:
        """Extract prompt using multiple possible key names"""
        for key in key_variations:
            if key in data:
                value = data[key]
                if value is not None:
                    return str(value).strip()
        return ""
        
    def extract_parameters(self, data: Dict[str, Any]) -> tuple[Dict[str, str], Set[str]]:
        """Extract and process parameters, returning (parameters, handled_keys)"""
        parameters = {}
        handled_keys = set()
        
        for ed_key, canonical_target in self.config.PARAMETER_MAPPINGS.items():
            if ed_key not in data or data[ed_key] is None:
                continue
                
            # Determine if we need special processing
            processor_name = self.config.SPECIAL_PROCESSING_KEYS.get(canonical_target)
            if isinstance(canonical_target, list):
                # Check if any target in the list needs special processing
                for target in canonical_target:
                    if target in self.config.SPECIAL_PROCESSING_KEYS:
                        processor_name = self.config.SPECIAL_PROCESSING_KEYS[target]
                        break
                        
            # Process the value
            raw_value = data[ed_key]
            if processor_name and processor_name in self._value_processors:
                processed_value = self._value_processors[processor_name](raw_value)
            else:
                processed_value = self._clean_value(raw_value)
                
            # Store in parameters
            if isinstance(canonical_target, list):
                # Store under the first target key
                if canonical_target and processed_value:
                    parameters[canonical_target[0]] = processed_value
            else:
                if processed_value:
                    parameters[canonical_target] = processed_value
                    
            handled_keys.add(ed_key)
            
        self.logger.debug(f"EasyDiffusion: Extracted {len(parameters)} parameters")
        return parameters, handled_keys
        
    def extract_dimensions(self, data: Dict[str, Any]) -> tuple[str, str, Set[str]]:
        """Extract width and height, returning (width, height, handled_keys)"""
        width = "0"
        height = "0"
        handled_keys = set()
        
        # Look for width
        for key in ["width", "Width", "image_width"]:
            if key in data and data[key] is not None:
                try:
                    width = str(int(data[key]))
                    handled_keys.add(key)
                    break
                except (ValueError, TypeError):
                    continue
                    
        # Look for height
        for key in ["height", "Height", "image_height"]:
            if key in data and data[key] is not None:
                try:
                    height = str(int(data[key]))
                    handled_keys.add(key)
                    break
                except (ValueError, TypeError):
                    continue
                    
        self.logger.debug(f"EasyDiffusion: Extracted dimensions - {width}x{height}")
        return width, height, handled_keys
        
    def _clean_value(self, value: Any) -> str:
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


class EasyDiffusion(BaseFormat):
    """
    Enhanced EasyDiffusion format parser with robust validation and flexible key handling.
    
    EasyDiffusion stores metadata as JSON with various key naming conventions.
    This parser handles the inconsistencies while extracting meaningful data.
    """
    
    tool = "Easy Diffusion"
    
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
        self.config = EasyDiffusionConfig()
        self.validator = EasyDiffusionValidator(self.config, self._logger)
        self.extractor = EasyDiffusionExtractor(self.config, self._logger)

    def _process(self) -> None:
        """Main processing pipeline for EasyDiffusion format"""
        self._logger.debug(f"{self.tool}: Starting EasyDiffusion format processing")
        
        # Validate and extract JSON data
        json_data = self.validator.validate_json_source(self._raw, self._info)
        if json_data is None:
            self._logger.debug(f"{self.tool}: No valid EasyDiffusion JSON found")
            self.status = self.Status.FORMAT_DETECTION_ERROR
            self._error = "No valid EasyDiffusion JSON structure found"
            return
            
        # Extract all data components
        success = self._extract_all_components(json_data)
        if not success:
            return  # Error already set
            
        # Build settings string from remaining data
        self._build_comprehensive_settings(json_data)
        
        # Validate extraction success
        if not self._has_meaningful_extraction():
            self._logger.warning(f"{self.tool}: No meaningful data extracted")
            self.status = self.Status.FORMAT_ERROR
            self._error = "EasyDiffusion parsing yielded no meaningful data"
            return
            
        self._logger.info(f"{self.tool}: Successfully parsed EasyDiffusion metadata")

    def _extract_all_components(self, data: Dict[str, Any]) -> bool:
        """Extract all data components from validated JSON"""
        try:
            # Track all handled keys
            all_handled_keys = set()
            
            # Extract prompts
            self._positive, self._negative = self.extractor.extract_prompts(data)
            
            # Add prompt keys to handled set
            for key_set in self.config.PROMPT_KEY_VARIATIONS.values():
                for key in key_set:
                    if key in data:
                        all_handled_keys.add(key)
                        
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
            self._error = f"EasyDiffusion extraction failed: {e}"
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
        """Get detailed information about the parsed EasyDiffusion data"""
        return {
            "format_name": self.tool,
            "has_positive_prompt": bool(self._positive),
            "has_negative_prompt": bool(self._negative),
            "parameter_count": len([v for v in self._parameter.values() if v and v != self.DEFAULT_PARAMETER_PLACEHOLDER]),
            "has_dimensions": self._width != "0" or self._height != "0",
            "dimensions": f"{self._width}x{self._height}" if self._width != "0" and self._height != "0" else None,
            "settings_items": len(self._setting.split(", ")) if self._setting else 0,
            "extracted_model_names": self._get_extracted_model_info(),
        }

    def _get_extracted_model_info(self) -> Dict[str, Optional[str]]:
        """Get information about extracted model names"""
        return {
            "main_model": self._parameter.get("model"),
            "vae_model": self._parameter.get("vae_model"),
            "sampler": self._parameter.get("sampler_name"),
        }

    def validate_easydiffusion_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate EasyDiffusion data structure and return detailed analysis.
        Useful for debugging and format verification.
        """
        analysis = {
            "is_valid_structure": False,
            "has_prompts": False,
            "has_generation_params": False,
            "found_prompt_keys": [],
            "found_param_keys": [],
            "dimension_info": {},
            "total_keys": len(data) if data else 0,
        }
        
        if not data:
            return analysis
            
        # Check for prompt keys
        for prompt_type, key_variations in self.config.PROMPT_KEY_VARIATIONS.items():
            found_keys = [key for key in key_variations if key in data]
            if found_keys:
                analysis["found_prompt_keys"].extend(found_keys)
                analysis["has_prompts"] = True
                
        # Check for generation parameter keys
        param_keys_found = [key for key in self.config.PARAMETER_MAPPINGS.keys() if key in data]
        analysis["found_param_keys"] = param_keys_found
        analysis["has_generation_params"] = bool(param_keys_found)
        
        # Check dimensions
        width_keys = [key for key in ["width", "Width", "image_width"] if key in data]
        height_keys = [key for key in ["height", "Height", "image_height"] if key in data]
        analysis["dimension_info"] = {
            "width_keys_found": width_keys,
            "height_keys_found": height_keys,
            "has_dimensions": bool(width_keys and height_keys)
        }
        
        # Overall validation
        analysis["is_valid_structure"] = analysis["has_prompts"] or analysis["has_generation_params"]
        
        return analysis

    def get_processing_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics about the processing pipeline"""
        return {
            "tool_name": self.tool,
            "processing_status": self.status.name if hasattr(self.status, 'name') else str(self.status),
            "error_message": self._error if hasattr(self, '_error') else None,
            "config_info": {
                "parameter_mappings_count": len(self.config.PARAMETER_MAPPINGS),
                "prompt_variations_count": sum(len(variations) for variations in self.config.PROMPT_KEY_VARIATIONS.values()),
                "special_processing_keys": list(self.config.SPECIAL_PROCESSING_KEYS.keys()),
            },
            "extraction_results": {
                "positive_prompt_length": len(self._positive) if hasattr(self, '_positive') else 0,
                "negative_prompt_length": len(self._negative) if hasattr(self, '_negative') else 0,
                "parameter_count": len(self._parameter) if hasattr(self, '_parameter') else 0,
                "has_dimensions": hasattr(self, '_width') and self._width != "0",
            }
        }