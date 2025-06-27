# dataset_tools/vendored_sdpr/format/yodayo.py

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .a1111 import A1111


class YodayoIdentificationMethod(Enum):
    """Methods for identifying Yodayo/Moescape format"""
    EXIF_SOFTWARE = "exif_software"
    NGMS_PARAMETER = "ngms_parameter"
    LORA_HASHES_KEY = "lora_hashes_key"
    MODEL_UUID = "model_uuid"
    VERSION_EXCLUSION = "version_exclusion"


@dataclass
class YodayoConfig:
    """Configuration for Yodayo/Moescape format parsing - systematically organized"""
    
    # Software tag identifiers
    SOFTWARE_IDENTIFIERS: Dict[str, Set[str]] = field(default_factory=lambda: {
        "positive": {"yodayo", "moescape"},
        "negative": {"automatic1111", "forge", "sd.next", "webui"}
    })
    
    # Yodayo-specific parameter mappings
    PARAMETER_MAPPINGS: Dict[str, str] = field(default_factory=lambda: {
        "NGMS": "yodayo_ngms",  # Yodayo-specific parameter
        "Lora hashes": "lora_hashes_data",
    })
    
    # Pattern for UUID model identification
    UUID_PATTERN: re.Pattern = field(default_factory=lambda: re.compile(
        r"[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}", re.IGNORECASE
    ))
    
    # Version patterns for exclusion
    VERSION_EXCLUSION_PATTERNS: Dict[str, re.Pattern] = field(default_factory=lambda: {
        "a1111_webui": re.compile(r"v\d+\.\d+\.\d+"),
        "forge_simple": re.compile(r".*forge.*", re.IGNORECASE),
        "forge_complex": re.compile(r"v\d+\.\d+\.\d+.*-v\d+\.\d+\.\d+"),
        "forge_alt": re.compile(r"f\d+\.\d+\.\d+v\d+\.\d+\.\d+"),
    })
    
    # A1111/Forge-specific parameter indicators (negative signals)
    NEGATIVE_INDICATORS: Dict[str, Set[str]] = field(default_factory=lambda: {
        "hires_params": {"Hires upscale", "Hires upscaler", "Hires steps"},
        "ultimate_sd": {"Ultimate SD upscale"},
        "adetailer": {"ADetailer"},
        "controlnet": {"ControlNet "},
    })
    
    # LoRA extraction patterns
    LORA_PATTERNS: Dict[str, re.Pattern] = field(default_factory=lambda: {
        "prompt_lora": re.compile(r"<lora:([^:]+):([0-9\.]+)(:[^>]+)?>"),
        "hash_entry": re.compile(r"([^:]+):\s*([0-9a-fA-F]+)"),
    })


class YodayoIdentificationEngine:
    """Advanced identification engine for Yodayo/Moescape detection"""
    
    def __init__(self, config: YodayoConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def identify_yodayo_format(self, settings_dict: Dict[str, str], 
                              info_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive Yodayo identification with confidence scoring.
        Returns detailed identification analysis.
        """
        identification_result = {
            "is_yodayo": False,
            "confidence_score": 0.0,
            "identification_methods": [],
            "positive_indicators": {},
            "negative_indicators": {},
            "analysis_details": {},
        }
        
        # Check EXIF software tag (highest priority)
        software_score = self._check_software_tag(info_data, identification_result)
        
        # Check Yodayo-specific parameters
        param_score = self._check_yodayo_parameters(settings_dict, identification_result)
        
        # Check model UUID format
        model_score = self._check_model_format(settings_dict, identification_result)
        
        # Check for negative indicators (A1111/Forge specific)
        negative_score = self._check_negative_indicators(settings_dict, identification_result)
        
        # Calculate overall confidence
        identification_result["confidence_score"] = self._calculate_confidence(
            software_score, param_score, model_score, negative_score
        )
        
        # Determine if this is definitively Yodayo
        identification_result["is_yodayo"] = self._is_definitive_yodayo(identification_result)
        
        self.logger.debug(f"Yodayo identification: confidence={identification_result['confidence_score']:.2f}, "
                         f"methods={identification_result['identification_methods']}")
        
        return identification_result
        
    def _check_software_tag(self, info_data: Optional[Dict[str, Any]], 
                           result: Dict[str, Any]) -> float:
        """Check EXIF software tag for Yodayo indicators"""
        if not info_data or "software_tag" not in info_data:
            return 0.0
            
        software_tag = str(info_data["software_tag"]).lower()
        result["analysis_details"]["software_tag"] = software_tag
        
        # Check positive indicators
        for identifier in self.config.SOFTWARE_IDENTIFIERS["positive"]:
            if identifier in software_tag:
                result["positive_indicators"]["software_positive"] = identifier
                result["identification_methods"].append(YodayoIdentificationMethod.EXIF_SOFTWARE)
                self.logger.debug(f"Yodayo: Positive software tag match: {identifier}")
                return 1.0  # Definitive positive
                
        # Check negative indicators
        for identifier in self.config.SOFTWARE_IDENTIFIERS["negative"]:
            if identifier in software_tag:
                result["negative_indicators"]["software_negative"] = identifier
                self.logger.debug(f"Yodayo: Negative software tag match: {identifier}")
                return -0.5  # Strong negative
                
        return 0.0
        
    def _check_yodayo_parameters(self, settings_dict: Dict[str, str], 
                                result: Dict[str, Any]) -> float:
        """Check for Yodayo-specific parameters"""
        score = 0.0
        
        # Check for NGMS parameter (very strong indicator)
        if "NGMS" in settings_dict:
            result["positive_indicators"]["ngms_present"] = settings_dict["NGMS"]
            result["identification_methods"].append(YodayoIdentificationMethod.NGMS_PARAMETER)
            score += 0.8
            self.logger.debug("Yodayo: NGMS parameter found")
            
        # Check for Yodayo-specific "Lora hashes" key format
        if "Lora hashes" in settings_dict:
            result["positive_indicators"]["lora_hashes_key"] = True
            result["identification_methods"].append(YodayoIdentificationMethod.LORA_HASHES_KEY)
            score += 0.6
            self.logger.debug("Yodayo: 'Lora hashes' key found")
            
        return score
        
    def _check_model_format(self, settings_dict: Dict[str, str], 
                           result: Dict[str, Any]) -> float:
        """Check if model is in UUID format (Yodayo indicator)"""
        model_value = settings_dict.get("Model", "")
        if not model_value:
            return 0.0
            
        is_uuid = bool(self.config.UUID_PATTERN.fullmatch(model_value))
        result["analysis_details"]["model_is_uuid"] = is_uuid
        result["analysis_details"]["model_value"] = model_value
        
        if is_uuid:
            result["positive_indicators"]["model_uuid"] = model_value
            result["identification_methods"].append(YodayoIdentificationMethod.MODEL_UUID)
            self.logger.debug(f"Yodayo: Model is UUID format: {model_value}")
            return 0.4
            
        return 0.0
        
    def _check_negative_indicators(self, settings_dict: Dict[str, str], 
                                  result: Dict[str, Any]) -> float:
        """Check for A1111/Forge-specific indicators (negative for Yodayo)"""
        negative_score = 0.0
        
        # Check version patterns
        version_str = settings_dict.get("Version", "")
        if version_str:
            for pattern_name, pattern in self.config.VERSION_EXCLUSION_PATTERNS.items():
                if pattern.match(version_str):
                    result["negative_indicators"][f"version_{pattern_name}"] = version_str
                    result["identification_methods"].append(YodayoIdentificationMethod.VERSION_EXCLUSION)
                    negative_score -= 0.6
                    self.logger.debug(f"Yodayo: Negative version pattern {pattern_name}: {version_str}")
                    break
                    
        # Check for A1111/Forge-specific parameter groups
        for indicator_name, param_set in self.config.NEGATIVE_INDICATORS.items():
            if indicator_name == "hires_params":
                # Check if all hires parameters are present
                if param_set.issubset(set(settings_dict.keys())):
                    result["negative_indicators"]["hires_parameters"] = list(param_set)
                    negative_score -= 0.4
                    self.logger.debug("Yodayo: Found A1111 hires parameters")
            else:
                # Check if any parameter starts with the indicator
                for param in param_set:
                    found_params = [key for key in settings_dict.keys() if key.startswith(param)]
                    if found_params:
                        result["negative_indicators"][indicator_name] = found_params
                        negative_score -= 0.3
                        self.logger.debug(f"Yodayo: Found {indicator_name} parameters: {found_params}")
                        break
                        
        return negative_score
        
    def _calculate_confidence(self, software_score: float, param_score: float, 
                             model_score: float, negative_score: float) -> float:
        """Calculate overall confidence score"""
        # Weight the scores (software tag is most important)
        weighted_score = (software_score * 3.0 + param_score * 2.0 + 
                         model_score * 1.0 + negative_score * 1.5)
        
        # Normalize to 0-1 scale
        max_possible = 3.0 + 2.8 + 0.4  # Rough maximum positive score
        return max(0.0, min(1.0, weighted_score / max_possible))
        
    def _is_definitive_yodayo(self, result: Dict[str, Any]) -> bool:
        """Determine if we have definitive proof this is Yodayo"""
        # Positive software tag is definitive
        if "software_positive" in result["positive_indicators"]:
            return True
            
        # NGMS parameter is very strong
        if "ngms_present" in result["positive_indicators"]:
            return True
            
        # Lora hashes key + UUID model is strong
        if ("lora_hashes_key" in result["positive_indicators"] and 
            "model_uuid" in result["positive_indicators"]):
            return True
            
        # High confidence without strong negatives
        if (result["confidence_score"] >= 0.7 and 
            not result["negative_indicators"]):
            return True
            
        return False


class YodayoLoRAProcessor:
    """Advanced LoRA processing for Yodayo format"""
    
    def __init__(self, config: YodayoConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def parse_lora_hashes(self, lora_hashes_str: str) -> List[Dict[str, str]]:
        """Parse LoRA hashes string into structured data"""
        if not lora_hashes_str:
            return []
            
        loras = []
        parts = [part.strip() for part in lora_hashes_str.split(",") if part.strip()]
        
        for part in parts:
            match = self.config.LORA_PATTERNS["hash_entry"].match(part)
            if match:
                lora_info = {
                    "id_or_name": match.group(1).strip(),
                    "hash": match.group(2).strip(),
                }
                loras.append(lora_info)
                self.logger.debug(f"Yodayo: Parsed LoRA hash - {lora_info['id_or_name']}: {lora_info['hash']}")
            else:
                self.logger.warning(f"Yodayo: Could not parse LoRA hash part: '{part}'")
                
        return loras
        
    def extract_loras_from_prompt(self, prompt_text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Extract LoRA tags from prompt text and return cleaned prompt"""
        if not prompt_text:
            return "", []
            
        loras = []
        cleaned_parts = []
        last_end = 0
        
        for match in self.config.LORA_PATTERNS["prompt_lora"].finditer(prompt_text):
            # Add text before this LoRA tag
            cleaned_parts.append(prompt_text[last_end:match.start()])
            
            # Extract LoRA information
            name_or_id = match.group(1)
            weight = match.group(2)
            
            lora_info = {
                "name_or_id": name_or_id,
                "weight": weight,
            }
            
            # Check for additional parameters
            if match.group(3):
                lora_info["additional_params"] = match.group(3)[1:]  # Remove leading ':'
                
            loras.append(lora_info)
            last_end = match.end()
            
            self.logger.debug(f"Yodayo: Extracted LoRA from prompt - {name_or_id}: {weight}")
            
        # Add remaining text
        cleaned_parts.append(prompt_text[last_end:])
        
        # Clean up the prompt
        cleaned_prompt = "".join(cleaned_parts)
        cleaned_prompt = re.sub(r"\s{2,}", " ", cleaned_prompt).strip(" ,")
        
        return cleaned_prompt, loras


class YodayoFormat(A1111):
    """
    Enhanced Yodayo/Moescape format parser with advanced identification and LoRA processing.
    
    Inherits from A1111 for base text parsing capabilities while providing
    sophisticated Yodayo-specific identification and parameter extraction.
    """
    
    tool = "Yodayo/Moescape"
    
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
        
        # Initialize Yodayo-specific components
        self.config = YodayoConfig()
        self.identification_engine = YodayoIdentificationEngine(self.config, self._logger)
        self.lora_processor = YodayoLoRAProcessor(self.config, self._logger)
        
        # Store processing results
        self.parsed_loras_from_hashes: List[Dict[str, str]] = []
        self._identification_result: Optional[Dict[str, Any]] = None

    def _process(self) -> None:
        """Main processing pipeline for Yodayo format"""
        self._logger.debug(f"{self.__class__.tool}: Starting Yodayo format processing")
        
        # First, use A1111 parser to handle the base text format
        super()._process()
        
        # Check if A1111 parsing was successful
        if self.status != self.Status.READ_SUCCESS:
            self._logger.debug(f"{self.__class__.tool}: A1111 parsing failed (status: {self.status})")
            return
            
        # Parse A1111 settings for Yodayo analysis
        settings_dict = {}
        if self._setting:
            settings_dict = self._parse_settings_string_to_dict(self._setting)
            
        # Perform Yodayo identification
        self._identification_result = self.identification_engine.identify_yodayo_format(
            settings_dict, self._info
        )
        
        # Check for early software tag confirmation
        is_yodayo_software = (
            self._info and 
            "software_tag" in self._info and
            any(identifier in str(self._info["software_tag"]).lower() 
                for identifier in self.config.SOFTWARE_IDENTIFIERS["positive"])
        )
        
        # Handle case where A1111 found no settings but we have Yodayo software tag
        if not settings_dict and not is_yodayo_software:
            self._logger.debug(f"{self.__class__.tool}: No settings found and no Yodayo software tag")
            self.status = self.Status.FORMAT_DETECTION_ERROR
            self._error = "A1111 found no settings block and no Yodayo software tag"
            return
            
        # Check identification result
        if not self._identification_result["is_yodayo"]:
            confidence = self._identification_result["confidence_score"]
            methods = self._identification_result["identification_methods"]
            self._logger.debug(f"{self.__class__.tool}: Not identified as Yodayo (confidence: {confidence:.2f}, methods: {methods})")
            self.status = self.Status.FORMAT_DETECTION_ERROR
            self._error = "A1111 text parsed but not identified as Yodayo/Moescape"
            return
            
        # Confirmed as Yodayo - proceed with Yodayo-specific processing
        self.tool = self.__class__.tool
        self._process_yodayo_specific_data(settings_dict)
        
        self._logger.info(f"{self.tool}: Successfully processed with {self._identification_result['confidence_score']:.2f} confidence")

    def _process_yodayo_specific_data(self, settings_dict: Dict[str, str]) -> None:
        """Process Yodayo-specific parameters and LoRA data"""
        handled_keys = set()
        
        # Extract Yodayo-specific parameters
        self._populate_parameters_from_map(settings_dict, self.config.PARAMETER_MAPPINGS, handled_keys)
        
        # Process LoRA hashes
        lora_hashes_str = settings_dict.get("Lora hashes")
        if lora_hashes_str:
            self.parsed_loras_from_hashes = self.lora_processor.parse_lora_hashes(lora_hashes_str)
            if self.parsed_loras_from_hashes:
                self._parameter["lora_hashes_data"] = self.parsed_loras_from_hashes
            handled_keys.add("Lora hashes")
            
        # Extract LoRAs from prompts
        if self._positive:
            cleaned_positive, extracted_loras_pos = self.lora_processor.extract_loras_from_prompt(self._positive)
            if extracted_loras_pos:
                self._positive = cleaned_positive
                self._parameter["loras_from_prompt_positive"] = extracted_loras_pos
                
        if self._negative:
            cleaned_negative, extracted_loras_neg = self.lora_processor.extract_loras_from_prompt(self._negative)
            if extracted_loras_neg:
                self._negative = cleaned_negative
                self._parameter["loras_from_prompt_negative"] = extracted_loras_neg
                
        # Add identification metadata
        if self._identification_result:
            self._parameter["yodayo_confidence"] = f"{self._identification_result['confidence_score']:.2f}"
            methods = [method.value for method in self._identification_result["identification_methods"]]
            self._parameter["identification_methods"] = str(len(methods))

    def get_format_info(self) -> Dict[str, Any]:
        """Get detailed information about the parsed Yodayo data"""
        return {
            "format_name": self.tool,
            "identification_result": self._identification_result,
            "has_positive_prompt": bool(self._positive),
            "has_negative_prompt": bool(self._negative),
            "parameter_count": len([v for v in self._parameter.values() if v and v != self.DEFAULT_PARAMETER_PLACEHOLDER]),
            "has_dimensions": self._width != "0" or self._height != "0",
            "dimensions": f"{self._width}x{self._height}" if self._width != "0" and self._height != "0" else None,
            "yodayo_features": self._analyze_yodayo_features(),
        }

    def _analyze_yodayo_features(self) -> Dict[str, Any]:
        """Analyze Yodayo-specific features detected"""
        features = {
            "has_ngms": False,
            "has_lora_hashes": False,
            "has_prompt_loras": False,
            "has_uuid_model": False,
            "lora_count": 0,
            "feature_summary": [],
        }
        
        # Check for NGMS
        features["has_ngms"] = "yodayo_ngms" in self._parameter
        if features["has_ngms"]:
            features["feature_summary"].append("NGMS parameter")
            
        # Check for LoRA hashes
        features["has_lora_hashes"] = bool(self.parsed_loras_from_hashes)
        if features["has_lora_hashes"]:
            features["lora_count"] += len(self.parsed_loras_from_hashes)
            features["feature_summary"].append(f"LoRA hashes: {len(self.parsed_loras_from_hashes)}")
            
        # Check for prompt LoRAs
        prompt_lora_count = 0
        if "loras_from_prompt_positive" in self._parameter:
            prompt_lora_count += len(self._parameter["loras_from_prompt_positive"])
        if "loras_from_prompt_negative" in self._parameter:
            prompt_lora_count += len(self._parameter["loras_from_prompt_negative"])
            
        features["has_prompt_loras"] = prompt_lora_count > 0
        if features["has_prompt_loras"]:
            features["lora_count"] += prompt_lora_count
            features["feature_summary"].append(f"Prompt LoRAs: {prompt_lora_count}")
            
        # Check for UUID model
        if self._identification_result:
            features["has_uuid_model"] = "model_uuid" in self._identification_result.get("positive_indicators", {})
            if features["has_uuid_model"]:
                features["feature_summary"].append("UUID model")
                
        return features

    def debug_yodayo_identification(self) -> Dict[str, Any]:
        """Get comprehensive debugging information about Yodayo identification"""
        return {
            "input_data": {
                "has_info": bool(self._info),
                "info_keys": list(self._info.keys()) if self._info else [],
                "has_raw": bool(self._raw),
                "raw_length": len(self._raw) if self._raw else 0,
                "a1111_parsing_status": self.status.name if hasattr(self.status, 'name') else str(self.status),
            },
            "identification_details": self._identification_result,
            "lora_processing": {
                "loras_from_hashes": len(self.parsed_loras_from_hashes),
                "loras_from_positive_prompt": len(self._parameter.get("loras_from_prompt_positive", [])),
                "loras_from_negative_prompt": len(self._parameter.get("loras_from_prompt_negative", [])),
            },
            "feature_analysis": self._analyze_yodayo_features(),
            "config_info": {
                "software_identifiers": {k: list(v) for k, v in self.config.SOFTWARE_IDENTIFIERS.items()},
                "parameter_mappings": len(self.config.PARAMETER_MAPPINGS),
                "negative_indicators": {k: list(v) for k, v in self.config.NEGATIVE_INDICATORS.items()},
            }
        }