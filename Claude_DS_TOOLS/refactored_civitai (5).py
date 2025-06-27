# dataset_tools/vendored_sdpr/format/forge_format.py

__author__ = "receyuki & Ktiseos Nyx"
__filename__ = "forge_format.py"
__copyright__ = "Copyright 2023, Receyuki; Modified 2025, Ktiseos Nyx"
__email__ = "receyuki@gmail.com; your_email@example.com"

import re
import logging
from typing import Any, Dict, List, Optional, Pattern, Set
from dataclasses import dataclass, field

from .a1111 import A1111


@dataclass
class ForgeSignatureConfig:
    """Configuration for Forge/ReForge signature detection - systematically organized"""
    
    # Primary signature patterns (definitive Forge/ReForge identifiers)
    PRIMARY_SIGNATURES: Dict[str, Pattern[str]] = field(default_factory=lambda: {
        "forge_version": re.compile(r"Version:\s*f", re.IGNORECASE),
        "reforge_version": re.compile(r"Version:\s*reforge", re.IGNORECASE),
        "forge_commit": re.compile(r"Version:\s*forge-[a-f0-9]+", re.IGNORECASE),
    })
    
    # Secondary signature markers (strong indicators but may need combination)
    SECONDARY_SIGNATURES: Set[str] = field(default_factory=lambda: {
        "Schedule type: Automatic",
        "Hires Module 1:",
        "Hires Module 2:", 
        "Hires Module 3:",
        "Forge couple:",
        "Forge attention couple:",
        "Regional Prompter:",
        "Ultimate SD Upscale:",
    })
    
    # Tertiary indicators (weaker signals, used for confidence scoring)
    TERTIARY_INDICATORS: Set[str] = field(default_factory=lambda: {
        "schedulers",  # Forge-specific scheduler references
        "forge_",      # Any forge_ prefixed parameters
        "reforge_",    # Any reforge_ prefixed parameters
        "automatic",   # In context of scheduling
        "regional",    # Regional prompting features
    })
    
    # Forge-specific parameter patterns
    FORGE_PARAMETER_PATTERNS: Dict[str, Pattern[str]] = field(default_factory=lambda: {
        "forge_param": re.compile(r"Forge\s+\w+:", re.IGNORECASE),
        "reforge_param": re.compile(r"ReForge\s+\w+:", re.IGNORECASE),
        "hires_module": re.compile(r"Hires\s+Module\s+\d+:", re.IGNORECASE),
        "schedule_type": re.compile(r"Schedule\s+type:", re.IGNORECASE),
    })


class ForgeSignatureDetector:
    """Advanced signature detection system for Forge/ReForge identification"""
    
    def __init__(self, config: ForgeSignatureConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def detect_forge_signature(self, raw_text: str) -> Dict[str, Any]:
        """
        Comprehensive Forge signature detection with confidence scoring.
        Returns detailed analysis of Forge markers found.
        """
        if not raw_text:
            return self._empty_detection_result()
            
        detection_result = {
            "is_forge": False,
            "confidence_score": 0.0,
            "forge_variant": None,  # "forge", "reforge", or "unknown"
            "primary_matches": [],
            "secondary_matches": [],
            "tertiary_matches": [],
            "parameter_matches": [],
            "total_indicators": 0,
        }
        
        # Check primary signatures (definitive identifiers)
        primary_score = self._check_primary_signatures(raw_text, detection_result)
        
        # Check secondary signatures (strong indicators)
        secondary_score = self._check_secondary_signatures(raw_text, detection_result)
        
        # Check tertiary indicators (supporting evidence)
        tertiary_score = self._check_tertiary_indicators(raw_text, detection_result)
        
        # Check parameter patterns
        parameter_score = self._check_parameter_patterns(raw_text, detection_result)
        
        # Calculate overall confidence
        detection_result["confidence_score"] = self._calculate_confidence_score(
            primary_score, secondary_score, tertiary_score, parameter_score
        )
        
        # Determine if this is definitively Forge
        detection_result["is_forge"] = self._is_definitive_forge(detection_result)
        
        # Determine Forge variant
        detection_result["forge_variant"] = self._determine_forge_variant(detection_result)
        
        self.logger.debug(f"Forge detection: confidence={detection_result['confidence_score']:.2f}, "
                         f"variant={detection_result['forge_variant']}, "
                         f"indicators={detection_result['total_indicators']}")
        
        return detection_result
        
    def _check_primary_signatures(self, text: str, result: Dict[str, Any]) -> float:
        """Check for primary Forge signatures (definitive markers)"""
        score = 0.0
        
        for name, pattern in self.config.PRIMARY_SIGNATURES.items():
            matches = pattern.findall(text)
            if matches:
                result["primary_matches"].append({
                    "signature": name,
                    "matches": matches,
                    "pattern": pattern.pattern
                })
                score += 1.0  # Each primary signature is worth 1.0 points
                
        result["total_indicators"] += len(result["primary_matches"])
        return score
        
    def _check_secondary_signatures(self, text: str, result: Dict[str, Any]) -> float:
        """Check for secondary Forge signatures (strong indicators)"""
        score = 0.0
        
        for signature in self.config.SECONDARY_SIGNATURES:
            if signature in text:
                result["secondary_matches"].append(signature)
                score += 0.7  # Each secondary signature is worth 0.7 points
                
        result["total_indicators"] += len(result["secondary_matches"])
        return score
        
    def _check_tertiary_indicators(self, text: str, result: Dict[str, Any]) -> float:
        """Check for tertiary indicators (supporting evidence)"""
        score = 0.0
        text_lower = text.lower()
        
        for indicator in self.config.TERTIARY_INDICATORS:
            if indicator.lower() in text_lower:
                result["tertiary_matches"].append(indicator)
                score += 0.3  # Each tertiary indicator is worth 0.3 points
                
        result["total_indicators"] += len(result["tertiary_matches"])
        return score
        
    def _check_parameter_patterns(self, text: str, result: Dict[str, Any]) -> float:
        """Check for Forge-specific parameter patterns"""
        score = 0.0
        
        for name, pattern in self.config.FORGE_PARAMETER_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                result["parameter_matches"].append({
                    "pattern_name": name,
                    "matches": matches
                })
                score += 0.5  # Each parameter pattern is worth 0.5 points
                
        result["total_indicators"] += len(result["parameter_matches"])
        return score
        
    def _calculate_confidence_score(self, primary: float, secondary: float, 
                                   tertiary: float, parameter: float) -> float:
        """Calculate overall confidence score with weighted components"""
        # Primary signatures have highest weight
        weighted_score = (primary * 4.0 + secondary * 2.0 + tertiary * 1.0 + parameter * 1.5)
        
        # Normalize to 0-1 scale (max possible score with reasonable limits)
        max_reasonable_score = 8.0  # Reasonable maximum for normalization
        normalized_score = min(weighted_score / max_reasonable_score, 1.0)
        
        return normalized_score
        
    def _is_definitive_forge(self, result: Dict[str, Any]) -> bool:
        """Determine if we have definitive proof this is Forge"""
        # Any primary signature is definitive
        if result["primary_matches"]:
            return True
            
        # Multiple strong secondary signatures can be definitive
        if len(result["secondary_matches"]) >= 2:
            return True
            
        # High confidence with multiple types of evidence
        if (result["confidence_score"] >= 0.8 and 
            result["secondary_matches"] and 
            result["parameter_matches"]):
            return True
            
        return False
        
    def _determine_forge_variant(self, result: Dict[str, Any]) -> Optional[str]:
        """Determine which Forge variant this is"""
        # Check primary matches for variant indicators
        for match in result["primary_matches"]:
            signature = match["signature"]
            if "reforge" in signature.lower():
                return "reforge"
            elif "forge" in signature.lower():
                return "forge"
                
        # Check secondary matches
        reforge_indicators = ["ReForge", "reforge"]
        forge_indicators = ["Forge", "forge"]
        
        has_reforge = any(any(indicator in match for indicator in reforge_indicators) 
                         for match in result["secondary_matches"])
        has_forge = any(any(indicator in match for indicator in forge_indicators) 
                       for match in result["secondary_matches"])
        
        if has_reforge and not has_forge:
            return "reforge"
        elif has_forge and not has_reforge:
            return "forge"
        elif has_reforge and has_forge:
            return "forge_family"  # Both variants detected
            
        return "unknown" if result["is_forge"] else None
        
    def _empty_detection_result(self) -> Dict[str, Any]:
        """Return empty detection result for invalid input"""
        return {
            "is_forge": False,
            "confidence_score": 0.0,
            "forge_variant": None,
            "primary_matches": [],
            "secondary_matches": [],
            "tertiary_matches": [],
            "parameter_matches": [],
            "total_indicators": 0,
        }


class ForgeParameterEnhancer:
    """Enhances parameter extraction with Forge-specific knowledge"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def enhance_forge_parameters(self, base_parameters: Dict[str, str], 
                                raw_text: str, detection_result: Dict[str, Any]) -> Dict[str, str]:
        """Enhance base parameters with Forge-specific additions"""
        enhanced = base_parameters.copy()
        
        # Add Forge variant information
        if detection_result["forge_variant"]:
            enhanced["forge_variant"] = detection_result["forge_variant"]
            
        # Add confidence score for debugging
        enhanced["forge_confidence"] = f"{detection_result['confidence_score']:.2f}"
        
        # Extract specific Forge parameters
        forge_specific = self._extract_forge_specific_parameters(raw_text)
        enhanced.update(forge_specific)
        
        return enhanced
        
    def _extract_forge_specific_parameters(self, text: str) -> Dict[str, str]:
        """Extract Forge-specific parameters from text"""
        forge_params = {}
        
        # Extract schedule type
        schedule_match = re.search(r"Schedule\s+type:\s*([^,\n]+)", text, re.IGNORECASE)
        if schedule_match:
            forge_params["schedule_type"] = schedule_match.group(1).strip()
            
        # Extract Hires modules
        hires_modules = re.findall(r"Hires\s+Module\s+(\d+):\s*([^,\n]+)", text, re.IGNORECASE)
        for module_num, module_value in hires_modules:
            forge_params[f"hires_module_{module_num}"] = module_value.strip()
            
        # Extract Forge couple settings
        forge_couple_match = re.search(r"Forge\s+couple:\s*([^,\n]+)", text, re.IGNORECASE)
        if forge_couple_match:
            forge_params["forge_couple"] = forge_couple_match.group(1).strip()
            
        return forge_params


class ForgeFormat(A1111):
    """
    Enhanced Forge/ReForge format parser with advanced signature detection.
    
    Inherits from A1111 for text parsing capabilities while providing
    sophisticated Forge-specific identification and parameter extraction.
    """
    
    tool = "Forge"
    
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
        
        # Initialize Forge-specific components
        self.signature_config = ForgeSignatureConfig()
        self.signature_detector = ForgeSignatureDetector(self.signature_config, self._logger)
        self.parameter_enhancer = ForgeParameterEnhancer(self._logger)
        
        # Store detection results for analysis
        self._forge_detection_result: Optional[Dict[str, Any]] = None

    def _process(self) -> None:
        """
        Process and identify Forge/ReForge-specific metadata.
        
        Uses advanced signature detection before falling back to A1111 parsing.
        Does NOT call super()._process() - performs independent validation.
        """
        self._logger.debug(f"{self.tool}: Starting Forge format processing")
        
        # Ensure we have raw text data
        if not self._raw:
            self._raw = self._extract_raw_data_from_info()
            
        if not self._raw:
            self._logger.debug(f"{self.tool}: No raw text data found")
            raise self.NotApplicableError("No raw text data found for Forge signature detection")
            
        # Perform comprehensive signature detection
        self._forge_detection_result = self.signature_detector.detect_forge_signature(self._raw)
        
        # Check if this is definitively Forge
        if not self._forge_detection_result["is_forge"]:
            confidence = self._forge_detection_result["confidence_score"]
            indicators = self._forge_detection_result["total_indicators"]
            self._logger.debug(f"{self.tool}: Not identified as Forge (confidence: {confidence:.2f}, indicators: {indicators})")
            raise self.NotApplicableError("No definitive Forge/ReForge signatures found")
            
        # Update tool name based on detected variant
        variant = self._forge_detection_result["forge_variant"]
        if variant == "reforge":
            self.tool = "ReForge"
        elif variant == "forge":
            self.tool = "Forge"
        elif variant == "forge_family":
            self.tool = "Forge/ReForge"
        else:
            self.tool = "Forge (Unknown Variant)"
            
        self._logger.info(f"Identified as {self.tool} with {self._forge_detection_result['confidence_score']:.2f} confidence")
        
        # Use inherited A1111 parsing for the heavy lifting
        self._parse_a1111_text_format()
        
        # Enhance parameters with Forge-specific additions
        self._enhance_forge_parameters()
        
        self._logger.info(f"{self.tool}: Successfully parsed Forge metadata")

    def _enhance_forge_parameters(self) -> None:
        """Enhance extracted parameters with Forge-specific data"""
        if self._forge_detection_result:
            enhanced_params = self.parameter_enhancer.enhance_forge_parameters(
                self._parameter, 
                self._raw, 
                self._forge_detection_result
            )
            self._parameter.update(enhanced_params)

    def get_forge_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of Forge detection and features"""
        if not self._forge_detection_result:
            return {"error": "No Forge detection performed"}
            
        analysis = {
            "detection_summary": {
                "is_forge": self._forge_detection_result["is_forge"],
                "confidence_score": self._forge_detection_result["confidence_score"],
                "forge_variant": self._forge_detection_result["forge_variant"],
                "total_indicators": self._forge_detection_result["total_indicators"],
            },
            "signature_breakdown": {
                "primary_signatures": len(self._forge_detection_result["primary_matches"]),
                "secondary_signatures": len(self._forge_detection_result["secondary_matches"]),
                "tertiary_indicators": len(self._forge_detection_result["tertiary_matches"]),
                "parameter_patterns": len(self._forge_detection_result["parameter_matches"]),
            },
            "detected_features": self._analyze_forge_features(),
        }
        
        return analysis
        
    def _analyze_forge_features(self) -> Dict[str, Any]:
        """Analyze specific Forge features detected"""
        features = {
            "has_advanced_scheduling": False,
            "has_hires_modules": False,
            "has_forge_coupling": False,
            "has_regional_prompting": False,
            "feature_count": 0,
        }
        
        if not self._forge_detection_result:
            return features
            
        # Check for advanced scheduling
        features["has_advanced_scheduling"] = any(
            "Schedule type: Automatic" in match 
            for match in self._forge_detection_result["secondary_matches"]
        )
        
        # Check for Hires modules
        features["has_hires_modules"] = any(
            "Hires Module" in match 
            for match in self._forge_detection_result["secondary_matches"]
        )
        
        # Check for Forge coupling
        features["has_forge_coupling"] = any(
            "Forge couple" in match 
            for match in self._forge_detection_result["secondary_matches"]
        )
        
        # Check for regional prompting
        features["has_regional_prompting"] = any(
            "Regional" in match 
            for match in self._forge_detection_result["secondary_matches"]
        )
        
        # Count active features
        features["feature_count"] = sum([
            features["has_advanced_scheduling"],
            features["has_hires_modules"], 
            features["has_forge_coupling"],
            features["has_regional_prompting"],
        ])
        
        return features

    def debug_forge_detection(self) -> Dict[str, Any]:
        """Get detailed debugging information about Forge detection"""
        if not self._forge_detection_result:
            return {"error": "No detection data available"}
            
        debug_info = {
            "raw_text_preview": self._raw[:200] if self._raw else None,
            "raw_text_length": len(self._raw) if self._raw else 0,
            "detection_result": self._forge_detection_result,
            "config_info": {
                "primary_signature_count": len(self.signature_config.PRIMARY_SIGNATURES),
                "secondary_signature_count": len(self.signature_config.SECONDARY_SIGNATURES),
                "tertiary_indicator_count": len(self.signature_config.TERTIARY_INDICATORS),
                "parameter_pattern_count": len(self.signature_config.FORGE_PARAMETER_PATTERNS),
            },
            "extraction_results": {
                "tool_name": self.tool,
                "parameter_count": len(self._parameter),
                "has_prompts": bool(self._positive or self._negative),
                "forge_specific_params": [
                    key for key in self._parameter.keys() 
                    if any(indicator in key.lower() for indicator in ["forge", "hires", "schedule"])
                ],
            }
        }
        
        return debug_info