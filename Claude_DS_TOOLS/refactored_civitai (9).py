# dataset_tools/vendored_sdpr/format/novelai.py

__author__ = "receyuki"
__filename__ = "novelai.py"
# MODIFIED by Ktiseos Nyx for Dataset-Tools
__copyright__ = "Copyright 2023, Receyuki; Modified 2025, Ktiseos Nyx"
__email__ = "receyuki@gmail.com"

import gzip
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from PIL import Image

from .base_format import BaseFormat


class NovelAIFormat(Enum):
    """NovelAI metadata format types"""
    LEGACY_PNG = "legacy_png"
    STEALTH_PNG = "stealth_png"
    UNKNOWN = "unknown"


@dataclass
class NovelAIConfig:
    """Configuration for NovelAI format parsing - systematically organized"""
    
    # Parameter mapping for NovelAI to standard names
    PARAMETER_MAPPINGS: Dict[str, Union[str, List[str]]] = field(default_factory=lambda: {
        "sampler": "sampler_name",
        "seed": "seed",
        "strength": "denoising_strength",
        "noise": "noise_offset",
        "scale": "cfg_scale",
        "steps": "steps",
        "sm": "sm_value",
        "sm_dyn": "sm_dynamic",
        "dynamic_thresholding": "dynamic_thresholding",
        "cfg_rescale": "cfg_rescale",
        "height": "height",
        "width": "width",
    })
    
    # Keys to exclude from settings string
    EXCLUDED_SETTINGS_KEYS: Set[str] = field(default_factory=lambda: {
        "uc", "prompt", "Description", "Comment", "width", "height"
    })
    
    # LSB extraction constraints
    LSB_CONSTRAINTS: Dict[str, Any] = field(default_factory=lambda: {
        "max_data_length": 10 * 1024 * 1024,  # 10MB max
        "min_data_length": 1,
        "required_image_mode": "RGBA",
        "min_alpha_channels": 4,
    })
    
    # Legacy PNG identification keys
    LEGACY_IDENTIFICATION_KEYS: Set[str] = field(default_factory=lambda: {
        "Software", "Description", "Comment"
    })


class LSBExtractor:
    """
    Advanced LSB (Least Significant Bit) extractor for NovelAI stealth PNGs.
    
    Extracts hidden metadata from the alpha channel of RGBA images using
    steganographic techniques.
    """
    
    def __init__(self, image: Image.Image, logger: Optional[logging.Logger] = None):
        self.image = image
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize extraction state
        self.pixel_data: List[Tuple[int, int, int, int]] = []
        self.width: int = 0
        self.height: int = 0
        self.lsb_bytes: bytearray = bytearray()
        self.byte_cursor: int = 0
        self.extraction_successful: bool = False
        
        # Perform extraction
        self._extract_lsb_data()
        
    def _extract_lsb_data(self) -> None:
        """Extract LSB data from image with comprehensive error handling"""
        try:
            # Validate image format
            if not self._validate_image_format():
                return
                
            # Get pixel data
            if not self._extract_pixel_data():
                return
                
            # Extract LSB bits
            if not self._process_lsb_bits():
                return
                
            self.extraction_successful = True
            self.logger.debug(f"NovelAI LSB: Successfully extracted {len(self.lsb_bytes)} bytes")
            
        except Exception as e:
            self.logger.error(f"NovelAI LSB: Extraction failed: {e}")
            self.extraction_successful = False
            
    def _validate_image_format(self) -> bool:
        """Validate that the image is suitable for LSB extraction"""
        if not self.image:
            self.logger.warning("NovelAI LSB: No image provided")
            return False
            
        if self.image.mode != "RGBA":
            self.logger.warning(f"NovelAI LSB: Image mode '{self.image.mode}' not RGBA")
            return False
            
        self.width, self.height = self.image.size
        if self.width <= 0 or self.height <= 0:
            self.logger.warning(f"NovelAI LSB: Invalid image dimensions: {self.width}x{self.height}")
            return False
            
        return True
        
    def _extract_pixel_data(self) -> bool:
        """Extract and validate pixel data from image"""
        try:
            raw_data = list(self.image.getdata())
            
            if not raw_data:
                self.logger.warning("NovelAI LSB: No pixel data extracted")
                return False
                
            # Validate pixel format
            if not isinstance(raw_data[0], (tuple, list)) or len(raw_data[0]) < 4:
                self.logger.warning("NovelAI LSB: Pixel data format invalid (expected RGBA tuples)")
                return False
                
            self.pixel_data = raw_data
            self.logger.debug(f"NovelAI LSB: Extracted {len(self.pixel_data)} pixels")
            return True
            
        except Exception as e:
            self.logger.error(f"NovelAI LSB: Failed to extract pixel data: {e}")
            return False
            
    def _process_lsb_bits(self) -> bool:
        """Process LSB bits from alpha channel"""
        try:
            current_byte = 0
            bit_count = 0
            
            for pixel_index, pixel in enumerate(self.pixel_data):
                try:
                    alpha_value = pixel[3]  # Alpha channel
                    lsb = alpha_value & 1   # Extract least significant bit
                    
                    current_byte = (current_byte << 1) | lsb
                    bit_count += 1
                    
                    if bit_count == 8:
                        self.lsb_bytes.append(current_byte)
                        current_byte = 0
                        bit_count = 0
                        
                except (IndexError, TypeError) as e:
                    self.logger.debug(f"NovelAI LSB: Skipping malformed pixel {pixel_index}: {e}")
                    continue
                    
            if bit_count > 0:
                # Handle remaining bits if not complete byte
                current_byte <<= (8 - bit_count)
                self.lsb_bytes.append(current_byte)
                
            return len(self.lsb_bytes) > 0
            
        except Exception as e:
            self.logger.error(f"NovelAI LSB: Failed to process LSB bits: {e}")
            return False

    def get_next_bytes(self, num_bytes: int) -> Optional[bytes]:
        """Get next n bytes from extracted LSB data"""
        if not self.extraction_successful:
            return None
            
        if self.byte_cursor + num_bytes > len(self.lsb_bytes):
            self.logger.debug(f"NovelAI LSB: Requested {num_bytes} bytes, only {len(self.lsb_bytes) - self.byte_cursor} available")
            return None
            
        result = bytes(self.lsb_bytes[self.byte_cursor:self.byte_cursor + num_bytes])
        self.byte_cursor += num_bytes
        return result

    def read_uint32_big_endian(self) -> Optional[int]:
        """Read 32-bit unsigned integer in big-endian format"""
        byte_data = self.get_next_bytes(4)
        if byte_data and len(byte_data) == 4:
            return int.from_bytes(byte_data, byteorder="big")
        return None

    def get_extraction_info(self) -> Dict[str, Any]:
        """Get detailed information about the LSB extraction"""
        return {
            "extraction_successful": self.extraction_successful,
            "image_dimensions": f"{self.width}x{self.height}",
            "pixel_count": len(self.pixel_data),
            "extracted_bytes": len(self.lsb_bytes),
            "bytes_remaining": len(self.lsb_bytes) - self.byte_cursor,
            "cursor_position": self.byte_cursor,
        }


class NovelAIFormatDetector:
    """Detects NovelAI format type and validates structure"""
    
    def __init__(self, config: NovelAIConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def detect_format(self, info_data: Optional[Dict[str, Any]], 
                     extractor: Optional[LSBExtractor]) -> Tuple[NovelAIFormat, Dict[str, Any]]:
        """
        Detect NovelAI format type and return detection details.
        Returns (format_type, detection_info)
        """
        detection_info = {
            "detected_format": NovelAIFormat.UNKNOWN,
            "confidence_score": 0.0,
            "legacy_indicators": {},
            "stealth_indicators": {},
            "validation_errors": [],
        }
        
        # Check for legacy PNG format
        legacy_score, legacy_info = self._check_legacy_format(info_data)
        detection_info["legacy_indicators"] = legacy_info
        
        # Check for stealth PNG format
        stealth_score, stealth_info = self._check_stealth_format(extractor)
        detection_info["stealth_indicators"] = stealth_info
        
        # Determine format based on scores
        if legacy_score > stealth_score and legacy_score > 0.5:
            detection_info["detected_format"] = NovelAIFormat.LEGACY_PNG
            detection_info["confidence_score"] = legacy_score
        elif stealth_score > 0.5:
            detection_info["detected_format"] = NovelAIFormat.STEALTH_PNG
            detection_info["confidence_score"] = stealth_score
        else:
            detection_info["validation_errors"].append("No NovelAI format detected")
            
        self.logger.debug(f"NovelAI detection: {detection_info['detected_format'].value}, "
                         f"confidence: {detection_info['confidence_score']:.2f}")
        
        return detection_info["detected_format"], detection_info
        
    def _check_legacy_format(self, info_data: Optional[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """Check for legacy PNG format indicators"""
        legacy_info = {
            "has_software_tag": False,
            "software_value": None,
            "has_description": False,
            "has_comment": False,
            "comment_is_json": False,
        }
        
        score = 0.0
        
        if not info_data:
            return score, legacy_info
            
        # Check Software tag
        software = info_data.get("Software", "")
        if software:
            legacy_info["has_software_tag"] = True
            legacy_info["software_value"] = software
            
            if "NovelAI" in str(software):
                score += 0.8  # Strong indicator
            else:
                score += 0.1  # Weak indicator
                
        # Check Description
        if info_data.get("Description"):
            legacy_info["has_description"] = True
            score += 0.2
            
        # Check Comment and validate JSON
        comment = info_data.get("Comment", "")
        if comment:
            legacy_info["has_comment"] = True
            score += 0.2
            
            try:
                parsed = json.loads(comment)
                if isinstance(parsed, dict):
                    legacy_info["comment_is_json"] = True
                    score += 0.3
            except json.JSONDecodeError:
                pass
                
        return score, legacy_info
        
    def _check_stealth_format(self, extractor: Optional[LSBExtractor]) -> Tuple[float, Dict[str, Any]]:
        """Check for stealth PNG format indicators"""
        stealth_info = {
            "extractor_available": False,
            "extraction_successful": False,
            "has_valid_header": False,
            "data_length": 0,
            "compression_valid": False,
        }
        
        score = 0.0
        
        if not extractor:
            return score, stealth_info
            
        stealth_info["extractor_available"] = True
        score += 0.1
        
        if not extractor.extraction_successful:
            return score, stealth_info
            
        stealth_info["extraction_successful"] = True
        score += 0.3
        
        # Try to read and validate header
        original_cursor = extractor.byte_cursor
        try:
            data_length = extractor.read_uint32_big_endian()
            if data_length and self._is_valid_data_length(data_length):
                stealth_info["has_valid_header"] = True
                stealth_info["data_length"] = data_length
                score += 0.4
                
                # Try to read some compressed data
                test_data = extractor.get_next_bytes(min(100, data_length))
                if test_data and self._test_gzip_header(test_data):
                    stealth_info["compression_valid"] = True
                    score += 0.4
                    
        except Exception as e:
            self.logger.debug(f"NovelAI stealth validation error: {e}")
        finally:
            # Reset cursor
            extractor.byte_cursor = original_cursor
            
        return score, stealth_info
        
    def _is_valid_data_length(self, length: int) -> bool:
        """Validate data length is within reasonable bounds"""
        return (self.config.LSB_CONSTRAINTS["min_data_length"] <= 
                length <= 
                self.config.LSB_CONSTRAINTS["max_data_length"])
        
    def _test_gzip_header(self, data: bytes) -> bool:
        """Test if data starts with valid gzip header"""
        return len(data) >= 2 and data[0] == 0x1f and data[1] == 0x8b


class NovelAIDataExtractor:
    """Handles extraction of NovelAI data from different formats"""
    
    def __init__(self, config: NovelAIConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def extract_legacy_data(self, info_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from legacy PNG format"""
        result = {
            "positive": "",
            "negative": "",
            "parameters": {},
            "raw_data": "",
            "extraction_errors": [],
        }
        
        try:
            # Extract positive prompt from Description
            result["positive"] = str(info_data.get("Description", "")).strip()
            
            # Parse Comment JSON for negative prompt and parameters
            comment_str = info_data.get("Comment", "{}")
            comment_data = self._parse_comment_json(comment_str)
            
            if comment_data:
                result["negative"] = str(comment_data.get("uc", "")).strip()
                
                # Extract parameters
                parameters = self._extract_parameters(comment_data)
                result["parameters"] = parameters
                
            # Build raw data representation
            raw_parts = []
            if result["positive"]:
                raw_parts.append(result["positive"])
            if source := info_data.get("Source"):
                raw_parts.append(f"Source: {source}")
            raw_parts.append(f"Comment: {comment_str}")
            
            result["raw_data"] = "\n".join(filter(None, raw_parts))
            
            self.logger.debug(f"NovelAI legacy: Extracted {len(result['parameters'])} parameters")
            
        except Exception as e:
            self.logger.error(f"NovelAI legacy extraction error: {e}")
            result["extraction_errors"].append(f"Legacy extraction failed: {e}")
            
        return result
        
    def extract_stealth_data(self, extractor: LSBExtractor) -> Dict[str, Any]:
        """Extract data from stealth PNG format"""
        result = {
            "positive": "",
            "negative": "",
            "parameters": {},
            "raw_data": "",
            "extraction_errors": [],
        }
        
        try:
            # Read data length
            data_length = extractor.read_uint32_big_endian()
            if not data_length or not self._is_valid_length(data_length):
                result["extraction_errors"].append(f"Invalid data length: {data_length}")
                return result
                
            # Read compressed data
            compressed_data = extractor.get_next_bytes(data_length)
            if not compressed_data or len(compressed_data) != data_length:
                result["extraction_errors"].append(f"Failed to read {data_length} bytes")
                return result
                
            # Decompress and parse JSON
            try:
                json_string = gzip.decompress(compressed_data).decode("utf-8")
                result["raw_data"] = json_string
                
                main_data = json.loads(json_string)
                if not isinstance(main_data, dict):
                    result["extraction_errors"].append("Decompressed data is not a JSON object")
                    return result
                    
            except gzip.BadGzipFile as e:
                result["extraction_errors"].append(f"Invalid gzip data: {e}")
                return result
            except json.JSONDecodeError as e:
                result["extraction_errors"].append(f"Invalid JSON data: {e}")
                return result
                
            # Extract prompts and parameters
            data_to_use = self._select_data_source(main_data)
            
            result["positive"] = str(data_to_use.get("prompt", 
                                   main_data.get("Description", ""))).strip()
            result["negative"] = str(data_to_use.get("uc", "")).strip()
            
            # Extract parameters
            parameters = self._extract_parameters(data_to_use)
            result["parameters"] = parameters
            
            self.logger.debug(f"NovelAI stealth: Extracted {len(result['parameters'])} parameters")
            
        except Exception as e:
            self.logger.error(f"NovelAI stealth extraction error: {e}")
            result["extraction_errors"].append(f"Stealth extraction failed: {e}")
            
        return result
        
    def _parse_comment_json(self, comment_str: str) -> Optional[Dict[str, Any]]:
        """Parse Comment field JSON with error handling"""
        if not comment_str:
            return None
            
        try:
            parsed = json.loads(comment_str)
            if isinstance(parsed, dict):
                return parsed
            else:
                self.logger.warning(f"NovelAI: Comment JSON is not an object: {type(parsed)}")
        except json.JSONDecodeError as e:
            self.logger.warning(f"NovelAI: Invalid Comment JSON: {e}")
            
        return None
        
    def _select_data_source(self, main_data: Dict[str, Any]) -> Dict[str, Any]:
        """Select the appropriate data source for parameter extraction"""
        # Check for nested Comment structure
        if "Comment" in main_data:
            comment_str = str(main_data.get("Comment", "{}"))
            comment_data = self._parse_comment_json(comment_str)
            
            if comment_data and isinstance(comment_data, dict):
                self.logger.debug("NovelAI stealth: Using nested Comment data")
                return comment_data
                
        self.logger.debug("NovelAI stealth: Using main data")
        return main_data
        
    def _extract_parameters(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Extract and standardize parameters from data"""
        parameters = {}
        
        for nai_key, standard_key in self.config.PARAMETER_MAPPINGS.items():
            if nai_key in data and data[nai_key] is not None:
                if isinstance(standard_key, list):
                    target_key = standard_key[0] if standard_key else nai_key
                else:
                    target_key = standard_key
                    
                processed_value = self._process_parameter_value(nai_key, data[nai_key])
                if processed_value:
                    parameters[target_key] = processed_value
                    
        return parameters
        
    def _process_parameter_value(self, key: str, value: Any) -> str:
        """Process parameter values with key-specific logic"""
        if value is None:
            return ""
            
        # Handle boolean values
        if isinstance(value, bool):
            return str(value).lower()
            
        # Handle numeric values
        if isinstance(value, (int, float)):
            return str(value)
            
        # Handle string values
        return str(value).strip()
        
    def _is_valid_length(self, length: int) -> bool:
        """Validate data length is within constraints"""
        return (self.config.LSB_CONSTRAINTS["min_data_length"] <= 
                length <= 
                self.config.LSB_CONSTRAINTS["max_data_length"])


class NovelAI(BaseFormat):
    """
    Enhanced NovelAI format parser with support for both legacy and stealth formats.
    
    Supports two NovelAI metadata formats:
    1. Legacy PNG: Standard PNG metadata fields (Software, Description, Comment)
    2. Stealth PNG: LSB-encoded compressed JSON data in alpha channel
    """
    
    tool = "NovelAI"
    
    def __init__(
        self,
        info: Optional[Dict[str, Any]] = None,
        raw: str = "",
        extractor: Optional[LSBExtractor] = None,
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
        self.config = NovelAIConfig()
        self.format_detector = NovelAIFormatDetector(self.config, self._logger)
        self.data_extractor = NovelAIDataExtractor(self.config, self._logger)
        
        # Store extractor and detection results
        self._extractor = extractor
        self._detected_format: NovelAIFormat = NovelAIFormat.UNKNOWN
        self._detection_info: Dict[str, Any] = {}
        self._extraction_result: Dict[str, Any] = {}

    def _process(self) -> None:
        """Main processing pipeline for NovelAI formats"""
        self._logger.debug(f"{self.tool}: Starting NovelAI format processing")
        
        # Detect format type
        self._detected_format, self._detection_info = self.format_detector.detect_format(
            self._info, self._extractor
        )
        
        if self._detected_format == NovelAIFormat.UNKNOWN:
            self._handle_unknown_format()
            return
            
        # Extract data based on detected format
        try:
            success = self._extract_by_format()
            if not success:
                return  # Error already set
                
        except Exception as e:
            self._logger.error(f"{self.tool}: Extraction failed: {e}")
            self.status = self.Status.FORMAT_ERROR
            self._error = f"NovelAI extraction failed: {e}"
            return
            
        # Apply extraction results
        self._apply_extraction_results()
        
        # Validate extraction success
        if not self._has_meaningful_extraction():
            self._logger.warning(f"{self.tool}: No meaningful data extracted")
            self.status = self.Status.FORMAT_ERROR
            self._error = "NovelAI parsing yielded no meaningful data"
            return
            
        self._logger.info(f"{self.tool}: Successfully parsed {self._detected_format.value} format")

    def _handle_unknown_format(self) -> None:
        """Handle cases where no NovelAI format is detected"""
        errors = self._detection_info.get("validation_errors", [])
        
        if not self._info and not self._extractor:
            self.status = self.Status.MISSING_INFO
            self._error = "No data source provided for NovelAI parser"
        elif self._extractor and not self._extractor.extraction_successful:
            self.status = self.Status.FORMAT_ERROR
            self._error = "LSB extraction failed for stealth PNG"
        else:
            self.status = self.Status.FORMAT_DETECTION_ERROR
            self._error = f"No NovelAI format detected: {'; '.join(errors)}"
            
        self._logger.debug(f"{self.tool}: {self._error}")

    def _extract_by_format(self) -> bool:
        """Extract data based on detected format"""
        if self._detected_format == NovelAIFormat.LEGACY_PNG:
            if not self._info:
                self.status = self.Status.FORMAT_ERROR
                self._error = "Legacy PNG format detected but no info data available"
                return False
            self._extraction_result = self.data_extractor.extract_legacy_data(self._info)
            
        elif self._detected_format == NovelAIFormat.STEALTH_PNG:
            if not self._extractor:
                self.status = self.Status.FORMAT_ERROR
                self._error = "Stealth PNG format detected but no LSB extractor available"
                return False
            self._extraction_result = self.data_extractor.extract_stealth_data(self._extractor)
            
        else:
            self.status = self.Status.FORMAT_ERROR
            self._error = f"Unknown format for extraction: {self._detected_format}"
            return False
            
        # Check for extraction errors
        if self._extraction_result.get("extraction_errors"):
            errors = self._extraction_result["extraction_errors"]
            self._logger.warning(f"{self.tool}: Extraction errors: {errors}")
            if not self._extraction_result.get("parameters") and not self._extraction_result.get("positive"):
                self.status = self.Status.FORMAT_ERROR
                self._error = f"Extraction failed: {'; '.join(errors)}"
                return False
                
        return True

    def _apply_extraction_results(self) -> None:
        """Apply extraction results to instance variables"""
        if not self._extraction_result:
            return
            
        # Apply prompts
        self._positive = self._extraction_result.get("positive", "")
        self._negative = self._extraction_result.get("negative", "")
        
        # Apply parameters
        parameters = self._extraction_result.get("parameters", {})
        self._parameter.update(parameters)
        
        # Handle dimensions
        if "width" in parameters:
            try:
                self._width = str(int(parameters["width"]))
            except (ValueError, TypeError):
                pass
        if "height" in parameters:
            try:
                self._height = str(int(parameters["height"]))
            except (ValueError, TypeError):
                pass
                
        # Update parameter dict with final dimensions
        if self._width != "0":
            self._parameter["width"] = self._width
        if self._height != "0":
            self._parameter["height"] = self._height
        if self._width != "0" and self._height != "0":
            self._parameter["size"] = f"{self._width}x{self._height}"
            
        # Set raw data
        if not self._raw and "raw_data" in self._extraction_result:
            self._raw = self._extraction_result["raw_data"]
            
        # Build settings string
        self._build_novelai_settings()

    def _build_novelai_settings(self) -> None:
        """Build settings string from extracted parameters"""
        # Use the config to determine which keys to exclude
        handled_keys = set(self.config.PARAMETER_MAPPINGS.keys())
        handled_keys.update(self.config.EXCLUDED_SETTINGS_KEYS)
        
        # For NovelAI, we'll build from the original data if available
        if self._detected_format == NovelAIFormat.LEGACY_PNG and self._info:
            comment_str = self._info.get("Comment", "{}")
            try:
                comment_data = json.loads(comment_str)
                if isinstance(comment_data, dict):
                    self._setting = self._build_settings_string(
                        remaining_data_dict=comment_data,
                        remaining_handled_keys=handled_keys,
                        include_standard_params=True,
                        sort_parts=True,
                    )
            except json.JSONDecodeError:
                pass
        else:
            # Build from standard parameters
            self._setting = self._build_settings_string(
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
        """Get detailed information about the detected and parsed format"""
        return {
            "detected_format": self._detected_format.value,
            "detection_info": self._detection_info,
            "extraction_result": {
                "has_positive_prompt": bool(self._positive),
                "has_negative_prompt": bool(self._negative),
                "parameter_count": len([v for v in self._parameter.values() if v and v != self.DEFAULT_PARAMETER_PLACEHOLDER]),
                "extraction_errors": self._extraction_result.get("extraction_errors", []) if self._extraction_result else [],
            },
            "lsb_info": self._extractor.get_extraction_info() if self._extractor else None,
            "has_dimensions": self._width != "0" or self._height != "0",
            "dimensions": f"{self._width}x{self._height}" if self._width != "0" and self._height != "0" else None,
        }

    def debug_novelai_detection(self) -> Dict[str, Any]:
        """Get comprehensive debugging information about NovelAI detection"""
        return {
            "input_summary": {
                "has_info": bool(self._info),
                "info_keys": list(self._info.keys()) if self._info else [],
                "has_extractor": bool(self._extractor),
                "extractor_info": self._extractor.get_extraction_info() if self._extractor else None,
                "has_raw": bool(self._raw),
            },
            "detection_details": self._detection_info,
            "extraction_details": self._extraction_result,
            "format_analysis": {
                "detected_format": self._detected_format.value,
                "confidence_score": self._detection_info.get("confidence_score", 0.0),
                "format_specific_info": self._get_format_specific_debug_info(),
            },
            "config_info": {
                "parameter_mappings": len(self.config.PARAMETER_MAPPINGS),
                "excluded_keys": list(self.config.EXCLUDED_SETTINGS_KEYS),
                "lsb_constraints": self.config.LSB_CONSTRAINTS,
            }
        }

    def _get_format_specific_debug_info(self) -> Dict[str, Any]:
        """Get debug info specific to the detected format"""
        if self._detected_format == NovelAIFormat.LEGACY_PNG:
            return {
                "format_type": "Legacy PNG",
                "software_tag": self._info.get("Software") if self._info else None,
                "has_description": bool(self._info.get("Description")) if self._info else False,
                "has_comment": bool(self._info.get("Comment")) if self._info else False,
            }
        elif self._detected_format == NovelAIFormat.STEALTH_PNG:
            return {
                "format_type": "Stealth PNG",
                "lsb_extraction": self._extractor.get_extraction_info() if self._extractor else None,
                "data_validation": {
                    "has_valid_header": self._detection_info.get("stealth_indicators", {}).get("has_valid_header", False),
                    "compression_valid": self._detection_info.get("stealth_indicators", {}).get("compression_valid", False),
                    "data_length": self._detection_info.get("stealth_indicators", {}).get("data_length", 0),
                }
            }
        else:
            return {"format_type": "Unknown"}

    @staticmethod
    def create_lsb_extractor(image_path_or_object: Union[str, Image.Image], 
                           logger: Optional[logging.Logger] = None) -> Optional[LSBExtractor]:
        """
        Factory method to create LSB extractor from image path or PIL Image object.
        Returns None if extraction fails.
        """
        try:
            if isinstance(image_path_or_object, str):
                image = Image.open(image_path_or_object)
            elif isinstance(image_path_or_object, Image.Image):
                image = image_path_or_object
            else:
                if logger:
                    logger.error("NovelAI: Invalid image input type")
                return None
                
            extractor = LSBExtractor(image, logger)
            if extractor.extraction_successful:
                return extractor
            else:
                if logger:
                    logger.warning("NovelAI: LSB extraction failed")
                return None
                
        except Exception as e:
            if logger:
                logger.error(f"NovelAI: Failed to create LSB extractor: {e}")
            return None

    def validate_stealth_png_capability(self, image: Image.Image) -> Dict[str, Any]:
        """
        Validate if an image can support stealth PNG analysis.
        Returns validation results without performing full extraction.
        """
        validation = {
            "can_extract": False,
            "image_format_valid": False,
            "dimensions_valid": False,
            "mode_correct": False,
            "estimated_capacity": 0,
            "issues": []
        }
        
        try:
            # Check image mode
            if image.mode == "RGBA":
                validation["mode_correct"] = True
            else:
                validation["issues"].append(f"Image mode '{image.mode}' is not RGBA")
                
            # Check dimensions
            width, height = image.size
            if width > 0 and height > 0:
                validation["dimensions_valid"] = True
                # Estimate LSB capacity (1 bit per pixel)
                validation["estimated_capacity"] = (width * height) // 8  # bytes
            else:
                validation["issues"].append(f"Invalid dimensions: {width}x{height}")
                
            # Overall validation
            validation["image_format_valid"] = validation["mode_correct"] and validation["dimensions_valid"]
            validation["can_extract"] = validation["image_format_valid"]
            
        except Exception as e:
            validation["issues"].append(f"Validation error: {e}")
            
        return validation