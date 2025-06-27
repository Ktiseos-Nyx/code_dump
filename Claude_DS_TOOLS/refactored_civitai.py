# dataset_tools/vendored_sdpr/format/civitai.py
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from .a1111 import A1111  # For A1111 text parsing capabilities
from .base_format import BaseFormat


class CivitaiFormat(BaseFormat):
    """
    Handles Civitai-specific metadata formats:
    1. ComfyUI JSON format (workflow with extraMetadata)
    2. A1111 text format (with Civitai resources marker)
    
    Engineered for robustness and maintainability.
    """
    
    tool = "Civitai"  # Base tool name, refined during parsing

    # Mojibake detection patterns - more systematic approach
    MOJIBAKE_INDICATORS = {
        'common_chars': ['笀', '∀', '』', '〉'],
        'patterns': [r'izarea', r'笀.*笀', r'∀.*∀'],
        'encoding_pairs': [
            ('latin-1', 'utf-16le'),
            ('latin-1', 'utf-16be'),
            ('cp1252', 'utf-16le')
        ]
    }
    
    # Parameter mapping for standardization
    PARAMETER_MAPPINGS = {
        'cfg_scale': ['CFG scale', 'cfgScale', 'cfg_scale'],
        'sampler_name': ['sampler', 'sampler_name', 'samplerName'],
        'steps': ['steps', 'sampling_steps'],
        'seed': ['seed', 'Seed'],
        'width': ['width', 'Width'],
        'height': ['height', 'Height']
    }

    def __init__(
        self,
        info: Dict[str, Any] = None,
        raw: str = "",
        width: int = 0,
        height: int = 0,
        **kwargs,
    ):
        super().__init__(info=info, raw=raw, width=width, height=height, **kwargs)
        self.workflow_data: Optional[Dict[str, Any]] = None
        self.civitai_resources_parsed: Optional[Union[List, Dict]] = None
        self._charset_prefix_pattern = re.compile(
            r'^charset\s*=\s*["\']?(UNICODE|UTF-16(?:LE|BE)?)["\']?\s*',
            re.IGNORECASE
        )

    def _detect_and_fix_mojibake(self, text: str) -> Tuple[str, bool]:
        """
        Systematic mojibake detection and repair.
        Returns (fixed_text, was_fixed)
        """
        if not text or not isinstance(text, str):
            return text, False
            
        # Quick heuristic check
        has_mojibake_chars = any(char in text for char in self.MOJIBAKE_INDICATORS['common_chars'])
        has_mojibake_patterns = any(re.search(pattern, text) for pattern in self.MOJIBAKE_INDICATORS['patterns'])
        looks_like_json = text.strip().startswith('{') and text.strip().endswith('}')
        
        if not (has_mojibake_chars or has_mojibake_patterns) or looks_like_json:
            return text, False
            
        self._logger.debug(f"{self.tool}: Mojibake detected, attempting repair")
        
        # Try different encoding pairs
        for source_enc, target_enc in self.MOJIBAKE_INDICATORS['encoding_pairs']:
            try:
                # Encode with source encoding, decode with target
                repaired = text.encode(source_enc, 'replace').decode(target_enc, 'replace')
                repaired = repaired.strip('\x00')  # Remove null chars
                
                # Validate the repair by checking if it's valid JSON
                if repaired.strip().startswith('{') and repaired.strip().endswith('}'):
                    json.loads(repaired)  # This will raise if invalid
                    self._logger.debug(f"{self.tool}: Mojibake repaired using {source_enc} -> {target_enc}")
                    return repaired, True
                    
            except (UnicodeError, json.JSONDecodeError):
                continue
                
        self._logger.debug(f"{self.tool}: Mojibake repair unsuccessful")
        return text, False

    def _clean_user_comment(self, raw_comment: str) -> Optional[str]:
        """
        Clean and decode UserComment string for JSON parsing.
        Handles charset prefixes and mojibake systematically.
        """
        if not raw_comment or not isinstance(raw_comment, str):
            self._logger.warning(f"{self.tool}: Empty or invalid UserComment")
            return None
            
        self._logger.debug(f"{self.tool}: Cleaning UserComment (preview): '{raw_comment[:70]}...'")
        
        # Remove charset prefix if present
        cleaned = raw_comment
        match = self._charset_prefix_pattern.match(raw_comment)
        if match:
            cleaned = raw_comment[len(match.group(0)):].strip()
            self._logger.debug(f"{self.tool}: Removed charset prefix")
            
        # Strip null characters
        cleaned = cleaned.strip('\x00')
        
        # Handle mojibake
        cleaned, was_repaired = self._detect_and_fix_mojibake(cleaned)
        if was_repaired:
            self._logger.debug(f"{self.tool}: Mojibake successfully repaired")
            
        # Final JSON validation
        if cleaned.strip().startswith('{') and cleaned.strip().endswith('}'):
            try:
                json.loads(cleaned)
                return cleaned
            except json.JSONDecodeError as e:
                self._logger.warning(f"{self.tool}: Final JSON validation failed: {e}")
                return None
                
        self._logger.debug(f"{self.tool}: Cleaned text is not valid JSON")
        return None

    def _extract_standardized_parameters(self, data_dict: Dict[str, Any]) -> Tuple[Dict[str, str], set]:
        """
        Extract and standardize parameters using the mapping table.
        Returns (extracted_params, handled_keys)
        """
        extracted = {}
        handled = set()
        
        for standard_key, possible_keys in self.PARAMETER_MAPPINGS.items():
            for key in possible_keys:
                if key in data_dict:
                    value = data_dict[key]
                    if value is not None:
                        extracted[standard_key] = str(value)
                        handled.add(key)
                        break  # Use first match
                        
        return extracted, handled

    def _extract_civitai_specific_data(self, data_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], set]:
        """
        Extract Civitai-specific metadata and resources.
        Returns (civitai_data, handled_keys)
        """
        civitai_data = {}
        handled = set()
        
        # Extract resources
        if 'resources' in data_dict:
            civitai_data['civitai_resources_from_extra'] = data_dict['resources']
            handled.add('resources')
            
        # Extract workflow ID
        if 'workflowId' in data_dict:
            civitai_data['civitai_workflowId_from_extra'] = str(data_dict['workflowId'])
            handled.add('workflowId')
            
        return civitai_data, handled

    def _parse_comfyui_format(self) -> bool:
        """
        Parse Civitai's ComfyUI JSON format.
        Structure: Main workflow JSON with 'extraMetadata' containing parameters.
        """
        if not self._raw:
            return False
            
        # Clean and validate the raw data
        cleaned_json = self._clean_user_comment(self._raw)
        if not cleaned_json:
            return False
            
        try:
            workflow_data = json.loads(cleaned_json)
            if not isinstance(workflow_data, dict):
                self._logger.debug(f"{self.tool}: Workflow data is not a dictionary")
                return False
                
            # Extract extraMetadata
            extra_metadata_str = workflow_data.get('extraMetadata')
            if not isinstance(extra_metadata_str, str):
                self._logger.debug(f"{self.tool}: Missing or invalid extraMetadata")
                return False
                
            extra_metadata = json.loads(extra_metadata_str)
            if not isinstance(extra_metadata, dict):
                self._logger.debug(f"{self.tool}: extraMetadata is not a dictionary")
                return False
                
            # Success! Update tool name and store data
            self.tool = "Civitai ComfyUI"
            self.workflow_data = workflow_data
            self._raw = cleaned_json
            
            # Extract prompts
            self._positive = str(extra_metadata.get('prompt', '')).strip()
            self._negative = str(extra_metadata.get('negativePrompt', '')).strip()
            
            # Extract standardized parameters
            standard_params, handled_standard = self._extract_standardized_parameters(extra_metadata)
            self._parameter.update(standard_params)
            
            # Extract Civitai-specific data
            civitai_params, handled_civitai = self._extract_civitai_specific_data(extra_metadata)
            self._parameter.update(civitai_params)
            
            # Handle dimensions
            all_handled = handled_standard | handled_civitai
            self._extract_and_set_dimensions(extra_metadata, 'width', 'height', all_handled)
            
            # Build settings string from remaining parameters
            self._setting = self._build_settings_string(
                include_standard_params=False,
                custom_settings_dict=None,
                remaining_data_dict=extra_metadata,
                remaining_handled_keys=all_handled,
                sort_parts=True,
            )
            
            self._logger.info(f"{self.tool}: Successfully parsed ComfyUI format")
            return True
            
        except json.JSONDecodeError as e:
            self._logger.debug(f"{self.tool}: JSON parsing failed: {e}")
            return False
        except Exception as e:
            self._logger.warning(f"{self.tool}: Unexpected error during ComfyUI parsing: {e}")
            return False

    def _parse_a1111_format(self) -> bool:
        """
        Parse Civitai's A1111 text format.
        Uses A1111 parser then validates Civitai-specific markers.
        """
        if not self._raw:
            return False
            
        # Use A1111 parser as utility
        a1111_parser = A1111(
            raw=self._raw,
            width=self.width,
            height=self.height,
            logger_obj=self._logger,
        )
        
        if a1111_parser.parse() != self.Status.READ_SUCCESS:
            self._logger.debug(f"{self.tool}: A1111 parsing failed")
            return False
            
        # Parse the settings to check for Civitai markers
        settings_dict = {}
        if a1111_parser.setting:
            settings_dict = a1111_parser._parse_settings_string_to_dict(a1111_parser.setting)
            
        # Look for Civitai-specific markers
        if 'Civitai resources' not in settings_dict:
            self._logger.debug(f"{self.tool}: No Civitai resources marker found")
            return False
            
        # Success! This is Civitai A1111 format
        self.tool = "Civitai A1111"
        
        # Copy data from A1111 parser
        self._positive = a1111_parser.positive
        self._negative = a1111_parser.negative
        self._parameter = a1111_parser.parameter.copy()
        self._setting = a1111_parser.setting
        self._width = a1111_parser.width
        self._height = a1111_parser.height
        
        # Parse Civitai-specific resources
        self._parse_civitai_resources(settings_dict)
        
        self._logger.info(f"{self.tool}: Successfully parsed A1111 format")
        return True

    def _parse_civitai_resources(self, settings_dict: Dict[str, str]) -> None:
        """Parse and store Civitai resources and metadata from A1111 settings."""
        # Parse Civitai resources JSON
        civitai_resources_str = settings_dict.get('Civitai resources')
        if civitai_resources_str:
            try:
                self.civitai_resources_parsed = json.loads(civitai_resources_str)
                self._parameter['civitai_resources_data'] = self.civitai_resources_parsed
            except json.JSONDecodeError:
                self._logger.warning(f"{self.tool}: Failed to parse Civitai resources JSON")
                self._parameter['civitai_resources_raw'] = civitai_resources_str
                
        # Parse Civitai metadata if present
        civitai_metadata_str = settings_dict.get('Civitai metadata')
        if civitai_metadata_str:
            self._parameter['civitai_metadata_raw'] = civitai_metadata_str

    def _process(self) -> None:
        """
        Main processing pipeline for Civitai formats.
        Tries ComfyUI format first, then A1111 format.
        """
        self._logger.info(f"{self.tool}: Starting Civitai format detection")
        
        if not self._raw:
            self._logger.warning(f"{self.tool}: No raw data to process")
            self.status = self.Status.MISSING_INFO
            self._error = "No raw data provided for Civitai parsing"
            return
            
        # Try ComfyUI format first (more specific)
        if self._parse_comfyui_format():
            self.status = self.Status.READ_SUCCESS
            return
            
        # Fallback to A1111 format
        self._logger.debug(f"{self.tool}: ComfyUI format failed, trying A1111 format")
        if self._parse_a1111_format():
            self.status = self.Status.READ_SUCCESS
            return
            
        # Neither format matched
        self._logger.debug(f"{self.tool}: No Civitai format detected")
        self.status = self.Status.FORMAT_DETECTION_ERROR
        self._error = "Data does not match any known Civitai format (ComfyUI JSON or A1111 with markers)"

    def get_format_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the detected format.
        Useful for debugging and format validation.
        """
        info = {
            'detected_format': self.tool,
            'has_workflow_data': self.workflow_data is not None,
            'has_civitai_resources': self.civitai_resources_parsed is not None,
            'parameter_count': len(self._parameter),
            'has_prompts': bool(self._positive or self._negative),
        }
        
        if self.workflow_data:
            info['workflow_keys'] = list(self.workflow_data.keys())
            
        if self.civitai_resources_parsed:
            if isinstance(self.civitai_resources_parsed, list):
                info['resource_count'] = len(self.civitai_resources_parsed)
            elif isinstance(self.civitai_resources_parsed, dict):
                info['resource_keys'] = list(self.civitai_resources_parsed.keys())
                
        return info