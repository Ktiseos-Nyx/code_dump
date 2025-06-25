# dataset_tools/metadata_engine.py

import json
import logging
import re
from pathlib import Path
from typing import (Any, BinaryIO, Callable, Dict,  # Use Dict, List, Optional explicitly
                    List, Optional, Type)

import piexif
import piexif.helper
from PIL import Image, UnidentifiedImageError

# Assuming these are in the same package or accessible
from dataset_tools.logger import get_logger
from .vendored_sdpr.format.base_format import BaseFormat # For Python class fallback

# --- Parser Class Registry (for Python class fallback) ---
_PARSER_CLASS_REGISTRY: Dict[str, Type[BaseFormat]] = {}

def register_parser_class(name: str, cls: Type[BaseFormat]):
    _PARSER_CLASS_REGISTRY[name] = cls

def get_parser_class_by_name(name: str) -> Optional[Type[BaseFormat]]:
    return _PARSER_CLASS_REGISTRY.get(name)

class MetadataEngine:
    def __init__(self, parser_definitions_path: str | Path, logger_obj: Optional[logging.Logger] = None):
        self.parser_definitions_path = Path(parser_definitions_path)
        if logger_obj:
            self.logger = logger_obj
        else:
            self.logger = get_logger("MetadataEngine")

        self.parser_definitions: List[Dict[str, Any]] = self._load_parser_definitions()
        self.sorted_definitions: List[Dict[str, Any]] = sorted(
            self.parser_definitions,
            key=lambda p: p.get("priority", 0),
            reverse=True
        )
        self.logger.info(f"Loaded {len(self.sorted_definitions)} parser definitions from {self.parser_definitions_path}")

    def _load_parser_definitions(self) -> List[Dict[str, Any]]:
        definitions: List[Dict[str, Any]] = []
        if not self.parser_definitions_path.is_dir():
            self.logger.error(f"Parser definitions path is not a directory: {self.parser_definitions_path}")
            return definitions

        for filepath in self.parser_definitions_path.glob("*.json"):
            self.logger.debug(f"Loading parser definition from: {filepath.name}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    definition = json.load(f)
                    if "parser_name" in definition:
                        definitions.append(definition)
                    else:
                        self.logger.warning(f"Skipping invalid parser definition (missing parser_name): {filepath.name}")
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to decode JSON from {filepath.name}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error loading parser definition {filepath.name}: {e}")
        return definitions

    def _prepare_context_data(self, file_path_or_obj: str | Path | BinaryIO) -> Optional[Dict[str, Any]]:
        context: Dict[str, Any] = {
            "pil_info": {}, "exif_dict": {}, "xmp_string": None,
            "parsed_xmp_dict": None, "png_chunks": {},
            "file_format": "", "width": 0, "height": 0,
            "raw_user_comment_str": None, "software_tag": None,
            "file_extension": "", "raw_file_content_text": None,
            "raw_file_content_bytes": None, "parsed_root_json_object": None,
            "safetensors_metadata": None, "safetensors_main_header": None,
            "gguf_metadata": None, "gguf_main_header": None,
            "file_path_original": str(file_path_or_obj.name if hasattr(file_path_or_obj, "name") and file_path_or_obj.name else str(file_path_or_obj))
        }
        is_binary_io = hasattr(file_path_or_obj, "read") and hasattr(file_path_or_obj, "seek")
        original_file_path_str = context["file_path_original"] # Use consistent path string

        try:
            img = Image.open(file_path_or_obj)
            context["pil_info"] = img.info.copy() if img.info else {}
            context["width"] = img.width
            context["height"] = img.height
            context["file_format"] = img.format.upper() if img.format else ""
            image_filename = getattr(img, 'filename', None)
            if image_filename: # PIL might not always set filename if opened from BytesIO
                 context["file_extension"] = Path(image_filename).suffix.lstrip('.').lower()
            elif isinstance(file_path_or_obj, (str, Path)):
                 context["file_extension"] = Path(file_path_or_obj).suffix.lstrip('.').lower()


            if exif_bytes := context["pil_info"].get("exif"):
                try:
                    loaded_exif = piexif.load(exif_bytes)
                    context["exif_dict"] = loaded_exif
                    uc_bytes = loaded_exif.get("Exif", {}).get(piexif.ExifIFD.UserComment)
                    if uc_bytes:
                        context["raw_user_comment_str"] = piexif.helper.UserComment.load(uc_bytes)
                    sw_bytes = loaded_exif.get("0th", {}).get(piexif.ImageIFD.Software)
                    if sw_bytes and isinstance(sw_bytes, bytes):
                        context["software_tag"] = sw_bytes.decode('ascii', 'ignore').strip('\x00').strip()
                except Exception as e_exif:
                    self.logger.debug(f"piexif failed to load EXIF or extract specific tags: {e_exif}")

            if xmp_str := context["pil_info"].get("XML:com.adobe.xmp"): # PNG XMP
                context["xmp_string"] = xmp_str
                # TODO: Add self._parse_xmp_string_to_dict(xmp_str) call here

            for key, val in context["pil_info"].items():
                if isinstance(val, str): context["png_chunks"][key] = val
            if "UserComment" in context["pil_info"] and "UserComment" not in context["png_chunks"]:
                context["png_chunks"]["UserComment"] = context["pil_info"]["UserComment"]
            img.close()

        except FileNotFoundError:
            self.logger.error(f"File not found: {original_file_path_str}")
            return None
        except UnidentifiedImageError:
            self.logger.info(f"Cannot identify as image: {original_file_path_str}. Checking for non-image types.")
            p = Path(original_file_path_str)
            context["file_extension"] = p.suffix.lstrip('.').lower()
            context["file_format"] = context["file_extension"].upper()

            # Define a helper to read from file_path_or_obj or path p
            def read_file_content(mode='r', encoding='utf-8', errors='replace'):
                if is_binary_io:
                    file_path_or_obj.seek(0)
                    content = file_path_or_obj.read()
                    if 'b' in mode: return content
                    return content.decode(encoding, errors=errors) if isinstance(content, bytes) else content
                else:
                    with open(p, mode, encoding=encoding if 'b' not in mode else None, errors=errors if 'b' not in mode else None) as f_obj:
                        return f_obj.read()

            if context["file_extension"] == "json":
                try:
                    content_str = read_file_content(mode='r', encoding='utf-8')
                    context["parsed_root_json_object"] = json.loads(content_str)
                    context["raw_file_content_text"] = content_str
                except (json.JSONDecodeError, OSError, UnicodeDecodeError, TypeError) as e_json_direct:
                    self.logger.error(f"Failed to read/parse direct JSON file {p.name}: {e_json_direct}")
                    if not context["raw_file_content_text"]: # Try to get raw text if JSON parse failed
                        try: context["raw_file_content_text"] = read_file_content(mode='r', encoding='utf-8', errors='replace')
                        except Exception: pass
            elif context["file_extension"] == "txt":
                try:
                    context["raw_file_content_text"] = read_file_content(mode='r', encoding='utf-8', errors='replace')
                except (OSError, UnicodeDecodeError, TypeError) as e_txt:
                    self.logger.error(f"Failed to read TXT file {p.name}: {e_txt}")
                    return None
            elif context["file_extension"] == "safetensors":
                try:
                    from dataset_tools.model_parsers.safetensors_parser import SafetensorsParser # Lazy
                    temp_parser = SafetensorsParser(original_file_path_str)
                    status = temp_parser.parse()
                    if status == temp_parser.status.SUCCESS:
                        context["safetensors_metadata"] = temp_parser.metadata_header
                        context["safetensors_main_header"] = temp_parser.main_header
                    elif status == temp_parser.status.FAILURE:
                         self.logger.warning(f"Safetensors parser failed for {p.name}: {temp_parser._error_message}")
                except ImportError: self.logger.error("SafetensorsParser not available.")
                except Exception as e_st: self.logger.error(f"Error during safetensors context prep: {e_st}")

            elif context["file_extension"] == "gguf":
                try:
                    from dataset_tools.model_parsers.gguf_parser import GGUFParser # Lazy
                    temp_parser = GGUFParser(original_file_path_str)
                    status = temp_parser.parse()
                    if status == temp_parser.status.SUCCESS:
                        context["gguf_metadata"] = temp_parser.metadata_header
                        context["gguf_main_header"] = temp_parser.main_header
                    elif status == temp_parser.status.FAILURE:
                        self.logger.warning(f"GGUF parser failed for {p.name}: {temp_parser._error_message}")
                except ImportError: self.logger.error("GGUFParser not available.")
                except Exception as e_gguf: self.logger.error(f"Error during GGUF context prep: {e_gguf}")
            else:
                self.logger.info(f"File {p.name} extension '{context['file_extension']}' not specifically handled in non-image path.")
                # Try to get raw bytes as a last resort for generic binary
                try:
                    context["raw_file_content_bytes"] = read_file_content(mode='rb')
                except Exception: pass


        except Exception as e: # Catch-all for other errors during initial processing
            self.logger.error(f"Error preparing context data for {original_file_path_str}: {e}", exc_info=True)
            return None

        self.logger.debug(f"Prepared context for {original_file_path_str}. Keys: {list(k for k,v in context.items() if v is not None)}")
        return context

    def _json_path_get(self, data_container: Any, path_str: Optional[str]) -> Any:
        if not path_str: return data_container
        keys = path_str.split('.')
        current = data_container
        for key_part in keys:
            if current is None: return None
            match = re.fullmatch(r"(\w+)\[(\d+)\]", key_part) # e.g. Options[0]
            if match:
                array_key, index_str = match.groups()
                index = int(index_str)
                if not isinstance(current, dict) or array_key not in current or not isinstance(current[array_key], list) or index >= len(current[array_key]):
                    return None
                current = current[array_key][index]
            elif key_part.startswith('[') and key_part.endswith(']'): # e.g. [0]
                 if not isinstance(current, list): return None
                 try:
                    index = int(key_part[1:-1])
                    if index >= len(current): return None
                    current = current[index]
                 except ValueError: return None
            elif isinstance(current, dict) and key_part in current:
                current = current[key_part]
            else:
                return None
        return current

    def _evaluate_detection_rule(self, rule: Dict[str, Any], context_data: Dict[str, Any]) -> bool:
        # (Logic from previous response, expanded for new source_types & operators)
        source_type = rule.get("source_type")
        source_key = rule.get("source_key")
        operator = rule.get("operator", "exists")
        expected_value = rule.get("value")
        expected_keys = rule.get("expected_keys") # For json_contains_keys
        regex_pattern = rule.get("regex_pattern") # For regex_match
        regex_patterns = rule.get("regex_patterns") # For regex_match_all/any
        json_path = rule.get("json_path")
        json_query_type = rule.get("json_query_type")
        class_types_to_check = rule.get("class_types_to_check")
        value_list = rule.get("value_list")
        line_number = rule.get("line_number")

        data_to_check: Any = None
        source_found_successfully = True # Assume true, set to false on issues

        # --- Determine data_to_check based on source_type ---
        if source_type == "pil_info_key": data_to_check = context_data["pil_info"].get(source_key)
        elif source_type == "png_chunk": data_to_check = context_data["png_chunks"].get(source_key)
        elif source_type == "exif_software_tag": data_to_check = context_data.get("software_tag")
        elif source_type == "exif_user_comment": data_to_check = context_data.get("raw_user_comment_str")
        elif source_type == "xmp_string_content": data_to_check = context_data.get("xmp_string")
        elif source_type == "file_format": data_to_check = context_data.get("file_format")
        elif source_type == "file_extension": data_to_check = context_data.get("file_extension")
        elif source_type == "raw_file_content_text": data_to_check = context_data.get("raw_file_content_text")
        elif source_type == "direct_context_key": data_to_check = context_data.get(source_key) # source_key is context key name
        elif source_type == "direct_context_key_path_value": data_to_check = self._json_path_get(context_data, source_key) # source_key is path
        elif source_type == "auto_detect_parameters_or_usercomment":
            param_str = context_data["pil_info"].get("parameters")
            uc_str = context_data.get("raw_user_comment_str")
            data_to_check = param_str if param_str else uc_str
            if data_to_check is None: source_found_successfully = False
        elif source_type == "file_content_line":
            raw_text = context_data.get("raw_file_content_text")
            if isinstance(raw_text, str) and line_number is not None:
                lines = raw_text.splitlines()
                data_to_check = lines[line_number] if 0 <= line_number < len(lines) else None
                if data_to_check is None: source_found_successfully = False
            else: source_found_successfully = False
        elif source_type in ["file_content_json", "pil_info_key_json_path", "pil_info_key_json_path_query", "file_content_json_path_value"]:
            # These source_types are handled within specific operators below.
            # For now, data_to_check remains None, operators will fetch.
            pass
        else:
            self.logger.warning(f"Unknown source_type in detection rule: {source_type}")
            return False # Rule fails if source_type is unknown

        # --- Apply Operator ---
        if not source_found_successfully and operator not in ["not_exists", "is_none"]:
            return False # If source data couldn't be found, rule (usually) fails

        try:
            if operator == "exists": return data_to_check is not None
            if operator == "not_exists": return data_to_check is None
            if operator == "is_none": return data_to_check is None
            if operator == "is_not_none": return data_to_check is not None

            if data_to_check is None: return False # Operators below here need data_to_check

            if operator == "equals": return str(data_to_check).strip() == str(expected_value).strip()
            if operator == "equals_case_insensitive": return str(data_to_check).strip().lower() == str(expected_value).strip().lower()
            if operator == "contains": return isinstance(data_to_check, str) and str(expected_value) in data_to_check
            if operator == "contains_case_insensitive": return isinstance(data_to_check, str) and str(expected_value).lower() in data_to_check.lower()
            if operator == "startswith": return isinstance(data_to_check, str) and data_to_check.startswith(str(expected_value))
            if operator == "endswith": return isinstance(data_to_check, str) and data_to_check.endswith(str(expected_value))
            if operator == "regex_match": return isinstance(data_to_check, str) and re.search(regex_pattern, data_to_check) is not None
            if operator == "regex_match_all":
                if not isinstance(data_to_check, str) or not regex_patterns: return False
                return all(re.search(p, data_to_check) for p in regex_patterns)
            if operator == "regex_match_any":
                if not isinstance(data_to_check, str) or not regex_patterns: return False
                return any(re.search(p, data_to_check) for p in regex_patterns)
            if operator == "is_string": return isinstance(data_to_check, str)
            if operator == "is_true": return data_to_check is True
            if operator == "is_in_list": return value_list and str(data_to_check) in value_list

            # JSON specific operators
            target_json_obj = None
            if source_type == "file_content_json": target_json_obj = context_data.get("parsed_root_json_object")
            elif source_type == "file_content_json_path_value": target_json_obj = data_to_check # data_to_check is already the value from path
            elif source_type == "pil_info_key_json_path": # data_to_check is the value from path
                 chunk_content = context_data["pil_info"].get(source_key)
                 if not isinstance(chunk_content, str): return False
                 try: target_json_obj = self._json_path_get(json.loads(chunk_content), json_path)
                 except json.JSONDecodeError: return False
            elif isinstance(data_to_check, str): # General case: data_to_check is a string to be parsed
                try: target_json_obj = json.loads(data_to_check)
                except json.JSONDecodeError:
                    if operator in ["is_valid_json", "json_contains_keys", "json_contains_all_keys", "is_valid_json_structure"]: return False
                    # else, it might be a string meant for other ops, continue

            if operator == "is_valid_json": return target_json_obj is not None # If it parsed, it's valid for this context
            if operator == "is_valid_json_structure": # For file_content_json source_type
                return context_data.get("parsed_root_json_object") is not None

            if operator in ["json_contains_keys", "json_contains_all_keys"]:
                if not isinstance(target_json_obj, dict) or not expected_keys: return False
                return all(k in target_json_obj for k in expected_keys)

            if operator == "is_true" and source_type == "pil_info_key_json_path_query":
                # data_to_check was set by the query logic in source_type handling (if added there)
                # For now, let's assume pil_info_key_json_path_query sets data_to_check directly
                chunk_val = context_data["pil_info"].get(source_key)
                if not isinstance(chunk_val, str): return False
                try:
                    json_obj_for_query = json.loads(chunk_val)
                    if json_query_type == "has_numeric_string_keys":
                        return any(k.isdigit() for k in json_obj_for_query.keys()) if isinstance(json_obj_for_query, dict) else False
                    if json_query_type == "has_any_node_class_type" and isinstance(json_obj_for_query, dict) and class_types_to_check:
                        return any(
                            isinstance(nd, dict) and nd.get("type") in class_types_to_check # Comfy uses "type"
                            for nd_id, nd in json_obj_for_query.get("nodes", {}).items() # Check common "nodes" structure
                        ) or any( # Or check top-level keys if nodes are directly there
                            isinstance(nd, dict) and nd.get("type") in class_types_to_check
                            for nd_id, nd in json_obj_for_query.items() if isinstance(nd_id, str) and nd_id.isdigit()
                        )

                except json.JSONDecodeError: return False

        except Exception as e_op:
            self.logger.error(f"Error evaluating operator '{operator}' for rule '{rule.get('comment', rule)}': {e_op}", exc_info=True)
            return False

        self.logger.warning(f"Unknown operator '{operator}' or unhandled data for source_type '{source_type}' in detection rule '{rule.get('comment', rule)}'.")
        return False


    # --- Field Extraction Methods ---
    def _get_a1111_kv_block(self, a1111_string: str) -> str:
        """Isolates the key-value parameter block from an A1111 string."""
        if not isinstance(a1111_string, str): return ""
        # Find end of positive prompt / start of negative or params
        neg_prompt_match = re.search(r"\nNegative prompt:", a1111_string)
        param_start_keywords = ["Steps:", "Sampler:", "CFG scale:", "Seed:", "Size:", "Model hash:", "Model:", "Version:"] # Add more
        first_param_match_idx = len(a1111_string)

        for keyword in param_start_keywords:
            match = re.search(rf"\n{re.escape(keyword)}", a1111_string)
            if match:
                first_param_match_idx = min(first_param_match_idx, match.start())

        if neg_prompt_match:
            # Parameters are after negative prompt
            param_block_start_idx = neg_prompt_match.end()
            # Find where negative prompt ends and parameters begin
            temp_block = a1111_string[param_block_start_idx:]
            actual_param_start_idx_in_temp = len(temp_block)
            for keyword in param_start_keywords:
                 match = re.search(rf"\n{re.escape(keyword)}", temp_block) # Search after neg prompt
                 if match:
                     actual_param_start_idx_in_temp = min(actual_param_start_idx_in_temp, match.start())

            return temp_block[actual_param_start_idx_in_temp:].strip()
        elif first_param_match_idx < len(a1111_string):
            # No negative prompt, parameters start after positive prompt
            return a1111_string[first_param_match_idx:].strip()
        return "" # No clear KV block found


    def _execute_field_extraction_method(self, method_def: Dict[str, Any],
                                       current_input_data: Any,
                                       context_data: Dict[str, Any],
                                       extracted_fields_cache: Dict[str, Any]) -> Any:
        method_name = method_def.get("method")
        value: Any = None # Initialize value

        # Helper to get data from various sources
        def get_source_data(source_def: Optional[Dict[str, Any]]) -> Any:
            if not source_def: return current_input_data # Default to primary input data for method
            src_type = source_def.get("type")
            src_key = source_def.get("key") # Or path for context
            if src_type == "pil_info_key": return context_data["pil_info"].get(src_key)
            if src_type == "png_chunk": return context_data["png_chunks"].get(src_key)
            if src_type == "exif_user_comment": return context_data.get("raw_user_comment_str")
            if src_type == "xmp_string": return context_data.get("xmp_string")
            if src_type == "file_content_raw_text": return context_data.get("raw_file_content_text")
            if src_type == "file_content_json_object": return context_data.get("parsed_root_json_object")
            if src_type == "direct_context_key": return context_data.get(src_key) # src_key is context key
            if src_type == "variable": return self._json_path_get(extracted_fields_cache, src_key) # src_key is var path
            return current_input_data # Fallback

        data_for_method = get_source_data(method_def.get("source_data_from_context"))

        try:
            if method_name == "direct_json_path":
                value = self._json_path_get(data_for_method, method_def.get("json_path"))
            elif method_name == "static_value":
                value = method_def.get("value")
            elif method_name == "direct_context_value": # Redundant with get_source_data, but explicit
                value = data_for_method
            elif method_name == "direct_string_value":
                value = str(data_for_method) if data_for_method is not None else None
            elif method_name == "regex_extract_group":
                if isinstance(data_for_method, str):
                    match = re.search(method_def["regex_pattern"], data_for_method)
                    if match: value = match.group(method_def.get("group_index", 1))
            elif method_name == "regex_capture_before_first_kv_match":
                 if isinstance(data_for_method, str):
                    delimiter_pattern = method_def.get("kv_block_delimiter_pattern")
                    match = re.search(delimiter_pattern, data_for_method)
                    if match: value = data_for_method[:match.start()].strip()
                    elif method_def.get("fallback_full_string"): value = data_for_method.strip()
            elif method_name == "a1111_extract_prompt_positive":
                if isinstance(data_for_method, str):
                    neg_match = re.search(r"\nNegative prompt:", data_for_method, re.IGNORECASE)
                    kv_block_str = self._get_a1111_kv_block(data_for_method)
                    end_index = len(data_for_method)
                    if neg_match: end_index = min(end_index, neg_match.start())
                    if kv_block_str: # Find where kv_block_str starts in original data_for_method
                        try:
                            kv_start_index_in_original = data_for_method.rindex(kv_block_str.split('\n',1)[0].strip()) # Find first line
                            end_index = min(end_index, kv_start_index_in_original)
                        except ValueError: pass # KV block not found as substring
                    value = data_for_method[:end_index].strip()
            elif method_name == "a1111_extract_prompt_negative":
                if isinstance(data_for_method, str):
                    neg_match = re.search(r"\nNegative prompt:(.*?)(?=(\n(Steps:|Sampler:|CFG scale:|Seed:|Size:|$)))", data_for_method, re.IGNORECASE | re.DOTALL)
                    value = neg_match.group(1).strip() if neg_match else ""
            elif method_name == "key_value_extract_from_a1111_block" or method_name == "key_value_extract_transform_from_a1111_block":
                if isinstance(data_for_method, str):
                    kv_block = self._get_a1111_kv_block(data_for_method)
                    key_to_find = method_def.get("key_name")
                    # More robust KV parsing needed here, e.g., using regex for "Key: Value,"
                    # This is a simplified placeholder:
                    # Pattern: KeyName: (capture stuff until next known key or EOL),
                    # Need to handle commas within values if they aren't separators for next A1111 KV pair.
                    # For A1111, pairs are usually comma separated, but value can have non-separating commas.
                    # Example: "Lora hashes: \"lora1:hash1, lora2:hash2\"" - here the inner comma is part of value
                    # Let's use a more specific regex for A1111 structure
                    # Key: Value, (lookahead for next potential key or end of string)
                    # Simplified:
                    match = re.search(rf"{re.escape(key_to_find)}:\s*((?:(?!(?:,\s*(?:Steps:|Sampler:|CFG scale:|Seed:|Size:|Model hash:|Model:|Version:|Clip skip:|Denoising strength:|Hires upscale:|Hires steps:|Hires upscaler:))).)*)", kv_block, re.IGNORECASE)
                    temp_val = match.group(1).strip() if match else None

                    if temp_val is not None and method_name == "key_value_extract_transform_from_a1111_block":
                        transform_match = re.search(method_def["transform_regex"], temp_val)
                        value = transform_match.group(method_def.get("transform_group", 1)) if transform_match else None
                    else:
                        value = temp_val
            elif method_name == "comfy_find_node_input": # Placeholder
                value = self._comfy_traverse_for_field(data_for_method, method_def.get("node_criteria"), method_def.get("input_key"))

            # ... Add all other method implementations from your JSON examples ...
            elif method_name == "direct_input_object_as_value": # For safetensors metadata
                value = data_for_method
            elif method_name == "json_or_python_dict_from_string_variable":
                source_var_path = method_def.get("source_variable_key")
                string_to_parse = self._json_path_get(extracted_fields_cache, source_var_path)
                if isinstance(string_to_parse, str):
                    try: value = json.loads(string_to_parse)
                    except json.JSONDecodeError:
                        try: value = eval(string_to_parse, {"__builtins__": {}}, {}) # Basic safety for dict eval
                        except: value = None
                else: value = None
            elif method_name == "parse_csv_like_lines_to_array_of_objects":
                # data_for_method is the full text content
                if isinstance(data_for_method, str):
                    lines = data_for_method.splitlines()
                    start_line = method_def.get("start_line_index", 0)
                    col_names = method_def.get("column_names", [])
                    delimiter = method_def.get("delimiter", ",")
                    result_array = []
                    for line in lines[start_line:]:
                        parts = [p.strip() for p in line.split(delimiter)]
                        if len(parts) == len(col_names):
                            result_array.append(dict(zip(col_names, parts)))
                    value = result_array
            else:
                self.logger.warning(f"Unknown field extraction method: {method_name}")

            # --- Value Type Conversion ---
            value_type = method_def.get("value_type")
            if value is not None and value_type:
                original_value_for_debug = value
                try:
                    if value_type == "integer": value = int(float(str(value))) # float cast for "1.0"
                    elif value_type == "float": value = float(str(value))
                    elif value_type == "string": value = str(value)
                    elif value_type == "boolean": value = str(value).strip().lower() in ['true', '1', 'yes', 'on', 'enabled']
                    elif value_type == "float_or_string": # Try float, then string
                        try: value = float(str(value))
                        except ValueError: value = str(value)
                    elif value_type == "integer_or_string":
                        try: value = int(float(str(value)))
                        except ValueError: value = str(value)
                    # "array", "object" are usually structural, conversion might happen earlier or be implicit.
                except (ValueError, TypeError):
                    self.logger.debug(f"Could not convert value '{original_value_for_debug}' (type {type(original_value_for_debug)}) to '{value_type}'. Method: '{method_name}'.")
                    if not method_def.get("optional", False): value = None # Nullify if not optional
                    else: value = original_value_for_debug # Keep original if optional and conversion failed
            elif value is None and not method_def.get("optional", False):
                self.logger.debug(f"Method '{method_name}' for non-optional field resulted in None.")
                # Value remains None, will be handled by calling logic

        except KeyError as e_key:
            self.logger.warning(f"KeyError during method '{method_name}': {e_key}. Check rule: {method_def.get('target_key')}")
            if not method_def.get("optional", False): value = None
        except Exception as e_method:
            self.logger.error(f"Error in field extraction method '{method_name}' for '{method_def.get('target_key')}': {e_method}", exc_info=True)
            if not method_def.get("optional", False): value = None

        return value

    def _comfy_traverse_for_field(self, workflow_graph: Any, node_criteria_list: Optional[List[Dict]], target_input_key: Optional[str]) -> Any:
        if not isinstance(workflow_graph, dict) or not node_criteria_list:
            self.logger.debug(f"ComfyUI traversal: Invalid workflow_graph or no node_criteria. Graph type: {type(workflow_graph)}")
            return None

        # Placeholder: This needs to be replaced with your actual, robust ComfyUI graph traversal logic
        # from your vendored_sdpr.format.comfyui.ComfyUI class, adapted to work here.
        # It should find nodes by class_type, id, title, trace links, and extract from widgets_values.
        # For now, a very naive example:
        graph_nodes = workflow_graph.get("nodes", {}) # ComfyUI usually has a "nodes" dict
        if not isinstance(graph_nodes, dict) and isinstance(workflow_graph, dict) and all(k.isdigit() for k in workflow_graph.keys()):
            graph_nodes = workflow_graph # API format where nodes are top-level keys

        for node_id, node_data in graph_nodes.items():
            if not isinstance(node_data, dict): continue
            for criterion in node_criteria_list:
                match = True
                if "class_type" in criterion and node_data.get("type") != criterion["class_type"]: # ComfyUI uses "type" for class_type
                    match = False
                if "node_id" in criterion and node_id != criterion["node_id"]:
                    match = False
                # Add title check: if criterion.get("meta_title_contains_priority") and node_data.get("title") ...

                if match and target_input_key:
                    # Check widgets_values (this needs mapping for each node type, like in code (3).txt)
                    # Example based on KSampler from code (3).txt where "Seed" is widgets_values[0]
                    if node_data.get("type") == "KSampler" and target_input_key == "seed":
                        return node_data.get("widgets_values", [None])[0]
                    if node_data.get("type") == "CLIPTextEncode" and target_input_key == "text":
                         # Text could be linked or a widget. If widget, it's often widgets_values[0]
                        input_text_def = node_data.get("inputs", {}).get("text")
                        if not isinstance(input_text_def, list): # Not a link, direct value
                             return node_data.get("widgets_values", [None])[0] # Assuming first widget is text
                        # Else it's linked, would need to trace.
                    # Add more specific node type handling here...
                    self.logger.debug(f"ComfyUI placeholder: Matched node {node_id} ({node_data.get('type')}), but complex input/widget logic for '{target_input_key}' not fully implemented in placeholder.")
                    return f"Placeholder for {target_input_key} from {node_data.get('type')}"
        return None # Placeholder

    def _substitute_template_vars(self, template: Any, extracted_data: Dict[str, Any],
                                context_data: Dict[str, Any],
                                original_input_data_str: Optional[str] = None,
                                input_json_object_for_template: Optional[Any] = None) -> Any:
        if isinstance(template, dict):
            return {k: self._substitute_template_vars(v, extracted_data, context_data, original_input_data_str, input_json_object_for_template) for k, v in template.items()}
        elif isinstance(template, list):
            return [self._substitute_template_vars(item, extracted_data, context_data, original_input_data_str, input_json_object_for_template) for item in template]
        elif isinstance(template, str):
            # Substitute $CONTEXT.path
            template = re.sub(r'\$CONTEXT\.([\w.]+)', lambda m: str(self._json_path_get(context_data, m.group(1))), template)

            if original_input_data_str is not None:
                template = template.replace("$INPUT_STRING_ORIGINAL_CHUNK", original_input_data_str)
                template = template.replace("$INPUT_STRING", original_input_data_str) # Assuming this is the primary processed input

            if input_json_object_for_template is not None and "$INPUT_JSON_OBJECT_AS_STRING" in template:
                 template = template.replace("$INPUT_JSON_OBJECT_AS_STRING", json.dumps(input_json_object_for_template, indent=2))

            def replace_field_var(match):
                var_path = match.group(1)
                # Use the flattened cache first for direct field lookups like $prompt
                # then try nested lookup for paths like $parameters.steps
                value = extracted_data.get(var_path.replace('.', '_')) # Try flattened key
                if value is None:
                    value = self._json_path_get(extracted_data, var_path) # Try nested path

                return str(value) if value is not None else "" # Or "null" / None based on preference
            template = re.sub(r'\$([\w.]+)', replace_field_var, template)
            return template
        return template # Numbers, booleans, None


    def get_parser_for_file(self, file_path_or_obj: str | Path | BinaryIO) -> Optional[Any]:
        display_name = getattr(file_path_or_obj, "name", str(file_path_or_obj))
        self.logger.info(f"MetadataEngine: Starting metadata parsing for: {display_name}")

        context_data = self._prepare_context_data(file_path_or_obj)
        if not context_data:
            self.logger.warning(f"MetadataEngine: Could not prepare context data for {display_name}.")
            return None

        chosen_parser_def: Optional[Dict[str, Any]] = None
        for parser_def in self.sorted_definitions:
            target_types_cfg = parser_def.get("target_file_types", ["*"])
            if not isinstance(target_types_cfg, list): target_types_cfg = [str(target_types_cfg)]
            target_types = [ft.upper() for ft in target_types_cfg]

            current_file_format_upper = context_data.get("file_format","").upper()
            current_file_ext_upper = context_data.get("file_extension","").upper()

            type_match = (
                "*" in target_types or
                (current_file_format_upper and current_file_format_upper in target_types) or
                (current_file_ext_upper and current_file_ext_upper in target_types)
            )
            if not type_match: continue

            all_rules_pass = True
            detection_rules = parser_def.get("detection_rules", [])
            if not detection_rules and target_types != ["*"]: # No rules, only specific type match needed
                pass # Type match is enough
            elif not detection_rules and target_types == ["*"]: # Generic fallback with no rules
                pass # Considered a match if it's the lowest priority
            else: # Has detection rules
                for rule_idx, rule in enumerate(detection_rules):
                    # Handle complex AND/OR conditions if rule is a dict with "condition" and "rules"
                    if isinstance(rule, dict) and "condition" in rule and "rules" in rule.get("rules", []):
                        condition_type = rule["condition"].upper()
                        sub_rules = rule["rules"]
                        if condition_type == "OR":
                            passed_block = any(self._evaluate_detection_rule(sub_rule, context_data) for sub_rule in sub_rules)
                        elif condition_type == "AND":
                            passed_block = all(self._evaluate_detection_rule(sub_rule, context_data) for sub_rule in sub_rules)
                        else: # Unknown complex condition, treat as failure for this rule
                            self.logger.warning(f"Unknown complex condition '{condition_type}' in rule {rule_idx} for {parser_def['parser_name']}")
                            passed_block = False
                        if not passed_block:
                            all_rules_pass = False; break
                    # Standard single rule
                    elif not self._evaluate_detection_rule(rule, context_data):
                        all_rules_pass = False
                        self.logger.debug(f"Rule failed for {parser_def['parser_name']}: {rule.get('comment', rule)}")
                        break
            if all_rules_pass:
                chosen_parser_def = parser_def
                self.logger.info(f"MetadataEngine: Matched parser definition: {chosen_parser_def['parser_name']}")
                break

        if not chosen_parser_def:
            self.logger.info(f"MetadataEngine: No suitable parser definition matched for {display_name}.")
            return None

        # --- New Parsing Logic ---
        if "parsing_instructions" in chosen_parser_def:
            self.logger.info(f"Using JSON-defined parsing instructions for {chosen_parser_def['parser_name']}.")
            instructions = chosen_parser_def["parsing_instructions"]
            extracted_fields: Dict[str, Any] = {"parameters": {}} # Ensure parameters sub-dict exists
            current_input_data_for_fields: Any = None
            original_input_for_template: Optional[Any] = None # Could be str or dict

            input_data_def = instructions.get("input_data", {})
            source_options = input_data_def.get("source_options", [])
            if not source_options and input_data_def.get("source_type"): source_options = [input_data_def]

            for src_opt in source_options:
                src_type = src_opt.get("source_type")
                src_key = src_opt.get("source_key")
                if src_type == "pil_info_key": current_input_data_for_fields = context_data["pil_info"].get(src_key)
                elif src_type == "exif_user_comment": current_input_data_for_fields = context_data.get("raw_user_comment_str")
                elif src_type == "xmp_string_content": current_input_data_for_fields = context_data.get("xmp_string")
                elif src_type == "file_content_raw_text": current_input_data_for_fields = context_data.get("raw_file_content_text")
                elif src_type == "file_content_json_object": current_input_data_for_fields = context_data.get("parsed_root_json_object")
                if current_input_data_for_fields is not None:
                    original_input_for_template = current_input_data_for_fields # Store before transformations
                    break

            transformations = input_data_def.get("transformations", [])
            for transform in transformations:
                transform_type = transform.get("type")
                if current_input_data_for_fields is None and transform_type not in ["create_if_not_exists"]: break # No data to transform

                if transform_type == "json_decode_string_value" and isinstance(current_input_data_for_fields, str):
                    try:
                        json_obj = json.loads(current_input_data_for_fields)
                        current_input_data_for_fields = self._json_path_get(json_obj, transform.get("path"))
                    except json.JSONDecodeError: current_input_data_for_fields = None; break
                elif transform_type == "json_decode_string_itself" and isinstance(current_input_data_for_fields, str):
                    try: current_input_data_for_fields = json.loads(current_input_data_for_fields)
                    except json.JSONDecodeError: current_input_data_for_fields = None; break

            # Store the (potentially transformed) input data if it's an object, for template var
            input_json_object_for_template = current_input_data_for_fields if isinstance(current_input_data_for_fields, (dict, list)) else None


            for field_def in instructions.get("fields", []):
                target_key_path = field_def.get("target_key")
                if not target_key_path: continue

                value = self._execute_field_extraction_method(field_def, current_input_data_for_fields, context_data, extracted_fields)
                # Store flattened key for variable access, and nested for output
                extracted_fields[target_key_path.replace('.', '_VAR_')] = value

                keys = target_key_path.split('.')
                current_dict_ptr = extracted_fields
                for i, key_segment in enumerate(keys[:-1]):
                    current_dict_ptr = current_dict_ptr.setdefault(key_segment, {})
                if value is not None or not field_def.get("optional", False): # Store if value or mandatory
                     current_dict_ptr[keys[-1]] = value


            output_template = chosen_parser_def.get("output_template")
            if output_template:
                final_output = self._substitute_template_vars(
                    output_template, extracted_fields, context_data,
                    str(original_input_for_template) if isinstance(original_input_for_template, (str,int,float,bool)) else None,
                    input_json_object_for_template
                )
                # Default width/height from context if not set by parser
                if isinstance(final_output.get("parameters"), dict):
                    if final_output["parameters"].get("width") is None and context_data.get("width", 0) > 0:
                        final_output["parameters"]["width"] = context_data["width"]
                    if final_output["parameters"].get("height") is None and context_data.get("height", 0) > 0:
                        final_output["parameters"]["height"] = context_data["height"]
                return final_output
            else: # No template, return cleaned extracted_fields
                cleaned_fields = {k:v for k,v in extracted_fields.items() if '_VAR_' not in k and k!="_input_data_object_for_template"}
                return cleaned_fields

        # --- Fallback to Python BaseFormat class ---
        elif "base_format_class" in chosen_parser_def:
            # ... (Existing Python class instantiation logic remains here) ...
            self.logger.info(f"Using Python class-based parser {chosen_parser_def['base_format_class']} for {chosen_parser_def['parser_name']}.")
            parser_class_name = chosen_parser_def["base_format_class"]
            ParserClass = get_parser_class_by_name(parser_class_name)

            if not ParserClass:
                self.logger.error(f"Python class '{parser_class_name}' not found in registry for {chosen_parser_def['parser_name']}.")
                return None

            raw_input_for_parser = ""
            primary_data_def = chosen_parser_def.get("primary_data_source_for_raw", {})
            pds_type = primary_data_def.get("source_type")
            pds_key = primary_data_def.get("source_key")

            if pds_type == "png_chunk" and pds_key: raw_input_for_parser = context_data["png_chunks"].get(pds_key, "")
            elif pds_type == "exif_user_comment": raw_input_for_parser = context_data.get("raw_user_comment_str", "")
            elif pds_type == "xmp_string_content": raw_input_for_parser = context_data.get("xmp_string", "")
            # Add more sources for raw input as needed

            parser_instance = ParserClass(
                info=context_data, raw=raw_input_for_parser,
                width=str(context_data["width"]), height=str(context_data["height"]),
                logger_obj=self.logger
            )
            parser_status = parser_instance.parse()
            if parser_status == BaseFormat.Status.READ_SUCCESS:
                self.logger.info(f"Python Parser {parser_instance.tool} succeeded.")
                return parser_instance
            else:
                status_name = parser_status.name if hasattr(parser_status, "name") else str(parser_status)
                self.logger.warning(f"Python Parser {parser_instance.tool} did not succeed. Status: {status_name}. Error: {parser_instance.error}")
                return None
        else:
            self.logger.error(f"Parser definition {chosen_parser_def['parser_name']} has neither 'parsing_instructions' nor 'base_format_class'.")
            return None

if __name__ == "__main__":
    # --- Basic Test Setup ---
    # Create a dummy logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_logger = logging.getLogger("TestMetadataEngine")
    test_logger.setLevel(logging.DEBUG) # Engine uses its own logger, this is for __main__

    # Create a temporary directory for parser definitions
    temp_defs_path = Path("./temp_parser_definitions_test")
    temp_defs_path.mkdir(exist_ok=True)

    # Create a dummy A1111 JSON definition for testing
    a1111_def_content = {
        "parser_name": "A1111 Test (JSON Driven)",
        "priority": 100,
        "target_file_types": ["PNG", "JPEG"],
        "detection_rules": [
            {"source_type": "auto_detect_parameters_or_usercomment", "operator": "regex_match", "regex_pattern": "Steps:"},
            {"source_type": "auto_detect_parameters_or_usercomment", "operator": "regex_match", "regex_pattern": "CFG scale:"}
        ],
        "parsing_instructions": {
            "input_data": {
                "source_options": [
                    {"source_type": "pil_info_key", "source_key": "parameters"},
                    {"source_type": "exif_user_comment"}
                ]
            },
            "fields": [
                {"target_key": "prompt", "method": "a1111_extract_prompt_positive"},
                {"target_key": "negative_prompt", "method": "a1111_extract_prompt_negative"},
                {"target_key": "parameters.steps", "method": "key_value_extract_from_a1111_block", "key_name": "Steps", "value_type": "integer"},
                {"target_key": "parameters.sampler", "method": "key_value_extract_from_a1111_block", "key_name": "Sampler", "value_type": "string"},
                {"target_key": "parameters.cfg_scale", "method": "key_value_extract_from_a1111_block", "key_name": "CFG scale", "value_type": "float"},
                {"target_key": "parameters.seed", "method": "key_value_extract_from_a1111_block", "key_name": "Seed", "value_type": "integer"}
            ],
            "output_template": {
                "tool": "A1111 (from JSON Test)",
                "prompt": "$prompt",
                "negative_prompt": "$negative_prompt",
                "parameters": {
                    "steps": "$parameters.steps",
                    "sampler": "$parameters.sampler",
                    "cfg_scale": "$parameters.cfg_scale",
                    "seed": "$parameters.seed",
                    "width": "$CONTEXT.width",
                    "height": "$CONTEXT.height"
                }
            }
        }
    }
    with open(temp_defs_path / "a1111_test.json", "w") as f:
        json.dump(a1111_def_content, f, indent=2)

    # Create a dummy TXT file parser definition
    txt_def_content = {
        "parser_name": "Plain Text Prompt File Test",
        "priority": 50,
        "target_file_types": ["TXT"],
        "detection_rules": [{"source_type": "file_extension", "operator": "equals_case_insensitive", "value": "txt"}],
        "parsing_instructions": {
            "input_data": {"source_type": "file_content_raw_text"},
            "fields": [{"target_key": "prompt", "method": "direct_string_value"}],
            "output_template": {"tool": "Text Prompt Test", "prompt": "$prompt", "parameters": {}}
        }
    }
    with open(temp_defs_path / "txt_test.json", "w") as f:
        json.dump(txt_def_content, f, indent=2)

    # Instantiate engine
    engine = MetadataEngine(parser_definitions_path=temp_defs_path, logger_obj=test_logger)

    # --- Test Case 1: Dummy PNG/A1111 style ---
    # Create a dummy image file in memory (or use a real one)
    try:
        from io import BytesIO
        dummy_png_a1111_data = "test positive prompt\nNegative prompt: test negative prompt\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 12345"
        img_a1111 = Image.new('RGB', (100, 150), color = 'red')
        pnginfo_a1111 = Image.PngInfo()
        pnginfo_a1111.add_text("parameters", dummy_png_a1111_data) # Simulate A1111 parameters chunk

        img_bytes_a1111 = BytesIO()
        img_a1111.save(img_bytes_a1111, format="PNG", pnginfo=pnginfo_a1111)
        img_bytes_a1111.seek(0) # Reset stream position
        img_bytes_a1111.name = "dummy_a1111.png" # Add name attribute

        test_logger.info("\n--- Testing A1111-style PNG (dummy_a1111.png) ---")
        result_a1111 = engine.get_parser_for_file(img_bytes_a1111)
        if result_a1111 and isinstance(result_a1111, dict):
            test_logger.info(f"A1111 Test Parsed successfully. Tool: {result_a1111.get('tool')}")
            test_logger.info(f"  Prompt: {result_a1111.get('prompt')}")
            test_logger.info(f"  Negative: {result_a1111.get('negative_prompt')}")
            test_logger.info(f"  Params: {result_a1111.get('parameters')}")
        else:
            test_logger.warning("A1111 Test PNG parsing failed or returned unexpected type.")

    except Exception as e_test_a1111:
        test_logger.error(f"Error during A1111 test setup or execution: {e_test_a1111}")


    # --- Test Case 2: Dummy TXT file ---
    dummy_txt_path = Path("./dummy_prompt.txt")
    with open(dummy_txt_path, "w") as f_txt:
        f_txt.write("This is a simple text prompt from a file.")

    test_logger.info(f"\n--- Testing TXT file ({dummy_txt_path.name}) ---")
    result_txt = engine.get_parser_for_file(dummy_txt_path)
    if result_txt and isinstance(result_txt, dict):
        test_logger.info(f"TXT Test Parsed successfully. Tool: {result_txt.get('tool')}")
        test_logger.info(f"  Prompt: {result_txt.get('prompt')}")
    else:
        test_logger.warning("TXT file parsing failed or returned unexpected type.")

    # Cleanup dummy files and dir
    try:
        dummy_txt_path.unlink(missing_ok=True)
        for f_json in temp_defs_path.glob("*.json"):
            f_json.unlink()
        temp_defs_path.rmdir()
    except OSError as e_cleanup:
        test_logger.error(f"Error cleaning up test files: {e_cleanup}")

    test_logger.info("\n--- Basic Tests Complete ---")