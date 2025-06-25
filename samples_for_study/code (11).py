# dataset_tools/metadata_engine.py
# ... (imports and existing registry functions remain the same) ...

# NEW: Import necessary libraries for advanced parsing methods
import xml.etree.ElementTree as ET # For XMP parsing if needed

class MetadataEngine:
    def __init__(self, parser_definitions_path: str | Path, logger_obj: Optional[logging.Logger] = None):
        # ... (existing __init__ logic for loading and sorting definitions) ...
        # Modify _load_parser_definitions to accept new JSON structure
        # For now, we'll assume it loads them as is.

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
                    # Validation: must have parser_name.
                    # 'base_format_class' might become optional if parsing_instructions are present.
                    # 'parsing_instructions' and/or 'output_template' might become required for this new style.
                    if "parser_name" in definition: # Basic check
                        definitions.append(definition)
                    else:
                        self.logger.warning(f"Skipping invalid parser definition (missing parser_name): {filepath.name}")
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to decode JSON from {filepath.name}: {e}")
            except Exception as e: # noqa: BLE001
                self.logger.error(f"Unexpected error loading parser definition {filepath.name}: {e}")
        return definitions


    def _prepare_context_data(self, file_path_or_obj: str | Path | BinaryIO) -> Optional[Dict[str, Any]]:
        # EXPANDED Context Data Preparation
        context: Dict[str, Any] = {
            "pil_info": {}, "exif_dict": {}, "xmp_string": None,
            "parsed_xmp_dict": None, # NEW: For structured XMP
            "png_chunks": {}, "file_format": "", "width": 0, "height": 0,
            "raw_user_comment_str": None, "software_tag": None,
            "file_extension": "", # NEW: For non-image files
            "raw_file_content_text": None, # NEW: For text files
            "raw_file_content_bytes": None, # NEW: For other binary files
            "parsed_root_json_object": None, # NEW: If file itself is JSON
            "file_path_original": str(file_path_or_obj.name if hasattr(file_path_or_obj, "name") else file_path_or_obj) # NEW
        }
        is_binary_io = hasattr(file_path_or_obj, "read") and hasattr(file_path_or_obj, "seek")

        try:
            # Try to open as an image first
            img = Image.open(file_path_or_obj)
            context["pil_info"] = img.info.copy() if img.info else {}
            context["width"] = img.width
            context["height"] = img.height
            context["file_format"] = img.format.upper() if img.format else ""
            context["file_extension"] = Path(img.filename).suffix.lstrip('.').lower() if hasattr(img, 'filename') and img.filename else ""

            if exif_bytes := context["pil_info"].get("exif"):
                # ... (existing EXIF extraction logic) ...
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


            if xmp_str := context["pil_info"].get("XML:com.adobe.xmp"):
                context["xmp_string"] = xmp_str
                # try: # Attempt to parse XMP string into a dict (simplified for now)
                #     context["parsed_xmp_dict"] = self._parse_xmp_string_to_dict(xmp_str)
                # except Exception as e_xmp_parse:
                #     self.logger.debug(f"Failed to parse XMP string into dict: {e_xmp_parse}")


            for key, val in context["pil_info"].items():
                if isinstance(val, str): context["png_chunks"][key] = val
            if "UserComment" in context["pil_info"] and "UserComment" not in context["png_chunks"]:
                context["png_chunks"]["UserComment"] = context["pil_info"]["UserComment"]
            img.close()

        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path_or_obj}")
            return None
        except UnidentifiedImageError:
            self.logger.info(f"Cannot identify as image: {file_path_or_obj}. Checking extension for non-image parsing.")
            file_path_str = str(file_path_or_obj.name if hasattr(file_path_or_obj, "name") else file_path_or_obj)
            p = Path(file_path_str)
            context["file_extension"] = p.suffix.lstrip('.').lower()
            context["file_format"] = context["file_extension"].upper() # Use extension as format

            if context["file_extension"] == "json":
                try:
                    if is_binary_io: file_path_or_obj.seek(0)
                    mode = 'rb' if is_binary_io else 'r'
                    encoding = None if is_binary_io else 'utf-8'
                    with (file_path_or_obj if is_binary_io else open(p, mode, encoding=encoding)) as f:
                        context["parsed_root_json_object"] = json.load(f)
                        # For consistency, also store raw text if easily obtainable
                        if hasattr(f, 'seek') and hasattr(f, 'read'): # Check if it's a file object
                            f.seek(0)
                            context["raw_file_content_text"] = f.read() if isinstance(f.read(), str) else f.read().decode('utf-8', 'replace')

                except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e_json_direct:
                    self.logger.error(f"Failed to read/parse direct JSON file {p.name}: {e_json_direct}")
                    # Potentially still try to get raw text if JSON parse failed
                    if not context["raw_file_content_text"]:
                         try:
                            if is_binary_io: file_path_or_obj.seek(0)
                            with (file_path_or_obj if is_binary_io else open(p, 'r', encoding='utf-8', errors='replace')) as f_txt:
                                context["raw_file_content_text"] = f_txt.read()
                         except Exception: pass # Ignore if raw text also fails
                    # return None # Or allow generic text parsing if JSON fails
            elif context["file_extension"] == "txt":
                try:
                    if is_binary_io: file_path_or_obj.seek(0)
                    with (file_path_or_obj if is_binary_io else open(p, 'r', encoding='utf-8', errors='replace')) as f_txt:
                        context["raw_file_content_text"] = f_txt.read()
                except (OSError, UnicodeDecodeError) as e_txt:
                    self.logger.error(f"Failed to read TXT file {p.name}: {e_txt}")
                    return None # Cannot process if TXT read fails
            # Add elif for .safetensors, .gguf to call their specific context prep
            elif context["file_extension"] == "safetensors":
                from dataset_tools.model_parsers.safetensors_parser import SafetensorsParser # Lazy import
                temp_parser = SafetensorsParser(str(p)) # Pass path str
                status = temp_parser.parse()
                if status == temp_parser.status.SUCCESS:
                    context["safetensors_metadata"] = temp_parser.metadata_header
                    context["safetensors_main_header"] = temp_parser.main_header
                elif status == temp_parser.status.NOT_APPLICABLE:
                     self.logger.info(f"Safetensors parser not applicable for {p.name}")
                     # It's not an image, and not safetensors by content
                else: # FAILURE
                    self.logger.warning(f"Safetensors parser failed for {p.name}: {temp_parser._error_message}")

            elif context["file_extension"] == "gguf":
                from dataset_tools.model_parsers.gguf_parser import GGUFParser # Lazy import
                temp_parser = GGUFParser(str(p))
                status = temp_parser.parse()
                if status == temp_parser.status.SUCCESS:
                    context["gguf_metadata"] = temp_parser.metadata_header
                    context["gguf_main_header"] = temp_parser.main_header # version, counts
                # Handle NOT_APPLICABLE / FAILURE similarly

            else: # Unknown non-image file
                self.logger.info(f"File {p.name} is not a recognized image or special non-image type.")
                # Could try to read as raw bytes for some generic binary parser
                # For now, we might not have enough info to proceed for a generic non-image binary.
                # However, some parser definitions might still match based on file_extension.

        except Exception as e: # General error during image processing or initial file access
            self.logger.error(f"Error preparing context data for {file_path_or_obj}: {e}", exc_info=True)
            return None

        self.logger.debug("Prepared context data for evaluation. Keys: %s", list(context.keys()))
        return context

    def _evaluate_detection_rule(self, rule: Dict[str, Any], context_data: Dict[str, Any]) -> bool:
        # EXPANDED for new source_types and operators
        source_type = rule.get("source_type")
        source_key = rule.get("source_key") # For pil_info, png_chunk, exif_tag_general
        operator = rule.get("operator", "exists")
        expected_value = rule.get("value")
        expected_keys = rule.get("expected_keys")
        regex_pattern = rule.get("regex_pattern")
        regex_patterns = rule.get("regex_patterns") # For regex_match_all / regex_match_any
        json_path = rule.get("json_path")           # For JSON path checks
        json_query_type = rule.get("json_query_type") # For custom JSON queries
        class_types_to_check = rule.get("class_types_to_check") # For ComfyUI node type checks
        value_list = rule.get("value_list") # For "is_in_list" operator
        line_number = rule.get("line_number") # For file_content_line

        data_to_check: Any = None
        found_source = True

        if source_type == "pil_info_key":
            data_to_check = context_data["pil_info"].get(source_key)
        elif source_type == "png_chunk":
            data_to_check = context_data["png_chunks"].get(source_key)
        elif source_type == "exif_software_tag":
            data_to_check = context_data.get("software_tag")
        elif source_type == "exif_user_comment":
            data_to_check = context_data.get("raw_user_comment_str")
        elif source_type == "exif_tag_general":
             # ... (existing logic) ...
            if context_data["exif_dict"] and "0th" in context_data["exif_dict"]:
                tag_code = getattr(piexif.ImageIFD, source_key, None) if source_key else None
                if tag_code:
                    val_bytes = context_data["exif_dict"]["0th"].get(tag_code)
                    if isinstance(val_bytes, bytes): data_to_check = val_bytes.decode('ascii','ignore').strip('\x00')
                    elif val_bytes is not None: data_to_check = str(val_bytes)

        elif source_type == "xmp_string_content":
            data_to_check = context_data.get("xmp_string")
        elif source_type == "file_format": # From PIL or derived from extension
            data_to_check = context_data.get("file_format")
        elif source_type == "file_extension": # NEW
            data_to_check = context_data.get("file_extension")

        # NEW: Handling JSON content directly from file (e.g. for .json, .state, .kohya configs)
        elif source_type == "file_content_json": # Indicates the rule operates on context_data["parsed_root_json_object"]
            if context_data.get("parsed_root_json_object") is None:
                found_source = False # The file wasn't valid JSON or wasn't parsed
            # data_to_check will be set within operator logic for json_contains_keys etc.
            # For 'is_valid_json_structure', the existence of parsed_root_json_object is enough.
            pass # Operator will handle this
        elif source_type == "file_content_json_path_value": # Get a specific value from parsed root JSON
            if context_data.get("parsed_root_json_object"):
                data_to_check = self._json_path_get(context_data["parsed_root_json_object"], json_path)
            else: found_source = False

        # NEW: For detection rules that need to look inside a string that is known to be JSON (e.g. PNG chunk value)
        elif source_type == "pil_info_key_json_path": # e.g. workflow chunk -> json_path
            chunk_content = context_data["pil_info"].get(source_key)
            if isinstance(chunk_content, str):
                try:
                    json_obj = json.loads(chunk_content)
                    data_to_check = self._json_path_get(json_obj, json_path)
                except json.JSONDecodeError: found_source = False # Chunk not valid JSON
            else: found_source = False # Chunk not string
        elif source_type == "pil_info_key_json_path_query": # custom query on chunk's JSON
            chunk_content = context_data["pil_info"].get(source_key)
            if isinstance(chunk_content, str):
                try:
                    json_obj = json.loads(chunk_content)
                    if json_query_type == "has_numeric_string_keys":
                        data_to_check = any(k.isdigit() for k in json_obj.keys()) if isinstance(json_obj, dict) else False
                    elif json_query_type == "has_any_node_class_type" and isinstance(json_obj, dict) and class_types_to_check:
                        # Assumes ComfyUI workflow structure where top level keys are node IDs
                        data_to_check = any(
                            isinstance(node_data, dict) and node_data.get("type") in class_types_to_check
                            for node_data in json_obj.values()
                        )
                    else: found_source = False # Unknown query
                except json.JSONDecodeError: found_source = False
            else: found_source = False

        # NEW: For A1111 style auto-detection
        elif source_type == "auto_detect_parameters_or_usercomment":
            param_str = context_data["pil_info"].get("parameters")
            uc_str = context_data.get("raw_user_comment_str")
            data_to_check = param_str if param_str else uc_str
            if data_to_check is None: found_source = False
        elif source_type == "auto_detect_parameters_or_usercomment_kv":
            # Special: this operator will handle getting the KV from the string
             param_str = context_data["pil_info"].get("parameters")
             uc_str = context_data.get("raw_user_comment_str")
             data_to_check = param_str if param_str else uc_str # The string itself
             if data_to_check is None: found_source = False
             # The operator "kv_exists", "kv_equals" etc. will parse this string

        # NEW: For .txt files (or other raw text content)
        elif source_type == "file_content_raw_text":
            data_to_check = context_data.get("raw_file_content_text")
        elif source_type == "file_content_line":
            raw_text = context_data.get("raw_file_content_text")
            if isinstance(raw_text, str) and line_number is not None:
                lines = raw_text.splitlines()
                if 0 <= line_number < len(lines):
                    data_to_check = lines[line_number]
                else: found_source = False # Line number out of bounds
            else: found_source = False

        # NEW: For direct context checks (e.g., safetensors_metadata)
        elif source_type == "direct_context_key": # e.g. context_key_path = "safetensors_metadata"
            data_to_check = context_data.get(source_key) # Here source_key is the top-level context key
        elif source_type == "direct_context_key_path_value": # e.g. context_key_path = "safetensors_metadata.format"
            data_to_check = self._json_path_get(context_data, source_key) # source_key is the full path

        else:
            self.logger.warning(f"Unknown source_type in detection rule: {source_type}")
            found_source = False

        if not found_source and operator != "not_exists": # If source wasn't found, rule usually fails
             return False
        if data_to_check is None and operator not in ["not_exists", "is_none"]:
            return False

        try:
            # ... (existing basic operators: exists, not_exists, is_none, is_not_none) ...
            # ... (existing string operators: equals, contains, startswith, endswith, regex_match) ...
            # ... (existing JSON operators: is_valid_json, json_contains_keys) ...
            if operator == "exists": return data_to_check is not None
            if operator == "not_exists": return data_to_check is None
            if operator == "is_none": return data_to_check is None
            if operator == "is_not_none": return data_to_check is not None

            if data_to_check is None: return False # Should be caught above but defensive

            if operator == "equals": return str(data_to_check).strip() == str(expected_value).strip()
            if operator == "equals_case_insensitive": return str(data_to_check).strip().lower() == str(expected_value).strip().lower()
            if operator == "contains": return isinstance(data_to_check, str) and expected_value in data_to_check
            if operator == "contains_case_insensitive": return isinstance(data_to_check, str) and expected_value.lower() in data_to_check.lower()
            if operator == "startswith": return isinstance(data_to_check, str) and data_to_check.startswith(expected_value)
            if operator == "endswith": return isinstance(data_to_check, str) and data_to_check.endswith(expected_value)
            if operator == "regex_match": return isinstance(data_to_check, str) and re.search(regex_pattern, data_to_check) is not None

            # NEW operators
            if operator == "regex_match_all":
                if not isinstance(data_to_check, str) or not regex_patterns: return False
                return all(re.search(p, data_to_check) for p in regex_patterns)
            if operator == "regex_match_any":
                if not isinstance(data_to_check, str) or not regex_patterns: return False
                return any(re.search(p, data_to_check) for p in regex_patterns)

            if operator == "is_valid_json":
                if not isinstance(data_to_check, str): return False
                try: json.loads(data_to_check); return True
                except json.JSONDecodeError: return False
            if operator == "json_contains_keys": # Operates on data_to_check if it's a string to be parsed, or on parsed_root_json_object
                target_json_obj = None
                if source_type == "file_content_json": # Use the pre-parsed root JSON
                    target_json_obj = context_data.get("parsed_root_json_object")
                elif isinstance(data_to_check, str): # Try to parse data_to_check as JSON
                    try: target_json_obj = json.loads(data_to_check)
                    except json.JSONDecodeError: return False
                elif isinstance(data_to_check, dict): # Already a dict
                    target_json_obj = data_to_check

                if not isinstance(target_json_obj, dict) or not expected_keys: return False
                return all(key in target_json_obj for key in expected_keys)
            if operator == "json_contains_all_keys": # Alias for clarity if used with file_content_json
                target_json_obj = context_data.get("parsed_root_json_object")
                if not isinstance(target_json_obj, dict) or not expected_keys: return False
                return all(key in target_json_obj for key in expected_keys)


            if operator == "is_string": return isinstance(data_to_check, str)
            if operator == "is_true": return data_to_check is True # For results of json_query_type
            if operator == "is_in_list":
                if not value_list: return False
                return str(data_to_check) in value_list # Compare as strings for simplicity

            # For source_type "pil_info_key_json_path_string_is_json"
            if operator == "is_valid_json" and source_type == "pil_info_key_json_path_string_is_json": # Special case
                # data_to_check here would be the string from the json_path
                if not isinstance(data_to_check, str): return False
                try: json.loads(data_to_check); return True
                except json.JSONDecodeError: return False

            # For "is_valid_json_structure" operator
            if operator == "is_valid_json_structure" and source_type == "file_content_json":
                return context_data.get("parsed_root_json_object") is not None


        except Exception as e_op:
            self.logger.error(f"Error evaluating operator '{operator}' for rule '{rule}': {e_op}", exc_info=True)
            return False

        self.logger.warning(f"Unknown operator '{operator}' or unhandled case for source_type '{source_type}' in detection rule.")
        return False

    # --- NEW: JSON Path Helper ---
    def _json_path_get(self, data_dict: Dict[str, Any], path_str: str) -> Any:
        """Rudimentary JSON path getter (e.g., 'extra.extraMetadata' or 'Options[0].Name')."""
        if not path_str: return data_dict
        keys = path_str.split('.')
        current = data_dict
        for key_part in keys:
            if isinstance(current, dict):
                match = re.match(r"(\w+)\[(\d+)\]", key_part) # Check for array access like key[0]
                if match:
                    array_key, index_str = match.groups()
                    index = int(index_str)
                    if array_key not in current or not isinstance(current[array_key], list) or index >= len(current[array_key]):
                        return None
                    current = current[array_key][index]
                elif key_part in current:
                    current = current[key_part]
                else:
                    return None # Key not found
            elif isinstance(current, list): # Direct array access like [0].key
                if key_part.startswith('[') and key_part.endswith(']'):
                    try:
                        index = int(key_part[1:-1])
                        if 0 <= index < len(current):
                            current = current[index]
                        else: return None # Index out of bounds
                    except ValueError: return None # Invalid index
                else: return None # Expected array index
            else:
                return None # Cannot traverse further
        return current


    # --- NEW: Field Extraction Methods Placeholder ---
    def _execute_field_extraction_method(self, method_def: Dict[str, Any], current_input_data: Any, context_data: Dict[str, Any], extracted_fields_cache: Dict[str, Any]) -> Any:
        method_name = method_def.get("method")
        # This will be a large dispatch table or series of if/elifs
        # self.logger.debug(f"Executing method: {method_name} with input_data type: {type(current_input_data)}")

        if method_name == "direct_json_path":
            # Assumes current_input_data is already a parsed JSON object (dict/list)
            json_path = method_def.get("json_path")
            value = self._json_path_get(current_input_data, json_path)
        elif method_name == "static_value":
            value = method_def.get("value")
        elif method_name == "direct_context_value":
            source_def = method_def.get("source_data_from_context", {})
            src_type = source_def.get("type")
            src_key = source_def.get("key")
            if src_type == "png_chunk": value = context_data.get("png_chunks", {}).get(src_key)
            elif src_type == "xmp_string": value = context_data.get("xmp_string")
            # ... more context sources
            else: value = None
        elif method_name == "direct_string_value": # For .txt prompt
            value = current_input_data
        # ... other methods like regex_capture_before_first_kv_match, key_value_extract, a1111_extract_prompt_positive etc.
        # These would call specific helper functions.

        # --- A1111 Specific Method Placeholders ---
        elif method_name == "a1111_extract_prompt_positive":
            # current_input_data is the full A1111 string
            # Placeholder: Actual logic would parse out the positive prompt
            if isinstance(current_input_data, str):
                # Simplified: assume prompt is before "Negative prompt:" or "Steps:"
                neg_match = re.search(r"\nNegative prompt:", current_input_data)
                param_match = re.search(r"\nSteps:", current_input_data)
                end_index = len(current_input_data)
                if neg_match: end_index = min(end_index, neg_match.start())
                if param_match: end_index = min(end_index, param_match.start())
                value = current_input_data[:end_index].strip()
            else: value = None
        elif method_name == "a1111_extract_prompt_negative":
            if isinstance(current_input_data, str):
                neg_match = re.search(r"\nNegative prompt:(.*?)(?:\nSteps:|\nCFG scale:|$)", current_input_data, re.DOTALL)
                value = neg_match.group(1).strip() if neg_match else ""
            else: value = None
        elif method_name == "key_value_extract_from_a1111_block":
            # Placeholder: This method needs to:
            # 1. Isolate the KV block from current_input_data (after prompts).
            # 2. Parse that block (e.g., split by comma, then by colon).
            # 3. Find method_def["key_name"] and return its value.
            # 4. Convert to method_def["value_type"].
            # For now, a very simplified example:
            if isinstance(current_input_data, str):
                key_to_find = method_def.get("key_name")
                # Naive: assumes "Key: Value,"
                match = re.search(rf"{re.escape(key_to_find)}:\s*([^,]+)", current_input_data)
                value = match.group(1).strip() if match else None
            else: value = None

        # --- ComfyUI Specific Method Placeholders ---
        elif method_name == "comfy_find_node_input":
            # current_input_data is the ComfyUI workflow graph (dict)
            # This would involve complex graph traversal logic.
            # For now, return placeholder. Actual implementation uses self._comfy_traverse_for_field
            value = self._comfy_traverse_for_field(current_input_data, method_def.get("node_criteria"), method_def.get("input_key"))

        else:
            self.logger.warning(f"Unknown field extraction method: {method_name}")
            value = None

        # Handle value_type conversion if specified (simplified)
        value_type = method_def.get("value_type")
        if value is not None and value_type:
            try:
                if value_type == "integer": value = int(value)
                elif value_type == "float": value = float(value)
                elif value_type == "string": value = str(value)
                elif value_type == "boolean": value = str(value).lower() in ['true', '1', 'yes', 'on']
                # "array", "object" are more about structure than simple type conversion here.
            except (ValueError, TypeError) as e_type:
                self.logger.debug(f"Could not convert value '{value}' to type '{value_type}': {e_type}")
                if not method_def.get("optional", False): value = None # Nullify if not optional and conversion fails
                # else keep original if optional and conversion fails
        return value

    # --- NEW: Placeholder for ComfyUI traversal logic ---
    def _comfy_traverse_for_field(self, workflow_graph: Dict, node_criteria: List[Dict], target_input_key: str) -> Any:
        # This is a VERY simplified placeholder. Real traversal is complex.
        # It needs to find nodes matching criteria, trace links if input is linked, etc.
        if not isinstance(workflow_graph, dict) or not node_criteria:
            return None

        for node_id, node_data in workflow_graph.get("nodes", {}).items(): # Iterate over nodes if "nodes" key exists
            if not isinstance(node_data, dict): continue

            for criterion in node_criteria: # Check against each criterion
                matches_criterion = True
                if "class_type" in criterion and node_data.get("type") != criterion["class_type"]:
                    matches_criterion = False
                if "node_id" in criterion and node_id != criterion["node_id"]:
                    matches_criterion = False
                # Add more criteria checks like meta_title_contains_priority

                if matches_criterion:
                    # Found a matching node. Now get the target_input_key
                    if target_input_key in node_data.get("inputs", {}):
                        input_val_or_link = node_data["inputs"][target_input_key]
                        if isinstance(input_val_or_link, list): # It's a link [from_node_id, from_slot_idx]
                            # Real logic would trace this link recursively.
                            # For placeholder, assume if it's a link, we can't get direct value easily.
                            self.logger.debug(f"ComfyUI: Field {target_input_key} for node {node_id} is linked, placeholder cannot resolve.")
                            return "LinkedValue (Placeholder)"
                        else: # Direct value (widget)
                             # Often in widgets_values based on an implicit index or widget name
                            if "widget" in node_data["inputs"][target_input_key] and "name" in node_data["inputs"][target_input_key]["widget"]:
                                widget_name = node_data["inputs"][target_input_key]["widget"]["name"]
                                # Find the widget_value by name (this mapping is not standard in base ComfyUI, needs convention)
                                # Or, more commonly, by index if widgets_values maps to inputs by order.
                                # This part is very complex for a generic solution.
                                # For now, let's check widgets_values directly if target_input_key is a common one.
                                if target_input_key in node_data.get("widgets_values", []): # This is not how it usually works
                                     return node_data["widgets_values"][target_input_key] # This is wrong
                                if node_data.get("widgets_values"): # A KSampler often has seed, steps, cfg etc. in order
                                    if target_input_key == "seed" and len(node_data["widgets_values"]) > 0: return node_data["widgets_values"][0]
                                    if target_input_key == "steps" and len(node_data["widgets_values"]) > 2: return node_data["widgets_values"][2] # Example
                                    # This needs proper mapping for each node type, like in your code (3).txt
                                self.logger.debug(f"ComfyUI: Found direct input for {target_input_key} in node {node_id}, but widget mapping is complex.")
                                return f"DirectValueFor_{target_input_key} (Placeholder)"

                        return input_val_or_link # Or potentially from widgets_values
                    # If target_input_key is for a widget value directly (not an "input" socket)
                    elif target_input_key in node_data.get("widgets_values", []): # This is also not quite right
                        # Need to know index of widget.
                        pass


        self.logger.debug(f"ComfyUI: Could not find node/input for criteria: {node_criteria}, key: {target_input_key}")
        return None


    # --- NEW: Output Template Substitution ---
    def _substitute_template_vars(self, template: Any, extracted_data: Dict[str, Any], context_data: Dict[str, Any], original_input_data_str: Optional[str] = None) -> Any:
        if isinstance(template, dict):
            return {k: self._substitute_template_vars(v, extracted_data, context_data, original_input_data_str) for k, v in template.items()}
        elif isinstance(template, list):
            return [self._substitute_template_vars(item, extracted_data, context_data, original_input_data_str) for item in template]
        elif isinstance(template, str):
            # Substitute $CONTEXT.key
            template = re.sub(r'\$CONTEXT\.(\w+)', lambda m: str(self._json_path_get(context_data, m.group(1))), template)
            # Substitute $INPUT_STRING (or other special input vars)
            if original_input_data_str is not None:
                 template = template.replace("$INPUT_STRING_ORIGINAL_CHUNK", original_input_data_str) # Example
                 template = template.replace("$INPUT_STRING", original_input_data_str) # If used for the processed input
            if "$INPUT_JSON_OBJECT_AS_STRING" in template and isinstance(extracted_data.get("_input_data_object_for_template"), dict): # Check if special input was stored
                 template = template.replace("$INPUT_JSON_OBJECT_AS_STRING", json.dumps(extracted_data.get("_input_data_object_for_template"), indent=2))


            # Substitute $field_path
            # This needs to handle nested paths like $parameters.tool_specific.version
            # A simple direct replacement won't work for nested keys.
            # We need to look up the full path in extracted_data.
            # For now, a simple replacement for top-level keys:
            # template = re.sub(r'\$(\w+(\.\w+)*)', lambda m: str(extracted_data.get(m.group(1))), template)
            # More robust:
            def replace_var(match):
                var_path = match.group(1)
                value = self._json_path_get(extracted_data, var_path)
                return str(value) if value is not None else "" # Or some placeholder like "None"
            template = re.sub(r'\$([\w.]+)', replace_var, template)

            return template
        else: # Numbers, booleans, None
            return template


    def get_parser_for_file(self, file_path_or_obj: str | Path | BinaryIO) -> Optional[Any]: # Return type will be Dict or BaseFormat
        display_name = file_path_or_obj.name if hasattr(file_path_or_obj, "name") else str(file_path_or_obj)
        self.logger.info(f"MetadataEngine: Starting metadata parsing for: {display_name}")

        context_data = self._prepare_context_data(file_path_or_obj)
        if not context_data:
            self.logger.warning(f"MetadataEngine: Could not prepare context data for {display_name}.")
            return None

        chosen_parser_def: Optional[Dict[str, Any]] = None
        for parser_def in self.sorted_definitions:
            target_types_cfg = parser_def.get("target_file_types", ["*"])
            # Ensure target_types_cfg is a list
            if not isinstance(target_types_cfg, list): target_types_cfg = [str(target_types_cfg)]

            target_types = [ft.upper() for ft in target_types_cfg]

            current_file_format = context_data.get("file_format","").upper()
            current_file_ext = context_data.get("file_extension","").upper()

            # Match if specific format matches, OR extension matches (for non-image/special types), OR "*"
            type_match = (
                "*" in target_types or
                (current_file_format and current_file_format in target_types) or
                (current_file_ext and current_file_ext in target_types)
            )
            if not type_match:
                continue


            all_rules_pass = True
            detection_rules = parser_def.get("detection_rules")
            if not detection_rules: # No rules, only type match needed
                self.logger.debug(f"Parser definition {parser_def['parser_name']} has no detection rules, considering for type match.")
            else:
                # Handle AND/OR conditions at the top level of rules array if present
                # For now, assuming simple list of ANDed rules, or one rule with internal OR
                for rule in detection_rules:
                    # TODO: Add logic for top-level "condition": "OR" for rules array
                    if "condition" in rule and rule["condition"].upper() == "OR" and "rules" in rule: # Nested OR block
                        passed_or_block = any(self._evaluate_detection_rule(sub_rule, context_data) for sub_rule in rule["rules"])
                        if not passed_or_block:
                            all_rules_pass = False; break
                    elif "condition" in rule and rule["condition"].upper() == "AND" and "rules" in rule: # Nested AND block
                        passed_and_block = all(self._evaluate_detection_rule(sub_rule, context_data) for sub_rule in rule["rules"])
                        if not passed_and_block:
                             all_rules_pass = False; break
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

        # --- New Parsing Logic based on chosen_parser_def ---
        if "parsing_instructions" in chosen_parser_def:
            self.logger.info(f"Using JSON-defined parsing instructions for {chosen_parser_def['parser_name']}.")
            instructions = chosen_parser_def["parsing_instructions"]
            extracted_fields: Dict[str, Any] = {"parameters": {}} # Initialize for nested params
            current_input_data_for_fields: Any = None
            original_input_str_for_template: Optional[str] = None # For $INPUT_STRING_ORIGINAL_CHUNK

            # 1. Prepare input_data for field extraction
            input_data_def = instructions.get("input_data", {})
            source_options = input_data_def.get("source_options", [])
            if not source_options and input_data_def.get("source_type"): # Handle single source
                source_options = [input_data_def]

            for src_opt in source_options:
                src_type = src_opt.get("source_type")
                src_key = src_opt.get("source_key")
                if src_type == "pil_info_key": current_input_data_for_fields = context_data["pil_info"].get(src_key)
                elif src_type == "exif_user_comment": current_input_data_for_fields = context_data.get("raw_user_comment_str")
                elif src_type == "xmp_string_content": current_input_data_for_fields = context_data.get("xmp_string")
                elif src_type == "file_content_raw_text": current_input_data_for_fields = context_data.get("raw_file_content_text")
                elif src_type == "file_content_json_object": current_input_data_for_fields = context_data.get("parsed_root_json_object")
                # ... more sources
                if current_input_data_for_fields is not None:
                    original_input_str_for_template = str(current_input_data_for_fields) if isinstance(current_input_data_for_fields, str) else None
                    break # Use first found source

            if current_input_data_for_fields is None and input_data_def : # No specific input, but instructions exist
                 self.logger.debug(f"No specific input_data found for {chosen_parser_def['parser_name']}, field methods must use context or be static.")


            # Apply transformations to current_input_data_for_fields
            transformations = input_data_def.get("transformations", [])
            for transform in transformations:
                transform_type = transform.get("type")
                if transform_type == "json_decode_string_value" and isinstance(current_input_data_for_fields, str):
                    try:
                        json_obj = json.loads(current_input_data_for_fields)
                        current_input_data_for_fields = self._json_path_get(json_obj, transform.get("path"))
                    except json.JSONDecodeError:
                        self.logger.warning("Failed to JSON decode input_data for transformation.")
                        current_input_data_for_fields = None; break
                elif transform_type == "json_decode_string_itself" and isinstance(current_input_data_for_fields, str):
                    try:
                        current_input_data_for_fields = json.loads(current_input_data_for_fields)
                    except json.JSONDecodeError:
                        self.logger.warning("Failed to JSON decode input_data (itself) for transformation.")
                        current_input_data_for_fields = None; break
            # Store the processed input if it's an object, for $INPUT_JSON_OBJECT_AS_STRING
            if isinstance(current_input_data_for_fields, (dict, list)):
                extracted_fields["_input_data_object_for_template"] = current_input_data_for_fields


            # 2. Execute field extractions
            for field_def in instructions.get("fields", []):
                target_key_path = field_def.get("target_key")
                value = self._execute_field_extraction_method(field_def, current_input_data_for_fields, context_data, extracted_fields)

                if value is not None or field_def.get("optional", False) is False : # Store if value found or not optional (even if None)
                    # Handle nested target_key (e.g., "parameters.steps")
                    keys = target_key_path.split('.')
                    current_dict = extracted_fields
                    for i, key_segment in enumerate(keys[:-1]):
                        current_dict = current_dict.setdefault(key_segment, {})
                    current_dict[keys[-1]] = value
                # Cache the extracted value if needed for subsequent 'source_variable_key'
                extracted_fields[target_key_path.replace('.', '_')] = value # Store flattened for easy var access


            # 3. Construct output from template
            output_template = chosen_parser_def.get("output_template")
            if output_template:
                final_output = self._substitute_template_vars(output_template, extracted_fields, context_data, original_input_str_for_template)
                # Ensure base width/height from context are used if not set by template/fields
                if "parameters" in final_output and isinstance(final_output["parameters"], dict):
                    if final_output["parameters"].get("width") is None and context_data.get("width", 0) > 0:
                        final_output["parameters"]["width"] = context_data["width"]
                    if final_output["parameters"].get("height") is None and context_data.get("height", 0) > 0:
                        final_output["parameters"]["height"] = context_data["height"]
                return final_output # This is a Dict
            else:
                # No template, return the raw extracted_fields (needs cleanup of helper keys)
                extracted_fields.pop("_input_data_object_for_template", None)
                # ... remove other helper/flattened keys ...
                return extracted_fields # This is a Dict

        # --- Fallback to Python BaseFormat class instantiation (existing logic) ---
        elif "base_format_class" in chosen_parser_def:
            self.logger.info(f"Using Python class-based parser {chosen_parser_def['base_format_class']} for {chosen_parser_def['parser_name']}.")
            parser_class_name = chosen_parser_def["base_format_class"]
            ParserClass = get_parser_class_by_name(parser_class_name)

            if not ParserClass:
                self.logger.error(f"Python class '{parser_class_name}' not found in registry for {chosen_parser_def['parser_name']}.")
                return None

            raw_input_for_parser = ""
            # ... (existing logic for determining raw_input_for_parser from primary_data_source_for_raw) ...
            primary_data_def = chosen_parser_def.get("primary_data_source_for_raw", {})
            pds_type = primary_data_def.get("source_type")
            pds_key = primary_data_def.get("source_key")

            if pds_type == "png_chunk" and pds_key:
                raw_input_for_parser = context_data["png_chunks"].get(pds_key, "")
            elif pds_type == "exif_user_comment":
                raw_input_for_parser = context_data.get("raw_user_comment_str", "")
            elif pds_type == "xmp_string_content":
                raw_input_for_parser = context_data.get("xmp_string", "")


            self.logger.debug(f"Instantiating {ParserClass.__name__} with raw (len {len(raw_input_for_parser)}), and full context as info.")
            parser_instance = ParserClass(
                info=context_data,
                raw=raw_input_for_parser,
                width=str(context_data["width"]),
                height=str(context_data["height"]),
                logger_obj=self.logger
            )
            parser_status = parser_instance.parse()
            if parser_status == BaseFormat.Status.READ_SUCCESS:
                self.logger.info(f"Python Parser {parser_instance.tool} succeeded.")
                return parser_instance # This is a BaseFormat instance
            else:
                status_name = parser_status.name if hasattr(parser_status, "name") else str(parser_status)
                self.logger.warning(f"Python Parser {parser_instance.tool} did not succeed. Status: {status_name}. Error: {parser_instance.error}")
                return None
        else:
            self.logger.error(f"Parser definition {chosen_parser_def['parser_name']} has neither 'parsing_instructions' nor 'base_format_class'. Cannot parse.")
            return None

# Example usage (conceptual)
if __name__ == "__main__":
    # Setup for testing:
    # 1. Create a dummy logger
    # 2. Create a 'parser_definitions' directory with some of your JSON files
    # 3. Register any BaseFormat classes needed by those JSONs if using the Python fallback
    # from .vendored_sdpr.format import A1111 # Example
    # register_parser_class("A1111", A1111)

    logging.basicConfig(level=logging.DEBUG)
    test_logger = logging.getLogger("TestEngine")
    # engine = MetadataEngine(parser_definitions_path="./parser_definitions", logger_obj=test_logger)

    # test_image_path = "path/to/your/test_image.png" # or .json, .txt
    # result = engine.get_parser_for_file(test_image_path)
    # if result:
    #     if isinstance(result, BaseFormat):
    #         test_logger.info(f"Parsed with Python class: {result.tool}")
    #         test_logger.info(f"Prompt: {result.positive[:100]}")
    #         test_logger.info(f"Parameters: {result.parameter}")
    #     elif isinstance(result, dict):
    #         test_logger.info(f"Parsed with JSON instructions. Tool: {result.get('tool')}")
    #         test_logger.info(f"Prompt: {str(result.get('prompt'))[:100]}")
    #         test_logger.info(f"Parameters: {result.get('parameters')}")
    # else:
    #     test_logger.warning(f"Could not parse file: {test_image_path}")
    pass