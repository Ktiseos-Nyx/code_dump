import json
import re
# from typing import Any # Uncomment if using older Python and need typing.Any

from .metadata_utils import json_path_get_utility  # IMPORT THE UTILITY

class RuleEvaluator:
    def __init__(self, logger):  # ONLY takes logger
        self.logger = logger
        # NO self.engine here

    def _get_source_data_and_status(self, rule: dict, context_data: dict) -> tuple[any, bool]:
        source_type = rule.get("source_type")
        source_key = rule.get("source_key")
        line_number = rule.get("line_number")
        iptc_field_name = rule.get("iptc_field_name") # For context_iptc_field_value

        data_to_check: any = None
        source_found_successfully = True

        if source_type == "pil_info_key":
            pil_info = context_data.get("pil_info", {}) # Ensure pil_info exists
            data_to_check = pil_info.get(source_key)
            if data_to_check is None: source_found_successfully = False
        elif source_type == "png_chunk":
            png_chunks = context_data.get("png_chunks", {}) # Ensure png_chunks exists
            data_to_check = png_chunks.get(source_key)
            if data_to_check is None: source_found_successfully = False
        elif source_type == "exif_software_tag" or source_type == "software_tag":
            data_to_check = context_data.get("software_tag")
            if data_to_check is None: source_found_successfully = False
        elif source_type == "exif_user_comment":
            data_to_check = context_data.get("raw_user_comment_str")
            if data_to_check is None: source_found_successfully = False
        elif source_type == "xmp_string_content":
            data_to_check = context_data.get("xmp_string")
            if data_to_check is None: source_found_successfully = False
        elif source_type == "file_format":
            data_to_check = context_data.get("file_format")
            if data_to_check is None: source_found_successfully = False
        elif source_type == "file_extension":
            data_to_check = context_data.get("file_extension")
            if data_to_check is None: source_found_successfully = False
        elif source_type == "raw_file_content_text":
            data_to_check = context_data.get("raw_file_content_text")
            if data_to_check is None: source_found_successfully = False
        elif source_type == "direct_context_key":
            data_to_check = context_data.get(source_key)
            if data_to_check is None: source_found_successfully = False
        elif source_type == "direct_context_key_path_value":
            data_to_check = json_path_get_utility(context_data, source_key)
            if data_to_check is None: source_found_successfully = False
        elif source_type == "auto_detect_parameters_or_usercomment":
            pil_info = context_data.get("pil_info", {})
            param_str = pil_info.get("parameters")
            uc_str = context_data.get("raw_user_comment_str")
            data_to_check = param_str if param_str else uc_str
            if data_to_check is None: source_found_successfully = False
        elif source_type == "file_content_line":
            raw_text = context_data.get("raw_file_content_text")
            if isinstance(raw_text, str) and line_number is not None:
                lines = raw_text.splitlines()
                if 0 <= line_number < len(lines):
                    data_to_check = lines[line_number]
                else:
                    source_found_successfully = False
            else:
                source_found_successfully = False
        elif source_type == "a1111_parameter_string_content":
            temp_str = None
            pil_info = context_data.get("pil_info", {})
            param_chunk_str = pil_info.get("parameters")
            if isinstance(param_chunk_str, str):
                temp_str = param_chunk_str
            if temp_str is None:
                user_comment_str = context_data.get("raw_user_comment_str")
                if isinstance(user_comment_str, str):
                    temp_str = user_comment_str
            if temp_str is not None:
                try:
                    potential_json_wrapper = json.loads(temp_str)
                    if isinstance(potential_json_wrapper, dict) and \
                       "parameters" in potential_json_wrapper and \
                       isinstance(potential_json_wrapper["parameters"], str):
                        data_to_check = potential_json_wrapper["parameters"]
                        self.logger.debug("RuleEvaluator: Source 'a1111_parameter_string_content': unwrapped from JSON.")
                    else:
                        data_to_check = temp_str
                        self.logger.debug("RuleEvaluator: Source 'a1111_parameter_string_content': used as-is (was JSON but not wrapper).")
                except json.JSONDecodeError:
                    data_to_check = temp_str
                    self.logger.debug("RuleEvaluator: Source 'a1111_parameter_string_content': used as-is (not JSON).")
                # Removed redundant check for data_to_check is None as it's handled by subsequent logic
            else:
                self.logger.debug("RuleEvaluator: Source 'a1111_parameter_string_content': No source string found.")
                source_found_successfully = False
        
        # --- IMPLEMENTATION FOR pil_info_key_or_exif_user_comment_json_path ---
        elif source_type == "pil_info_key_or_exif_user_comment_json_path":
            pil_source_key = rule.get("source_key")
            json_path_to_extract = rule.get("json_path")

            if not pil_source_key:
                self.logger.warning(f"RuleEvaluator: Source type '{source_type}' requires 'source_key' in the rule definition.")
                source_found_successfully = False
            elif not json_path_to_extract:
                self.logger.warning(f"RuleEvaluator: Source type '{source_type}' requires 'json_path' in the rule definition.")
                source_found_successfully = False
            else:
                raw_json_str = None
                pil_info = context_data.get("pil_info", {})
                if isinstance(pil_info, dict):
                    raw_json_str = pil_info.get(pil_source_key)

                if raw_json_str is None:
                    raw_json_str = context_data.get("raw_user_comment_str")

                if raw_json_str is not None:
                    if isinstance(raw_json_str, str):
                        try:
                            parsed_json_obj = json.loads(raw_json_str)
                            data_to_check = json_path_get_utility(parsed_json_obj, json_path_to_extract)
                            # source_found_successfully remains True even if json_path_get_utility returns None (path not found)
                            # The operator (e.g. "exists") will determine the rule's outcome.
                        except json.JSONDecodeError:
                            self.logger.debug(f"RuleEvaluator: Source '{source_type}', data is not valid JSON. Data: '{str(raw_json_str)[:100]}...'")
                            source_found_successfully = False # Source was found, but unparsable for this type.
                        except Exception as e_path: # Catch errors from json_path_get_utility
                            self.logger.error(f"RuleEvaluator: Source '{source_type}', error applying JSONPath '{json_path_to_extract}': {e_path}", exc_info=True)
                            source_found_successfully = False
                    else:
                        self.logger.debug(f"RuleEvaluator: Source '{source_type}', expected string data but found {type(raw_json_str)}.")
                        source_found_successfully = False
                else:
                    self.logger.debug(f"RuleEvaluator: Source '{source_type}', no data found in pil_info['{pil_source_key}'] or raw_user_comment.")
                    source_found_successfully = False
        # --- End of pil_info_key_or_exif_user_comment_json_path ---

        elif source_type == "pil_info_pil_mode":
            data_to_check = context_data.get("pil_mode")
            if data_to_check is None:
                self.logger.debug("RuleEvaluator: 'pil_mode' not found in context_data for 'pil_info_pil_mode'.")
                source_found_successfully = False
        elif source_type == "pil_info_object":
            data_to_check = context_data.get("pil_info")
            if not isinstance(data_to_check, dict) or not data_to_check: # Check if it's a non-empty dict
                self.logger.debug(f"RuleEvaluator: 'pil_info' context data is not a valid/non-empty dictionary for 'pil_info_object'. Type: {type(data_to_check)}")
                source_found_successfully = False
                data_to_check = None # Ensure data_to_check is None if source not valid
        elif source_type == "context_iptc_field_value":
            iptc_source_dict = context_data.get("iptc_data_pil", context_data.get("parsed_iptc", {}))
            if isinstance(iptc_source_dict, dict) and iptc_field_name:
                data_to_check = iptc_source_dict.get(iptc_field_name)
                if isinstance(data_to_check, list) and len(data_to_check) == 1: # Unwrap single-item lists
                    data_to_check = data_to_check[0]
                # Ensure data is serializable or a common type if it's not already.
                # This could be more specific based on expected IPTC field types.
                if data_to_check is not None and not isinstance(data_to_check, (str, int, float, bool, dict, list, bytes)):
                    try:
                        data_to_check = str(data_to_check)
                    except Exception: # Broad catch if str() fails for some exotic object
                        self.logger.warning(f"RuleEvaluator: Could not convert IPTC field '{iptc_field_name}' value of type {type(data_to_check)} to string.")
                        data_to_check = None # or some placeholder
            if data_to_check is None:
                source_found_successfully = False
            self.logger.debug(f"RuleEvaluator: Source 'context_iptc_field_value' for field '{iptc_field_name}' got: {str(data_to_check)[:70]}")
        elif source_type in [
            "file_content_json", "pil_info_key_json_path",
            "pil_info_key_json_path_query", "file_content_json_path_value",
        ]:
            if source_type == "pil_info_key_json_path" or source_type == "pil_info_key_json_path_query":
                pil_info = context_data.get("pil_info", {})
                data_to_check = pil_info.get(source_key)
                if data_to_check is None: source_found_successfully = False
            elif source_type == "file_content_json":
                # Data is the parsed JSON object itself, handled by operator
                data_to_check = context_data.get("parsed_root_json_object")
                if data_to_check is None: source_found_successfully = False
            elif source_type == "file_content_json_path_value":
                root_json = context_data.get("parsed_root_json_object")
                if root_json is None:
                    source_found_successfully = False
                else:
                    json_path_for_value = rule.get("json_path")
                    if json_path_for_value:
                        data_to_check = json_path_get_utility(root_json, json_path_for_value)
                        # source_found_successfully remains true if root_json was present,
                        # even if path doesn't yield a value (data_to_check will be None).
                    else:
                        self.logger.warning(f"RuleEvaluator: Source type '{source_type}' needs 'json_path' in rule.")
                        source_found_successfully = False
            # No specific debug for data_to_check is None here, as operators handle it.
        else:
            # This 'else' will catch truly unknown source_types, including None if not handled by an explicit check.
            if source_type is None:
                 self.logger.warning(f"RuleEvaluator: Encountered a rule with 'source_type: None' or missing. Rule details: {rule.get('comment', 'N/A')}")
            else:
                self.logger.warning(f"RuleEvaluator: Unknown source_type in detection rule: {source_type}")
            source_found_successfully = False

        return data_to_check, source_found_successfully

    def _apply_operator(self, operator: str, data_to_check: any, rule: dict, context_data: dict) -> bool:
        # Parameters extracted from 'rule'
        expected_value = rule.get("value")
        expected_keys = rule.get("expected_keys")
        regex_pattern = rule.get("regex_pattern")
        regex_patterns = rule.get("regex_patterns")
        json_path = rule.get("json_path")
        json_query_type = rule.get("json_query_type")
        class_types_to_check = rule.get("class_types_to_check")
        value_list = rule.get("value_list")
        source_type = rule.get("source_type") # Available for context if needed by operator

        try:
            if operator == "exists":
                return data_to_check is not None
            elif operator == "not_exists":
                return data_to_check is None
            # ... (your existing operators, ensure they handle `data_to_check` being None appropriately) ...
            
            # --- json_contains_any_key operator ---
            elif operator == "json_contains_any_key":
                target_json_obj_for_keys = None
                if isinstance(data_to_check, dict):
                    target_json_obj_for_keys = data_to_check
                elif isinstance(data_to_check, str):
                    try:
                        target_json_obj_for_keys = json.loads(data_to_check)
                    except json.JSONDecodeError:
                        self.logger.debug(f"RuleEvaluator: Op '{operator}', data_to_check string not valid JSON.")
                        return False
                
                if not isinstance(target_json_obj_for_keys, dict):
                    self.logger.debug(f"RuleEvaluator: Op '{operator}', target for key check is not a dictionary (was {type(data_to_check)}, source type: {source_type}).")
                    return False

                if not expected_keys or not isinstance(expected_keys, list):
                    self.logger.warning(f"RuleEvaluator: Op '{operator}' needs a non-empty list for 'expected_keys' in rule.")
                    return False
                
                if not expected_keys: # Explicitly handle empty list if necessary, though `any` over empty is False.
                    self.logger.debug(f"RuleEvaluator: Op '{operator}', 'expected_keys' list is empty. Returning False.")
                    return False

                return any(k in target_json_obj_for_keys for k in expected_keys)

            # --- exists_and_is_dictionary operator (ensure only one definition) ---
            elif operator == "exists_and_is_dictionary":
                is_dict = isinstance(data_to_check, dict)
                is_not_empty = bool(data_to_check) if is_dict else False # True if dict is not empty
                self.logger.debug(f"RuleEvaluator: Op '{operator}': data type {type(data_to_check)}, is_dict={is_dict}, is_not_empty={is_not_empty}")
                return is_dict and is_not_empty # Checks it's a dictionary AND it's not empty
            
            # ... (your other existing operators) ...

            # Ensure this is the last elif before the final else
            elif operator == "json_path_value_equals":
                if not json_path: self.logger.warning("RuleEvaluator: Op 'json_path_value_equals': 'json_path' not provided."); return False
                target_obj = None
                if isinstance(data_to_check, (dict, list)): target_obj = data_to_check
                elif isinstance(data_to_check, str):
                    try: target_obj = json.loads(data_to_check)
                    except json.JSONDecodeError: self.logger.debug("RuleEvaluator: Op 'json_path_value_equals': data string not valid JSON."); return False
                else: self.logger.debug("RuleEvaluator: Op 'json_path_value_equals': data not suitable for JSON path."); return False
                
                if target_obj is None: # If data_to_check was, for example, an int, not JSON or dict/list
                    self.logger.debug("RuleEvaluator: Op 'json_path_value_equals': target object for path extraction is None.")
                    return False # Cannot apply JSONPath to None

                value_at_path = json_path_get_utility(target_obj, json_path)
                
                # Careful comparison for None vs actual values
                if value_at_path is None and expected_value is None: return True
                if value_at_path is None or expected_value is None: return False # One is None, the other isn't
                return str(value_at_path).strip() == str(expected_value).strip()


            # This 'else' should be the VERY LAST condition in the operator chain
            else:
                self.logger.warning(f"RuleEvaluator: Operator '{operator}' is not implemented or recognized. Rule: {rule.get('comment', 'Unnamed')}")
                return False

        except Exception as e_op:
            self.logger.error(
                f"RuleEvaluator: Error evaluating op '{operator}' for rule '{rule.get('comment', 'Unnamed rule')}': {e_op}",
                exc_info=True,
            )
            return False

    def evaluate_rule(self, rule: dict, context_data: dict) -> bool:
        # Ensure 'operator' exists, default to 'exists' if not specified for some reason.
        operator_for_rule = rule.get("operator", "exists") 
        
        data_to_check, source_found = self._get_source_data_and_status(rule, context_data)

        # If the source itself was not found (e.g., key missing, file part missing),
        # then only operators that explicitly check for non-existence should pass.
        # All other operators implicitly require the source data to be present.
        if not source_found:
            # If source was not found, only 'not_exists' or 'is_none' can be true.
            # 'exists' and other operators that require data will be false.
            if operator_for_rule == "not_exists" or operator_for_rule == "is_none":
                 # data_to_check will be None if source_found is False
                return self._apply_operator(operator_for_rule, data_to_check, rule, context_data)
            else:
                rule_comment = rule.get('comment', f"source_type: {rule.get('source_type')}, operator: {operator_for_rule}")
                self.logger.debug(f"RuleEvaluator: Source data not found or invalid for rule '{rule_comment}', operator '{operator_for_rule}' cannot proceed positively.")
                return False # Source not found, and operator isn't 'not_exists' or 'is_none'.

        # Source was found (or was irrelevant for the source_type), proceed to apply operator
        return self._apply_operator(operator_for_rule, data_to_check, rule, context_data)