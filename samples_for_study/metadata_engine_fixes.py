# dataset_tools/metadata_engine.py

import contextlib
import json
import logging
import re
from pathlib import Path
from typing import Any, BinaryIO

import piexif  # type: ignore
import piexif.helper  # type: ignore
from PIL import Image, UnidentifiedImageError  # type: ignore
from PIL.PngImagePlugin import PngInfo  # type: ignore

# Assuming these are in the same package or accessible
from .logger import get_logger
from .metadata_utils import get_a1111_kv_block_utility, json_path_get_utility
from .parser_registry import get_parser_class_by_name
from .rule_evaluator import RuleEvaluator
from .vendored_sdpr.format.base_format import BaseFormat


class MetadataEngine:
    def __init__(
        self,
        parser_definitions_path: str | Path,
        logger_obj: logging.Logger | None = None,
    ) -> None:
        self.parser_definitions_path = Path(parser_definitions_path)
        if logger_obj:
            self.logger = logger_obj
        else:
            self.logger = get_logger("MetadataEngine")
        self.rule_evaluator = RuleEvaluator(self.logger)

        self.parser_definitions: list[dict[str, Any]] = self._load_parser_definitions()
        self.sorted_definitions: list[dict[str, Any]] = sorted(
            self.parser_definitions, key=lambda p: p.get("priority", 0), reverse=True
        )
        self.logger.info(
            f"Loaded {len(self.sorted_definitions)} parser definitions from {self.parser_definitions_path}"
        )

    def _load_parser_definitions(self) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = []
        if not self.parser_definitions_path.is_dir():
            self.logger.error(f"Parser definitions path is not a directory: {self.parser_definitions_path}")
            return definitions

        for filepath in self.parser_definitions_path.glob("*.json"):
            self.logger.debug(f"Loading parser definition from: {filepath.name}")
            try:
                with open(filepath, encoding="utf-8") as f:
                    definition = json.load(f)
                    if "parser_name" in definition:
                        definitions.append(definition)
                    else:
                        self.logger.warning(
                            f"Skipping invalid parser definition (missing parser_name): {filepath.name}"
                        )
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to decode JSON from {filepath.name}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error loading parser definition {filepath.name}: {e}")
        return definitions

    def _prepare_context_data(
        self, file_path_or_obj: str | Path | BinaryIO
    ) -> dict[str, Any] | None:
        context: dict[str, Any] = {
            "pil_info": {},
            "exif_dict": {},
            "xmp_string": None,
            "parsed_xmp_dict": None,
            "png_chunks": {},
            "file_format": "",
            "width": 0,
            "height": 0,
            "raw_user_comment_str": None,
            "software_tag": None,
            "file_extension": "",
            "raw_file_content_text": None,
            "raw_file_content_bytes": None,
            "parsed_root_json_object": None,
            "safetensors_metadata": None,
            "safetensors_main_header": None,
            "gguf_metadata": None,
            "gguf_main_header": None,
            "file_path_original": str(
                file_path_or_obj.name
                if hasattr(file_path_or_obj, "name") and file_path_or_obj.name
                else str(file_path_or_obj)
            ),
        }
        is_binary_io = hasattr(file_path_or_obj, "read") and hasattr(file_path_or_obj, "seek")
        original_file_path_str = context["file_path_original"]

        try:
            img = Image.open(file_path_or_obj)
            context["pil_info"] = img.info.copy() if img.info else {}
            context["width"] = img.width
            context["height"] = img.height
            context["file_format"] = img.format.upper() if img.format else ""
            image_filename = getattr(img, "filename", None)
            if image_filename:
                context["file_extension"] = Path(image_filename).suffix.lstrip(".").lower()
            elif isinstance(file_path_or_obj, (str, Path)):
                context["file_extension"] = Path(str(file_path_or_obj)).suffix.lstrip(".").lower()

            if exif_bytes := context["pil_info"].get("exif"):
                try:
                    loaded_exif = piexif.load(exif_bytes)
                    context["exif_dict"] = loaded_exif
                    uc_bytes = loaded_exif.get("Exif", {}).get(piexif.ExifIFD.UserComment)
                    if uc_bytes:
                        context["raw_user_comment_str"] = piexif.helper.UserComment.load(uc_bytes)
                    sw_bytes = loaded_exif.get("0th", {}).get(piexif.ImageIFD.Software)
                    if sw_bytes and isinstance(sw_bytes, bytes):
                        context["software_tag"] = sw_bytes.decode("ascii", "ignore").strip("\x00").strip()
                except Exception as e_exif:
                    self.logger.debug(f"piexif failed to load EXIF or extract specific tags: {e_exif}")

            if xmp_str := context["pil_info"].get("XML:com.adobe.xmp"):
                context["xmp_string"] = xmp_str

            for key, val in context["pil_info"].items():
                if isinstance(val, str):
                    context["png_chunks"][key] = val
            if "UserComment" in context["pil_info"] and "UserComment" not in context["png_chunks"]:
                context["png_chunks"]["UserComment"] = context["pil_info"]["UserComment"]
            img.close()

        except FileNotFoundError:
            self.logger.error(f"File not found: {original_file_path_str}")
            return None
        except UnidentifiedImageError:
            self.logger.info(f"Cannot identify as image: {original_file_path_str}. Checking for non-image types.")
            p = Path(original_file_path_str)
            context["file_extension"] = p.suffix.lstrip(".").lower()
            context["file_format"] = context["file_extension"].upper()

            def read_file_content(mode="r", encoding="utf-8", errors="replace"):
                if is_binary_io:
                    file_path_or_obj.seek(0)  # type: ignore
                    content = file_path_or_obj.read()  # type: ignore
                    if "b" in mode:
                        return content
                    return content.decode(encoding, errors=errors) if isinstance(content, bytes) else content
                else:
                    with open(
                        p,
                        mode,
                        encoding=encoding if "b" not in mode else None,
                        errors=errors if "b" not in mode else None,
                    ) as f_obj:
                        return f_obj.read()

            if context["file_extension"] == "json":
                try:
                    content_str = read_file_content(mode="r", encoding="utf-8")
                    context["parsed_root_json_object"] = json.loads(content_str)
                    context["raw_file_content_text"] = content_str
                except (
                    json.JSONDecodeError,
                    OSError,
                    UnicodeDecodeError,
                    TypeError,
                ) as e_json_direct:
                    self.logger.error(f"Failed to read/parse direct JSON file {p.name}: {e_json_direct}")
                    if not context["raw_file_content_text"]:
                        with contextlib.suppress(Exception):
                            context["raw_file_content_text"] = read_file_content(
                                mode="r", encoding="utf-8", errors="replace"
                            )
            elif context["file_extension"] == "txt":
                try:
                    context["raw_file_content_text"] = read_file_content(mode="r", encoding="utf-8", errors="replace")
                except (OSError, UnicodeDecodeError, TypeError) as e_txt:
                    self.logger.error(f"Failed to read TXT file {p.name}: {e_txt}")
                    return None
            elif context["file_extension"] == "safetensors":
                try:
                    from dataset_tools.model_parsers.safetensors_parser import ModelParserStatus, SafetensorsParser

                    temp_parser = SafetensorsParser(original_file_path_str)
                    status = temp_parser.parse()
                    if status == ModelParserStatus.SUCCESS:
                        context["safetensors_metadata"] = temp_parser.metadata_header
                        context["safetensors_main_header"] = temp_parser.main_header
                    elif status == ModelParserStatus.FAILURE:
                        self.logger.warning(f"Safetensors parser failed for {p.name}: {temp_parser._error_message}")
                except ImportError:
                    self.logger.error("SafetensorsParser not available or ModelParserStatus not found.")
                except Exception as e_st:
                    self.logger.error(f"Error during safetensors context prep: {e_st}")

            elif context["file_extension"] == "gguf":
                try:
                    from dataset_tools.model_parsers.gguf_parser import GGUFParser, ModelParserStatus

                    temp_parser = GGUFParser(original_file_path_str)
                    status = temp_parser.parse()
                    if status == ModelParserStatus.SUCCESS:
                        context["gguf_metadata"] = temp_parser.metadata_header
                        context["gguf_main_header"] = temp_parser.main_header
                    elif status == ModelParserStatus.FAILURE:
                        self.logger.warning(f"GGUF parser failed for {p.name}: {temp_parser._error_message}")
                except ImportError:
                    self.logger.error("GGUFParser not available or ModelParserStatus not found.")
                except Exception as e_gguf:
                    self.logger.error(f"Error during GGUF context prep: {e_gguf}")
            else:
                self.logger.info(
                    f"File {p.name} extension '{context['file_extension']}' not specifically handled in non-image path."
                )
                with contextlib.suppress(Exception):
                    context["raw_file_content_bytes"] = read_file_content(mode="rb")

        except Exception as e:
            self.logger.error(
                f"Error preparing context data for {original_file_path_str}: {e}",
                exc_info=True,
            )
            return None

        self.logger.debug(
            f"Prepared context for {original_file_path_str}. Keys: {list(k for k, v in context.items() if v is not None)}"
        )
        return context

    def _execute_field_extraction_method(
        self,
        method_def: dict[str, Any],
        current_input_data: Any,
        context_data: dict[str, Any],
        extracted_fields_cache: dict[str, Any],
    ) -> Any:
        method_name = method_def.get("method")
        value: Any = None

        def get_source_data(source_definition, mtd_input_data, ctx_data, fields_cache):
            if not source_definition:
                return mtd_input_data
            src_type = source_definition.get("type")
            src_key = source_definition.get("key")
            if src_type == "pil_info_key":
                return ctx_data.get("pil_info", {}).get(src_key)
            if src_type == "png_chunk":
                return ctx_data.get("png_chunks", {}).get(src_key)
            if src_type == "exif_user_comment":
                return ctx_data.get("raw_user_comment_str")
            if src_type == "xmp_string":
                return ctx_data.get("xmp_string")
            if src_type == "file_content_raw_text":
                return ctx_data.get("raw_file_content_text")
            if src_type == "file_content_json_object":
                return ctx_data.get("parsed_root_json_object")
            if src_type == "direct_context_key":
                return ctx_data.get(src_key)
            if src_type == "variable":
                variable_name_in_cache = src_key.replace(".", "_") + "_VAR_"
                return fields_cache.get(variable_name_in_cache)
            self.logger.warning(f"Unknown source_data_from_context type: {src_type}")
            return mtd_input_data

        data_for_method = get_source_data(
            method_def.get("source_data_from_context"), current_input_data, context_data, extracted_fields_cache
        )

        try:
            if method_name == "direct_json_path":
                value = json_path_get_utility(data_for_method, method_def.get("json_path"))

            elif method_name == "static_value":
                value = method_def.get("value")

            elif method_name == "direct_context_value":
                value = data_for_method

            elif method_name == "direct_string_value":
                value = str(data_for_method) if data_for_method is not None else None

            elif method_name == "a1111_extract_prompt_positive":
                if isinstance(data_for_method, str):
                    neg_match = re.search(r"\nNegative prompt:", data_for_method, re.IGNORECASE)
                    kv_block_str = get_a1111_kv_block_utility(data_for_method)
                    end_index = len(data_for_method)
                    if neg_match:
                        end_index = min(end_index, neg_match.start())
                    if kv_block_str:
                        try:
                            first_line_of_kv = kv_block_str.split("\n", 1)[0].strip()
                            if first_line_of_kv:
                                kv_start_index_in_original = data_for_method.rfind(first_line_of_kv)
                                if kv_start_index_in_original != -1:
                                    end_index = min(end_index, kv_start_index_in_original)
                        except (ValueError, AttributeError):
                            pass
                    value = data_for_method[:end_index].strip()

            elif method_name == "a1111_extract_prompt_negative":
                if isinstance(data_for_method, str):
                    neg_match = re.search(
                        r"\nNegative prompt:(.*?)(?=(\n(?:Steps:|Sampler:|CFG scale:|Seed:|Size:|Model hash:|Model:|Version:|$)))",
                        data_for_method,
                        re.IGNORECASE | re.DOTALL,
                    )
                    value = neg_match.group(1).strip() if neg_match else ""

            elif method_name in ["key_value_extract_from_a1111_block", "key_value_extract_transform_from_a1111_block"]:
                if isinstance(data_for_method, str):
                    kv_block = get_a1111_kv_block_utility(data_for_method)
                    key_to_find = method_def.get("key_name")
                    if kv_block and key_to_find:
                        lookahead_pattern = r"(?:,\s*(?:Steps:|Sampler:|CFG scale:|Seed:|Size:|Model hash:|Model:|Version:|Clip skip:|Denoising strength:|Hires upscale:|Hires steps:|Hires upscaler:|Lora hashes:|TI hashes:|Emphasis:|NGMS:|ADetailer model:|Schedule type:))|$"
                        actual_key_pattern = re.escape(key_to_find)
                        match = re.search(
                            rf"{actual_key_pattern}:\s*(.*?)(?={lookahead_pattern})", kv_block, re.IGNORECASE
                        )
                        temp_val = match.group(1).strip() if match else None
                        if (
                            temp_val is not None
                            and method_name == "key_value_extract_transform_from_a1111_block"
                            and "transform_regex" in method_def
                        ):
                            transform_match = re.search(method_def["transform_regex"], temp_val)
                            value = (
                                transform_match.group(method_def.get("transform_group", 1))
                                if transform_match
                                else None
                            )
                        else:
                            value = temp_val

            elif method_name == "json_from_string_variable":
                source_var_key = method_def.get("source_variable_key")
                variable_name_in_cache = source_var_key.replace(".", "_") + "_VAR_"
                string_to_parse = extracted_fields_cache.get(variable_name_in_cache)

                value = None
                if isinstance(string_to_parse, str):
                    try:
                        value = json.loads(string_to_parse)
                        self.logger.debug(
                            f"Method '{method_name}': successfully parsed JSON from var '{variable_name_in_cache}'."
                        )
                    except json.JSONDecodeError as e:
                        self.logger.warning(
                            f"Method '{method_name}': Failed to parse string from var '{variable_name_in_cache}' as JSON. Error: {e}"
                        )
                elif string_to_parse is not None:
                    self.logger.warning(
                        f"Method '{method_name}': source var '{variable_name_in_cache}' is not a string (type: {type(string_to_parse)}), cannot parse."
                    )

            elif "comfy_" in (method_name or ""):
                self.logger.warning(f"Method '{method_name}' is a ComfyUI tool and is not fully implemented yet.")
                value = f"PLACEHOLDER FOR {method_name}"

            else:
                self.logger.warning(f"Unknown field extraction method: '{method_name}'")

        except Exception as e_method:
            self.logger.error(
                f"An unexpected error occurred inside method '{method_name}': {e_method}", exc_info=True
            )
            value = None

        value_type = method_def.get("value_type")
        if value is not None and value_type:
            try:
                if value_type == "integer":
                    value = int(float(str(value)))
                elif value_type == "float":
                    value = float(str(value))
            except (ValueError, TypeError):
                self.logger.debug(f"Could not convert value to '{value_type}'.")
                value = None

        return value

    def _comfy_traverse_for_field(
        self,
        workflow_graph: Any,
        node_criteria_list: list[dict[str, Any]] | None,
        target_input_key: str | None,
    ) -> Any:
        if not isinstance(workflow_graph, dict) or not node_criteria_list:
            self.logger.debug(
                f"ComfyUI traversal: Invalid workflow_graph or no node_criteria. Graph type:{type(workflow_graph)}"
            )
            return None
        graph_nodes = workflow_graph.get("nodes", {})
        if not isinstance(graph_nodes, dict) and isinstance(workflow_graph, dict) and all(k.isdigit() for k in workflow_graph):
            graph_nodes = workflow_graph
        for node_id, node_data in graph_nodes.items():
            if not isinstance(node_data, dict):
                continue
            for criterion in node_criteria_list:
                match = True
                if "class_type" in criterion and node_data.get("type") != criterion["class_type"]:
                    match = False
                if "node_id" in criterion and node_id != criterion["node_id"]:
                    match = False
                if match and target_input_key:
                    if node_data.get("type") == "KSampler" and target_input_key == "seed":
                        return node_data.get("widgets_values", [None])[0]
                    if node_data.get("type") == "CLIPTextEncode" and target_input_key == "text":
                        input_text_def = node_data.get("inputs", {}).get("text")
                        if not isinstance(input_text_def, list):
                            return node_data.get("widgets_values", [None])[0]
                    self.logger.debug(
                        f"ComfyUI placeholder: Matched node {node_id} ({node_data.get('type')}), complex logic for '{target_input_key}' not fully implemented."
                    )
                    return f"Placeholder for {target_input_key} from {node_data.get('type')}"
        return None

    def _substitute_template_vars(
        self,
        template: Any,
        extracted_data: dict[str, Any],
        context_data: dict[str, Any],
        original_input_data_str: str | None = None,
        input_json_object_for_template: Any | None = None,
    ) -> Any:
        if isinstance(template, dict):
            return {
                k: self._substitute_template_vars(
                    v, extracted_data, context_data, original_input_data_str, input_json_object_for_template
                )
                for k, v in template.items()
            }
        if isinstance(template, list):
            return [
                self._substitute_template_vars(
                    item, extracted_data, context_data, original_input_data_str, input_json_object_for_template
                )
                for item in template
            ]

        if isinstance(template, str):

            def replacer(match):
                var_path = match.group(1)
                value = json_path_get_utility(extracted_data, var_path)
                if value is not None:
                    return str(value)
                if var_path.startswith("CONTEXT."):
                    context_path = var_path.replace("CONTEXT.", "", 1)
                    context_value = json_path_get_utility(context_data, context_path)
                    if context_value is not None:
                        return str(context_value)
                if var_path == "INPUT_STRING_ORIGINAL_CHUNK" and original_input_data_str is not None:
                    return original_input_data_str
                if var_path == "INPUT_JSON_OBJECT_AS_STRING" and input_json_object_for_template is not None:
                    return json.dumps(input_json_object_for_template, indent=2)
                self.logger.debug(f"Template variable '${var_path}' not found, replacing with empty string.")
                return ""

            return re.sub(r"\$([\w.]+)", replacer, template)

        return template

    def get_parser_for_file(self, file_path_or_obj: str | Path | BinaryIO) -> Any | None:
        display_name = getattr(file_path_or_obj, "name", str(file_path_or_obj))
        self.logger.info(f"MetadataEngine: Starting metadata parsing for: {display_name}")

        context_data = self._prepare_context_data(file_path_or_obj)
        if not context_data:
            self.logger.warning(f"MetadataEngine: Could not prepare context data for {display_name}.")
            return None

        chosen_parser_def: dict[str, Any] | None = None
        for parser_def in self.sorted_definitions:
            target_types_cfg = parser_def.get("target_file_types", ["*"])
            if not isinstance(target_types_cfg, list):
                target_types_cfg = [str(target_types_cfg)]
            target_types = [ft.upper() for ft in target_types_cfg]
            current_file_format_upper = context_data.get("file_format", "").upper()
            current_file_ext_upper = context_data.get("file_extension", "").upper()
            type_match = (
                "*" in target_types
                or (current_file_format_upper and current_file_format_upper in target_types)
                or (current_file_ext_upper and current_file_ext_upper in target_types)
            )
            if not type_match:
                continue

            all_rules_pass = True
            detection_rules = parser_def.get("detection_rules", [])
            if not detection_rules:
                # No detection rules - type match is sufficient
                pass
            else:
                for rule_idx, rule in enumerate(detection_rules):
                    # Check for complex rule (AND/OR block)
                    if isinstance(rule, dict) and "condition" in rule and isinstance(rule.get("rules"), list):
                        condition_type = rule["condition"].upper()
                        sub_rules = rule.get("rules", [])
                        passed_block = False

                        if not sub_rules:
                            self.logger.warning(
                                f"Complex rule for {parser_def['parser_name']} has no sub-rules. Condition: {condition_type}"
                            )
                            all_rules_pass = False
                            break

                        if condition_type == "OR":
                            passed_block = any(
                                self.rule_evaluator.evaluate_rule(sub_rule, context_data) for sub_rule in sub_rules
                            )
                        elif condition_type == "AND":
                            passed_block = all(
                                self.rule_evaluator.evaluate_rule(sub_rule, context_data) for sub_rule in sub_rules
                            )
                        else:
                            self.logger.warning(
                                f"Unknown complex cond '{condition_type}' in rule {rule_idx} for {parser_def['parser_name']}"
                            )
                            passed_block = False

                        if not passed_block:
                            all_rules_pass = False
                            break

                    # Simple rule
                    elif not self.rule_evaluator.evaluate_rule(rule, context_data):
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

        if "parsing_instructions" in chosen_parser_def:
            self.logger.info(f"Using JSON-defined parsing instructions for {chosen_parser_def['parser_name']}.")
            instructions = chosen_parser_def["parsing_instructions"]
            extracted_fields: dict[str, Any] = {"parameters": {}}
            current_input_data_for_fields: Any = None
            original_input_for_template: Any | None = None

            # Handle input data
            input_data_def = instructions.get("input_data", {})
            source_options = input_data_def.get("source_options", [])
            if not source_options and input_data_def.get("source_type"):
                source_options = [input_data_def]

            for src_opt in source_options:
                src_type = src_opt.get("source_type")
                src_key = src_opt.get("source_key")
                if src_type == "pil_info_key":
                    current_input_data_for_fields = context_data["pil_info"].get(src_key)
                elif src_type == "exif_user_comment":
                    current_input_data_for_fields = context_data.get("raw_user_comment_str")
                elif src_type == "xmp_string_content":
                    current_input_data_for_fields = context_data.get("xmp_string")
                elif src_type == "file_content_raw_text":
                    current_input_data_for_fields = context_data.get("raw_file_content_text")
                elif src_type == "file_content_json_object":
                    current_input_data_for_fields = context_data.get("parsed_root_json_object")
                if current_input_data_for_fields is not None:
                    original_input_for_template = current_input_data_for_fields
                    break

            # Handle transformations
            transformations = input_data_def.get("transformations", [])
            for transform in transformations:
                transform_type = transform.get("type")
                if current_input_data_for_fields is None and transform_type not in ["create_if_not_exists"]:
                    break
                if transform_type == "json_decode_string_value" and isinstance(current_input_data_for_fields, str):
                    try:
                        json_obj = json.loads(current_input_data_for_fields)
                        current_input_data_for_fields = json_path_get_utility(json_obj, transform.get("path"))
                    except json.JSONDecodeError:
                        current_input_data_for_fields = None
                        break

                elif transform_type == "conditional_json_unwrap_parameters_string":
                    if isinstance(current_input_data_for_fields, str):
                        try:
                            potential_json_wrapper = json.loads(current_input_data_for_fields)
                            if (
                                isinstance(potential_json_wrapper, dict)
                                and "parameters" in potential_json_wrapper
                                and isinstance(potential_json_wrapper["parameters"], str)
                            ):
                                current_input_data_for_fields = potential_json_wrapper["parameters"]
                                self.logger.debug(
                                    "Applied 'conditional_json_unwrap_parameters_string': unwrapped parameters."
                                )
                        except json.JSONDecodeError:
                            pass

                elif transform_type == "json_decode_string_itself" and isinstance(current_input_data_for_fields, str):
                    try:
                        current_input_data_for_fields = json.loads(current_input_data_for_fields)
                    except json.JSONDecodeError:
                        current_input_data_for_fields = None
                        break

            input_json_object_for_template = (
                current_input_data_for_fields if isinstance(current_input_data_for_fields, (