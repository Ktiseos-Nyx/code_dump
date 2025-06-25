# Dataset-Tools/metadata_parser.py
# ... (other imports remain largely the same) ...

# NEW: Import MetadataEngine
from .metadata_engine import MetadataEngine
# BaseFormat is still needed if MetadataEngine returns BaseFormat instances as a fallback
from .vendored_sdpr.format.base_format import BaseFormat # Or wherever it's defined

# ... (make_paired_str_dict, _populate_ui_from_specific_parser, process_pyexiv2_data stay) ...
# _populate_ui_from_vendored_reader might become less used or adapted if ImageDataReader is phased out for initial parse.

# NEW HELPER: To convert the dict output from MetadataEngine (JSON-instructed parse)
# into the UpField/DownField structure if needed, or directly use it.
def _populate_ui_from_engine_dict(engine_dict_output: dict[str, Any], ui_dict_to_update: dict[str, Any]):
    """
    Populates the UI dictionary from the direct dictionary output of MetadataEngine
    when it used JSON parsing_instructions.
    This assumes engine_dict_output is already structured somewhat like final_ui_dict.
    """
    tool_name = engine_dict_output.get("tool", "Unknown (Engine JSON)")
    nfo(f"[DT._populate_ui_engine_dict] Populating UI from MetadataEngine (JSON-driven). Tool: {tool_name}")

    # Prompts
    prompt_data = ui_dict_to_update.get(UpField.PROMPT.value, {})
    if "prompt" in engine_dict_output and engine_dict_output["prompt"] is not None: # Check for None
        prompt_data["Positive"] = str(engine_dict_output["prompt"])
    if "negative_prompt" in engine_dict_output and engine_dict_output["negative_prompt"] is not None:
        prompt_data["Negative"] = str(engine_dict_output["negative_prompt"])
    # Add SDXL prompt handling if your engine_dict_output template includes them
    # e.g., if engine_dict_output["parameters"]["sdxl_prompts"]["positive_g"] etc.
    if prompt_data:
        ui_dict_to_update[UpField.PROMPT.value] = prompt_data

    # Generation Data & Parameters
    gen_data = ui_dict_to_update.get(DownField.GENERATION_DATA.value, {})
    engine_params = engine_dict_output.get("parameters", {})
    if isinstance(engine_params, dict):
        for key, value in engine_params.items():
            if key == "tool_specific": # Handle nested tool_specific parameters
                if isinstance(value, dict):
                    for ts_key, ts_value in value.items():
                        if ts_value is not None and ts_value != PARAMETER_PLACEHOLDER:
                            display_key = ts_key.replace("_", " ").capitalize() + " (Tool Specific)"
                            gen_data[display_key] = str(ts_value)
                continue # Move to next main parameter

            if value is not None and value != PARAMETER_PLACEHOLDER:
                display_key = key.replace("_", " ").capitalize()
                gen_data[display_key] = str(value)
    if gen_data:
        ui_dict_to_update[DownField.GENERATION_DATA.value] = gen_data

    # Raw Data (if the template includes it, e.g., $INPUT_STRING_ORIGINAL_CHUNK)
    # Or if the engine specifically adds a "raw_data_from_engine" key
    if "workflow" in engine_dict_output and engine_dict_output["workflow"] is not None: # ComfyUI example
        ui_dict_to_update[DownField.RAW_DATA.value] = str(engine_dict_output["workflow"])
    elif "source_raw_input_string" in engine_dict_output and engine_dict_output["source_raw_input_string"] is not None: # Fooocus example
        ui_dict_to_update[DownField.RAW_DATA.value] = str(engine_dict_output["source_raw_input_string"])
    # Add other raw data sources as needed based on your output_templates

    # Metadata (Tool Name)
    if UpField.METADATA.value not in ui_dict_to_update:
        ui_dict_to_update[UpField.METADATA.value] = {}
    ui_dict_to_update[UpField.METADATA.value]["Detected Tool"] = tool_name
    if "parser_name_from_engine" in engine_dict_output: # If engine adds this for clarity
         ui_dict_to_update[UpField.METADATA.value]["Engine Parser Def"] = engine_dict_output["parser_name_from_engine"]


def parse_metadata(file_path_named: str) -> dict[str, Any]:
    final_ui_dict: dict[str, Any] = {}
    path_obj = Path(file_path_named)
    file_ext_lower = path_obj.suffix.lower()
    # is_txt_file = file_ext_lower == ".txt" # Engine will determine type based on content/ext
    potential_ai_parsed = False
    pyexiv2_raw_data: dict[str, Any] | None = None
    # description_text_from_pyexiv2 = "" # Will be handled by engine context or specific parsers

    placeholder_key_str: str
    try:
        placeholder_key_str = EmptyField.PLACEHOLDER.value
    except AttributeError:
        nfo("CRITICAL [DT.metadata_parser]: EmptyField.PLACEHOLDER.value not accessible. Using fallback key.")
        placeholder_key_str = "_dt_internal_placeholder_"

    nfo(f"[DT.metadata_parser]: >>> ENTERING parse_metadata for: {file_path_named} (NEW ENGINE FLOW)")

    # --- STEP 1: Primary Parse Attempt with MetadataEngine ---
    engine = MetadataEngine(parser_definitions_path="dataset_tools/parser_definitions") # Ensure path is correct
    engine_result = None
    try:
        engine_result = engine.get_parser_for_file(file_path_named) # Can also pass file object
    except Exception as e_engine_call:
        nfo(f"CRITICAL ERROR calling MetadataEngine: {e_engine_call}", exc_info=True)
        final_ui_dict[placeholder_key_str] = {"Error": f"MetadataEngine execution error: {e_engine_call}"}
        # Potentially return here or try old system as ultimate fallback

    if engine_result:
        parser_name_from_def = "Unknown" # Get from engine_result if it's a dict
        if isinstance(engine_result, dict) and "parser_name_from_engine" in engine_result : # A field we can add in engine
             parser_name_from_def = engine_result["parser_name_from_engine"]

        nfo(f"[DT.metadata_parser]: MetadataEngine returned a result. Type: {type(engine_result).__name__}. Parser Def: {parser_name_from_def}")
        if isinstance(engine_result, BaseFormat): # Python class fallback was used by engine
            if engine_result.status == BaseFormat.Status.READ_SUCCESS:
                nfo(f"  Engine used Python class: {engine_result.tool}, success.")
                _populate_ui_from_specific_parser(engine_result, final_ui_dict)
                potential_ai_parsed = True
            else:
                nfo(f"  Engine used Python class: {engine_result.tool}, but it reported status: {engine_result.status.name if hasattr(engine_result.status, 'name') else engine_result.status}. Error: {engine_result.error}")
                # Decide if this is a hard fail or if we should try other things.
                # For now, assume if engine picked a Python parser and it failed, that's the result from engine.
                final_ui_dict[placeholder_key_str] = {"Error": f"Engine Parser ({engine_result.tool}) Error: {engine_result.error or 'Unknown error'}"}

        elif isinstance(engine_result, dict): # Parsed by JSON instructions
            # engine_result is already the structured output.
            # We need a new helper to populate final_ui_dict from this engine_result dict.
            nfo(f"  Engine used JSON instructions. Tool in result: {engine_result.get('tool')}")
            _populate_ui_from_engine_dict(engine_result, final_ui_dict)
            potential_ai_parsed = True # Assume success if dict is returned
        else:
            nfo(f"  Engine returned unexpected data type: {type(engine_result)}. Treating as no parse.")
            final_ui_dict[placeholder_key_str] = {"Error": "MetadataEngine returned unknown data type."}
    else:
        nfo("[DT.metadata_parser]: MetadataEngine found no suitable parser or failed to parse.")
        # NO FALLBACK TO ImageDataReader here if engine is primary.
        # If you want ImageDataReader as fallback, this is where its logic would go.
        # For now, if engine fails, we proceed to pyexiv2 for non-AI metadata.
        # Consider what error message to show if engine_result is None.
        # if not final_ui_dict.get(placeholder_key_str): # only set if no other error
        #    final_ui_dict[placeholder_key_str] = {"Error": "No AI metadata parser matched via engine."}
        pass


    # --- Step 2 (Optional Fallback): Standard EXIF/XMP with pyexiv2 (if not a text/JSON/model file handled by engine) ---
    # The engine might already extract some of this via a "GenericXMP" or "GenericEXIF" parser definition.
    # This block remains useful if you want to ensure standard photo metadata is always added,
    # or if the engine doesn't yet have generic EXIF/XMP parsers.
    # We need to know if the file was a type that pyexiv2 can handle.
    # `file_ext_lower` is still useful here.

    # Check if it's an image type that pyexiv2 might handle AND AI parsing didn't happen or was minimal
    is_std_image_format = file_ext_lower in [".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"]
    # if not potential_ai_parsed and is_std_image_format: # Only if AI parse failed
    if is_std_image_format: # Always try to get standard photo meta for images
        nfo("[DT.metadata_parser]: Attempting to read standard photo EXIF/XMP with pyexiv2.")
        std_reader = MetadataFileReader()
        if file_ext_lower.endswith((".jpg", ".jpeg", ".webp")):
            pyexiv2_raw_data = std_reader.read_jpg_header_pyexiv2(file_path_named)
        elif file_ext_lower.endswith((".png", ".tif", ".tiff")): # Added TIFF
            pyexiv2_raw_data = std_reader.read_png_header_pyexiv2(file_path_named) # (or a new tiff_header func)

        if pyexiv2_raw_data:
            standard_photo_meta = process_pyexiv2_data(pyexiv2_raw_data, ai_tool_parsed=potential_ai_parsed)
            if standard_photo_meta:
                # Merge this carefully with what engine might have produced
                for key, value_map in standard_photo_meta.items():
                    if key not in final_ui_dict:
                        final_ui_dict[key] = value_map
                    elif isinstance(final_ui_dict.get(key), dict) and isinstance(value_map, dict):
                        for sub_key, sub_value in value_map.items():
                             # Prioritize already parsed AI data over generic EXIF descriptions if there's a clash
                            if sub_key == "Description" and "Positive" in final_ui_dict.get(UpField.PROMPT.value, {}):
                                if "Description (XMP)" not in final_ui_dict[key]: # Add as separate if not present
                                     final_ui_dict[key]["Description (XMP - Std)"] = sub_value
                                continue
                            if sub_key not in final_ui_dict[key]:
                                final_ui_dict[key][sub_key] = sub_value
                nfo("[DT.metadata_parser]: Added/merged standard EXIF/XMP data (via pyexiv2).")
        elif not potential_ai_parsed and not final_ui_dict.get(placeholder_key_str): # If still no error and no data
             final_ui_dict[placeholder_key_str] = {
                "Info": "Standard image, but no processable EXIF/XMP fields found by pyexiv2."
            }

    # --- Step 3: Midjourney Specific Check (SHOULD BE HANDLED BY METADATA ENGINE JSON DEF NOW) ---
    # This block should ideally be removed if MetadataEngine has a robust Midjourney JSON definition.
    # If kept as a fallback, it needs to be carefully conditioned.
    # For now, assume engine handles it. Commenting out:
    # if VENDORED_SDPR_OK and MidjourneyFormat and not is_txt_file:
    #     ... (your existing MidjourneyFormat logic) ...

    # --- Final Checks and Return ---
    if not final_ui_dict or (len(final_ui_dict) == 1 and placeholder_key_str in final_ui_dict and not potential_ai_parsed):
        if not (placeholder_key_str in final_ui_dict and "Error" in final_ui_dict.get(placeholder_key_str, {})):
            final_ui_dict.setdefault(placeholder_key_str, {}).update({"Info": "No processable metadata found after all attempts."})
        nfo(f"Failed to find/load significant metadata for file: {file_path_named}")
    elif not final_ui_dict.get(UpField.PROMPT.value) and not final_ui_dict.get(DownField.GENERATION_DATA.value) and not final_ui_dict.get(DownField.JSON_DATA.value):
        # If we have some metadata (like EXIF only) but no AI params or gen data.
        if not (placeholder_key_str in final_ui_dict and "Error" in final_ui_dict.get(placeholder_key_str, {})):
            # Check if it was a model file parsed successfully
            if final_ui_dict.get(UpField.METADATA.value, {}).get("Detected Model Format"):
                 pass # Model file, no prompt/gen data expected in that way
            else:
                final_ui_dict.setdefault(placeholder_key_str, {}).update({"Info": "No AI generation parameters found. Displaying other metadata."})


    nfo(f"[DT.metadata_parser]: <<< EXITING parse_metadata. Returning keys: {list(final_ui_dict.keys())}")
    return final_ui_dict