# Add this function to your metadata_parser.py file
# This goes BEFORE the parse_metadata function

def detect_ai_tool_priority(file_path: str) -> str:
    """
    Pre-detection logic to determine which AI tool to prioritize.
    Returns a hint for the vendored SDPR or handles specific cases.
    """
    import json
    from pathlib import Path
    
    path_obj = Path(file_path)
    file_ext_lower = path_obj.suffix.lower()
    
    if file_ext_lower == ".txt":
        return "txt_file"
    
    # Quick metadata peek to make better decisions
    std_reader = MetadataFileReader()
    quick_metadata = None
    
    try:
        if file_ext_lower in [".jpg", ".jpeg", ".webp"]:
            quick_metadata = std_reader.read_jpg_header_pyexiv2(file_path)
        elif file_ext_lower == ".png":
            quick_metadata = std_reader.read_png_header_pyexiv2(file_path)
    except:
        return "unknown"
    
    if not quick_metadata:
        return "unknown"
    
    # Check EXIF UserComment for specific signatures
    exif_data = quick_metadata.get("EXIF", {})
    user_comment = exif_data.get("Exif.Photo.UserComment", "")
    
    # Convert bytes to string if needed
    comment_text = ""
    if isinstance(user_comment, bytes):
        if user_comment.startswith(b"ASCII\x00\x00\x00"):
            comment_text = user_comment[8:].decode("ascii", "replace")
        elif user_comment.startswith(b"UNICODE\x00"):
            comment_text = user_comment[8:].decode("utf-16", "replace")
        else:
            try:
                comment_text = user_comment.decode("utf-8", "replace")
            except:
                comment_text = ""
    elif isinstance(user_comment, str):
        comment_text = user_comment
    
    comment_text = comment_text.strip("\x00 ").strip()
    
    # PRIORITY ORDER (most specific first)
    
    # 1. Civitai has very specific format
    if comment_text and '"extraMetadata"' in comment_text:
        try:
            json.loads(comment_text)
            return "civitai_comfyui"
        except:
            pass
    
    # 2. RuinedFooocus - specific JSON structure
    if comment_text and comment_text.startswith('{"'):
        try:
            parsed = json.loads(comment_text)
            if "ruined_fooocus" in comment_text.lower() or any(
                key in parsed for key in ["base_model_name", "performance_selection"]
            ):
                return "ruined_fooocus"
        except:
            pass
    
    # 3. Check PNG chunks for ComfyUI workflow (most reliable)
    if file_ext_lower == ".png":
        # Look for ComfyUI workflow in PNG chunks
        # (You'd need to add PNG chunk reading here, or check if SDPR can handle this)
        pass
    
    # 4. Forge vs A1111 distinction (Forge usually has "Forge" in parameters)
    if "parameters:" in comment_text.lower() or "Steps:" in comment_text:
        if "forge" in comment_text.lower():
            return "forge"
        else:
            return "a1111"
    
    # 5. Yodayo specific patterns
    if "yodayo" in comment_text.lower() or '"platform":"yodayo"' in comment_text:
        return "yodayo"
    
    # 6. Generic Fooocus (JSON in comment but not the specific types above)
    if comment_text and comment_text.startswith('{') and comment_text.endswith('}'):
        try:
            json.loads(comment_text)
            return "fooocus_generic"
        except:
            pass
    
    return "unknown"


# Then modify your parse_metadata function around line 280-300
# Replace this section:

def parse_metadata(file_path_named: str) -> dict:
    final_ui_dict = {}
    path_obj = Path(file_path_named)
    file_ext_lower = path_obj.suffix.lower()
    is_txt_file = file_ext_lower == ".txt"
    potential_ai_parsed = False
    placeholder_key_str: str

    try:
        placeholder_key_str = EmptyField.PLACEHOLDER.value
    except AttributeError:
        nfo("CRITICAL [DT.metadata_parser]: EmptyField.PLACEHOLDER.value not accessible. Using fallback key.")
        placeholder_key_str = "_dt_internal_placeholder_"

    nfo(f"[DT.metadata_parser]: >>> ENTERING parse_metadata for: {file_path_named}")

    # ADD THIS NEW DETECTION LOGIC HERE:
    detected_tool_hint = detect_ai_tool_priority(file_path_named)
    nfo(f"[DT.metadata_parser]: Pre-detection suggests: {detected_tool_hint}")

    if VENDORED_SDPR_OK and ImageDataReader is not None and BaseFormat is not None:
        vendored_reader_instance = None
        try:
            nfo(f"[DT.metadata_parser]: Attempting to init VENDORED ImageDataReader (is_txt: {is_txt_file})")
            
            # Pass hint to SDPR if possible, or handle specific cases yourself
            if detected_tool_hint == "civitai_comfyui":
                # You might want to handle Civitai specially
                nfo("[DT.metadata_parser]: Detected Civitai format, using standard SDPR")
            elif detected_tool_hint in ["ruined_fooocus", "fooocus_generic"]:
                # Fooocus gets special handling to prevent it from eating everything
                nfo("[DT.metadata_parser]: Detected Fooocus variant, being careful with parsing")
            
            # Continue with the rest of your existing SDPR logic...
            if is_txt_file:
                with open(file_path_named, encoding="utf-8", errors="replace") as f_obj:
                    vendored_reader_instance = ImageDataReader(f_obj, is_txt=True)
            else:
                vendored_reader_instance = ImageDataReader(file_path_named)
                
            # Rest of your existing code continues unchanged...
