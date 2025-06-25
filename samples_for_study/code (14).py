# In MetadataEngine._prepare_context_data
# ...
                    uc_bytes = loaded_exif.get("Exif", {}).get(piexif.ExifIFD.UserComment)
                    if uc_bytes:
                        decoded_comment = None
                        try:
                            # First, try piexif's standard way
                            decoded_comment = piexif.helper.UserComment.load(uc_bytes)
                            # Basic check: if it looks like it might still be mojibake (e.g., lots of \x00 if not expected)
                            # This is heuristic and might need refinement.
                            if decoded_comment and '\x00' in decoded_comment[1::2] and len(decoded_comment) > 20: # Heuristic for lingering wide-char issues
                                self.logger.debug("piexif.helper.UserComment.load result for UserComment might still have encoding issues, attempting fallbacks.")
                                raise ValueError("Potential piexif decoding issue") # Force fallback
                        except (ValueError, UnicodeDecodeError, Exception) as e_piexif_decode:
                            self.logger.debug(f"piexif.helper.UserComment.load failed or produced suspicious result: {e_piexif_decode}. Trying manual decodes.")
                            # Fallback decoding attempts for UserComment if piexif struggles
                            # The order of these attempts matters.
                            encodings_to_try = [
                                ('utf-16-le', b'UNICODE\x00'), # Check for explicit prefix first
                                ('utf-16-be', b'UNICODE\x00'), # Less common for UserComment
                                ('utf-8', b'UTF-8\x00\x00\x00\x00'),
                                ('ascii', b'ASCII\x00\x00\x00'),
                                # If no standard prefix, try common encodings directly on the bytes after a potential prefix
                                ('utf-16-le', None), # Try without assuming standard 8-byte prefix
                                ('utf-8', None),
                                ('latin-1', None), # Common fallback
                                ('cp1252', None)
                            ]
                            
                            processed_uc_bytes = uc_bytes
                            found_standard_prefix = False
                            for _, prefix_bytes_val in encodings_to_try:
                                if prefix_bytes_val and uc_bytes.startswith(prefix_bytes_val):
                                    processed_uc_bytes = uc_bytes[len(prefix_bytes_val):] # Strip known prefix
                                    found_standard_prefix = True
                                    break
                            
                            # If no standard prefix was found, but it starts with typical wide-char nulls,
                            # it might be UTF-16 without the "UNICODE" prefix.
                            if not found_standard_prefix and len(uc_bytes) > 1 and uc_bytes[0] != 0 and uc_bytes[1] == 0:
                                self.logger.debug("UserComment has no standard prefix but looks like UTF-16LE, trying direct decode.")
                                try:
                                    decoded_comment = uc_bytes.decode('utf-16-le', errors='replace').strip('\x00').strip()
                                except UnicodeDecodeError: pass

                            if decoded_comment is None: # If direct UTF-16LE attempt failed or wasn't applicable
                                for encoding, prefix_bytes in encodings_to_try:
                                    current_bytes_to_decode = processed_uc_bytes if prefix_bytes else uc_bytes # Use stripped or original
                                    try:
                                        decoded_comment = current_bytes_to_decode.decode(encoding, errors='strict').strip('\x00').strip()
                                        self.logger.debug(f"Successfully decoded UserComment with fallback: {encoding}")
                                        break # Success
                                    except (UnicodeDecodeError, Exception):
                                        continue # Try next encoding
                            
                            if decoded_comment is None: # If all fallbacks fail
                                self.logger.warning(f"Could not decode UserComment after multiple attempts. Storing as repr or placeholder. Original bytes preview: {uc_bytes[:50]!r}")
                                decoded_comment = f"[Undecodable UserComment: {len(uc_bytes)} bytes]"

                        context["raw_user_comment_str"] = decoded_comment.strip() if decoded_comment else None
# ...