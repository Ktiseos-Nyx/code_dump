# dataset_tools/vendored_sdpr/image_data_reader.py
# This is YOUR VENDORED and MODIFIED copy - FUSED VERSION

__author__ = "receyuki"
__filename__ = "image_data_reader.py"
# MODIFIED by Ktiseos Nyx for Dataset-Tools
__copyright__ = "Copyright 2023, Receyuki"
__email__ = "receyuki@gmail.com"

import json
import logging
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import piexif
import piexif.helper
from defusedxml import minidom
from PIL import Image, UnidentifiedImageError

from .constants import PARAMETER_PLACEHOLDER
from .format import (
    A1111,
    CivitaiFormat,
    ComfyUI,
    DrawThings,
    EasyDiffusion,
    Fooocus,
    InvokeAI,
    MochiDiffusionFormat,
    NovelAI,
    RuinedFooocusFormat,
    SwarmUI,
)
from .format import TensorArtFormat as TensorArt
from .format import YodayoFormat as Yodayo
from .format import BaseFormat
from .logger import get_logger

# EXIF and IPTC tag constants
USER_COMMENT_TAG = 37510  # 0x9286
IMAGE_DESCRIPTION_TAG = 270  # 0x010e
IPTC_CAPTION_ABSTRACT = (2, 120)
IPTC_ORIGINATING_PROGRAM = (2, 65)
IPTC_PROGRAM_VERSION = (2, 70)


def fix_sjis_utf8_mojibake(mojibake_text: str) -> str:
    """
    Fixes a string that was UTF-8 but incorrectly decoded as Shift-JIS.
    
    Args:
        mojibake_text: The corrupted text string
        
    Returns:
        The corrected UTF-8 string, or original if correction fails
    """
    try:
        return mojibake_text.encode('shift_jis_2004').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return mojibake_text


def get_generation_parameters(file_path: str) -> str | None:
    """
    Robustly extracts AI generation parameters from an image file.
    
    This function provides a comprehensive fallback mechanism that checks
    multiple metadata locations across PNG, JPEG, and WEBP formats.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Extracted parameter string if found, None otherwise
    """
    try:
        with Image.open(file_path) as img:
            # Strategy 1: Check common PNG text chunks (most reliable)
            for chunk_key in ['parameters', 'prompt', 'workflow', 'Comment']:
                if chunk_key in img.info:
                    return img.info[chunk_key]

            # Strategy 2: Check JPEG/WEBP comment block
            comment_data = img.info.get('comment')
            if comment_data and isinstance(comment_data, bytes):
                return comment_data.decode('utf-8', 'ignore')

            # Strategy 3: Check standard EXIF UserComment tag
            exif_data = img.getexif()
            if USER_COMMENT_TAG in exif_data:
                user_comment = exif_data[USER_COMMENT_TAG]
                
                if isinstance(user_comment, bytes) and len(user_comment) > 8:
                    # Skip 8-byte EXIF header (e.g., b'UNICODE\x00')
                    return user_comment[8:].decode('utf-8', 'ignore')
                elif isinstance(user_comment, str):
                    return fix_sjis_utf8_mojibake(user_comment)

            # Strategy 4: Check image description as last resort
            if IMAGE_DESCRIPTION_TAG in exif_data:
                description = exif_data[IMAGE_DESCRIPTION_TAG]
                if isinstance(description, bytes):
                    return description.decode('utf-8', 'ignore')
                elif isinstance(description, str):
                    return description

    except (FileNotFoundError, UnidentifiedImageError):
        return None
    except Exception as e:
        logger = logging.getLogger("DSVendored_SDPR.ImageDataReader")
        logger.warning(f"get_generation_parameters error: {e}")
        return None

    return None


class ImageDataReader:
    """
    A comprehensive reader for AI-generated image metadata across multiple formats.
    
    This class handles extraction and parsing of generation parameters from images
    created by various AI tools including A1111, ComfyUI, Fooocus, and many others.
    It supports PNG, JPEG, and WEBP formats with robust fallback mechanisms.
    """
    
    NOVELAI_MAGIC = "stealth_pngcomp"
    
    # Parser order for PNG files - most specific first
    PARSER_CLASSES_PNG = [
        ComfyUI,           # Generic ComfyUI JSON format
        TensorArt,         # ComfyUI-based with specific patterns
        CivitaiFormat,     # ComfyUI or A1111 format
        RuinedFooocusFormat,  # Specific JSON with software tag
        SwarmUI,           # 'sui_image_params' format
        A1111,             # Standard A1111 text format
        Yodayo,            # A1111 variant
        InvokeAI,          # 'sd-metadata' chunks
        EasyDiffusion,     # JSON in UserComment
        DrawThings,        # XMP-based format
        NovelAI,           # LSB steganography
        MochiDiffusionFormat,  # IPTC-based
    ]
    
    # Parser order for JPEG/WEBP files
    PARSER_CLASSES_JPEG_WEBP = [
        CivitaiFormat,     # EXIF UserComment with ComfyUI/A1111
        TensorArt,         # ComfyUI JSON in UserComment
        Yodayo,            # A1111-like EXIF UserComment
        A1111,             # Generic A1111 EXIF UserComment
    ]

    def __init__(self, file_path_or_obj: str | Path | TextIO | BinaryIO, is_txt: bool = False):
        """
        Initialize the ImageDataReader.
        
        Args:
            file_path_or_obj: Path to file or file-like object
            is_txt: Whether the input is a text file (default: False)
        """
        self._initialize_attributes()
        self._is_txt = is_txt
        self._logger = get_logger("DSVendored_SDPR.ImageDataReader")
        self.read_data(file_path_or_obj)

    def _initialize_attributes(self) -> None:
        """Initialize all instance attributes to default values."""
        # Image properties
        self._height: int = 0
        self._width: int = 0
        self._format_str: str = ""
        
        # Metadata containers
        self._info: dict[str, Any] = {}
        self._parsed_iptc_info: dict[str, str] = {}
        self._exif_software_tag: str | None = None
        
        # Parsed content
        self._positive: str = ""
        self._negative: str = ""
        self._setting: str = ""
        self._raw: str = ""
        self._tool: str = ""
        
        # SDXL-specific content
        self._positive_sdxl: dict[str, Any] = {}
        self._negative_sdxl: dict[str, Any] = {}
        self._is_sdxl: bool = False
        
        # Parameters and status
        base_param_keys = getattr(BaseFormat, "PARAMETER_KEY", None)
        self._parameter_key: list[str] = (
            base_param_keys if isinstance(base_param_keys, list) 
            else ["model", "sampler", "seed", "cfg", "steps", "size"]
        )
        self._parameter: dict[str, Any] = dict.fromkeys(self._parameter_key, PARAMETER_PLACEHOLDER)
        
        # Parser state
        self._parser: BaseFormat | None = None
        self._status: BaseFormat.Status = BaseFormat.Status.UNREAD
        self._error: str = ""

    def read_data(self, file_path_or_obj: str | Path | TextIO | BinaryIO) -> None:
        """
        Main entry point for reading and parsing image or text data.
        
        Args:
            file_path_or_obj: File path or file-like object to read
        """
        self._reset_state()
        file_display_name = self._get_display_name(file_path_or_obj)
        
        self._logger.debug(
            "Reading data for: %s (is_txt: %s)", 
            file_display_name, 
            self._is_txt
        )
        
        try:
            if self._is_txt:
                self._handle_text_input(file_path_or_obj)
            else:
                self._handle_image_input(file_path_or_obj, file_display_name)
        except Exception as e:
            self._logger.error("Unexpected error in read_data: %s", e, exc_info=True)
            self._set_error_state(f"Unexpected error: {e}")
        
        self._finalize_reading(file_display_name)

    def _reset_state(self) -> None:
        """Reset all parsing state for a fresh read operation."""
        self._status = BaseFormat.Status.UNREAD
        self._error = ""
        self._parser = None
        self._tool = ""
        self._raw = ""
        self._info = {}
        self._parsed_iptc_info = {}
        self._width = 0
        self._height = 0
        self._positive = ""
        self._negative = ""
        self._setting = ""
        self._positive_sdxl = {}
        self._negative_sdxl = {}
        self._is_sdxl = False
        self._parameter = dict.fromkeys(self._parameter_key, PARAMETER_PLACEHOLDER)
        self._format_str = ""
        self._exif_software_tag = None

    def _handle_text_input(self, file_obj: str | Path | TextIO) -> None:
        """Handle text file input."""
        if hasattr(file_obj, "read") and callable(file_obj.read):
            self._process_text_file(file_obj)
        else:
            try:
                with open(file_obj, encoding="utf-8") as f:
                    self._process_text_file(f)
            except Exception as e:
                self._logger.error("Error opening text file: %s", e)
                self._set_error_state(f"Could not open text file: {e}")

    def _process_text_file(self, file_obj: TextIO) -> None:
        """Process the content of a text file."""
        try:
            raw_content = file_obj.read()
            if not self._try_parser(A1111, raw=raw_content):
                self._set_error_state("Failed to parse text file as A1111 format")
            if not self._parser:
                self._raw = raw_content
        except Exception as e:
            self._logger.error("Error reading text file content: %s", e, exc_info=True)
            self._set_error_state(f"Could not read text file content: {e}")

    def _handle_image_input(self, file_path_or_obj: str | Path | BinaryIO, display_name: str) -> None:
        """Handle image file input."""
        try:
            with Image.open(file_path_or_obj) as img:
                self._extract_basic_image_info(img, display_name)
                self._extract_metadata(img)
                self._attempt_parsing(img)
                self._apply_fallback_if_needed(file_path_or_obj)
                
        except FileNotFoundError:
            self._set_error_state("Image file not found", BaseFormat.Status.FORMAT_ERROR)
        except UnidentifiedImageError as e:
            self._set_error_state(f"Cannot identify image: {e}", BaseFormat.Status.FORMAT_ERROR)
        except OSError as e:
            self._logger.error("OS/IO error opening image '%s': %s", display_name, e, exc_info=True)
            self._set_error_state(f"File system error: {e}", BaseFormat.Status.FORMAT_ERROR)
        except Exception as e:
            self._logger.error("Error processing image '%s': %s", display_name, e, exc_info=True)
            self._set_error_state(f"Processing error: {e}", BaseFormat.Status.FORMAT_ERROR)

    def _extract_basic_image_info(self, img: Image.Image, display_name: str) -> None:
        """Extract basic image properties."""
        self._width = img.width
        self._height = img.height
        self._info = img.info.copy() if img.info else {}
        self._format_str = img.format or ""
        
        self._logger.debug(
            "Image opened: %s, Format: %s, Size: %sx%s",
            display_name, self._format_str, self._width, self._height
        )

    def _extract_metadata(self, img: Image.Image) -> None:
        """Extract EXIF and IPTC metadata from the image."""
        exif_bytes = self._info.get("exif")
        if not exif_bytes:
            return
            
        try:
            exif_data = piexif.load(exif_bytes)
            self._extract_software_tag(exif_data)
            self._extract_iptc_data(exif_data)
        except Exception as e:
            self._logger.debug("Could not parse EXIF for metadata: %s", e)

    def _extract_software_tag(self, exif_data: dict) -> None:
        """Extract software tag from EXIF data."""
        software_bytes = exif_data.get("0th", {}).get(piexif.ImageIFD.Software)
        if software_bytes and isinstance(software_bytes, bytes):
            self._exif_software_tag = software_bytes.decode("ascii", "ignore").strip("\x00").strip()
            self._logger.debug("Found EXIF:Software tag: %s", self._exif_software_tag)

    def _extract_iptc_data(self, exif_data: dict) -> None:
        """Extract IPTC data from EXIF."""
        iptc_dict = exif_data.get("IPTC", {})
        if not iptc_dict:
            return
            
        # Extract common IPTC fields for MochiDiffusion and others
        for iptc_key, internal_key in [
            (IPTC_CAPTION_ABSTRACT, "iptc_caption_abstract"),
            (IPTC_ORIGINATING_PROGRAM, "iptc_originating_program"),
            (IPTC_PROGRAM_VERSION, "iptc_program_version"),
        ]:
            if iptc_key in iptc_dict:
                value = iptc_dict[iptc_key]
                if isinstance(value, bytes):
                    self._parsed_iptc_info[internal_key] = value.decode("utf-8", "ignore")
                elif isinstance(value, str):
                    self._parsed_iptc_info[internal_key] = value

    def _attempt_parsing(self, img: Image.Image) -> None:
        """Attempt to parse the image using format-specific parsers."""
        self._attempt_legacy_swarm_exif(img)
        
        if not self._parser:
            if self._format_str == "PNG":
                self._process_png_format(img)
            elif self._format_str in ["JPEG", "WEBP"]:
                self._process_jpeg_webp_format(img)
            else:
                self._process_generic_format()

    def _attempt_legacy_swarm_exif(self, img: Image.Image) -> None:
        """Check for legacy SwarmUI format in EXIF tag 0x0110."""
        if self._parser:
            return
            
        try:
            exif_pil = img.getexif()
            exif_json_str = exif_pil.get(0x0110) if exif_pil else None
            
            if not exif_json_str:
                return
                
            if isinstance(exif_json_str, bytes):
                exif_json_str = exif_json_str.decode("utf-8", errors="ignore")
                
            if "sui_image_params" in exif_json_str:
                try:
                    exif_dict = json.loads(exif_json_str)
                    if isinstance(exif_dict, dict) and "sui_image_params" in exif_dict:
                        self._try_parser(SwarmUI, info=exif_dict)
                except json.JSONDecodeError as e:
                    self._logger.debug("SwarmUI legacy EXIF: Invalid JSON: %s", e)
        except Exception as e:
            self._logger.debug("SwarmUI legacy EXIF check failed: %s", e)

    def _process_png_format(self, img: Image.Image) -> None:
        """Process PNG-specific formats and chunks."""
        pil_info = self._info
        
        # Log available chunks for debugging
        chunk_info = {
            key: bool(pil_info.get(key)) 
            for key in ["parameters", "Comment", "workflow", "prompt", "UserComment", "XML:com.adobe.xmp"]
        }
        self._logger.debug("PNG Chunks found: %s", chunk_info)

        # Try each PNG parser in order
        for parser_class in self.PARSER_CLASSES_PNG:
            if self._parser:
                break
                
            kwargs = self._prepare_png_parser_kwargs(parser_class, pil_info)
            if kwargs is not None:
                self._try_parser(parser_class, **kwargs)

        # Special post-processing for formats that need it
        if not self._parser:
            self._handle_special_png_formats(img, pil_info)

    def _prepare_png_parser_kwargs(self, parser_class: type, pil_info: dict) -> dict | None:
        """Prepare kwargs for a specific PNG parser class."""
        kwargs = {
            "info": pil_info.copy(),
            "width": self._width,
            "height": self._height,
        }
        
        # Determine the most appropriate raw data for each parser
        raw_data = self._get_raw_data_for_parser(parser_class, pil_info)
        if raw_data is not None:
            kwargs["raw"] = raw_data
            self._logger.debug(
                "Passing raw data (len %d) to %s", 
                len(raw_data), 
                parser_class.__name__
            )
        
        return kwargs

    def _get_raw_data_for_parser(self, parser_class: type, pil_info: dict) -> str | None:
        """Get the most appropriate raw data chunk for a specific parser."""
        # ComfyUI-based parsers
        if parser_class is ComfyUI:
            return pil_info.get("prompt") or pil_info.get("workflow")
        elif parser_class in [TensorArt, CivitaiFormat]:
            return (pil_info.get("UserComment") or 
                   pil_info.get("parameters") or 
                   pil_info.get("workflow"))
        
        # Comment-based parsers
        elif parser_class in [Fooocus, RuinedFooocusFormat]:
            return pil_info.get("Comment")
        
        # SwarmUI specific
        elif parser_class is SwarmUI:
            params = pil_info.get("parameters", "")
            return params if "sui_image_params" in params else None
        
        # A1111-style parsers
        elif parser_class in [Yodayo, A1111]:
            return pil_info.get("parameters") or pil_info.get("UserComment")
        
        # Other parsers handle their own data extraction
        return None

    def _handle_special_png_formats(self, img: Image.Image, pil_info: dict) -> None:
        """Handle special PNG formats that need custom processing."""
        # DrawThings XMP processing
        xmp_chunk = pil_info.get("XML:com.adobe.xmp")
        if xmp_chunk and not self._parser:
            self._parse_drawthings_xmp(xmp_chunk)
        
        # NovelAI LSB steganography
        if img.mode == "RGBA" and not self._parser:
            self._parse_novelai_lsb(img)
        
        # Final A1111 fallback
        parameters_chunk = pil_info.get("parameters")
        if parameters_chunk and not self._parser:
            self._logger.debug("Trying A1111 final fallback on 'parameters' chunk")
            self._try_parser(A1111, raw=parameters_chunk, info=pil_info.copy())

    def _process_jpeg_webp_format(self, img: Image.Image) -> None:
        """Process JPEG/WEBP specific formats."""
        user_comment = self._extract_user_comment()
        jfif_comment = self._extract_jfif_comment()
        
        # Special handling for RuinedFooocus
        if self._try_ruined_fooocus_parsing(user_comment):
            return
        
        # Try standard JPEG/WEBP parsers
        for parser_class in self.PARSER_CLASSES_JPEG_WEBP:
            if self._parser:
                break
                
            kwargs = self._prepare_jpeg_parser_kwargs(parser_class, user_comment)
            if kwargs is not None:
                self._try_parser(parser_class, **kwargs)
        
        # Try JFIF comment for Fooocus
        if not self._parser and jfif_comment:
            self._try_fooocus_jfif_parsing(jfif_comment)
        
        # NovelAI LSB as last resort
        if img.mode == "RGBA" and not self._parser:
            self._parse_novelai_lsb(img)

    def _extract_user_comment(self) -> str | None:
        """Extract UserComment from EXIF data."""
        exif_bytes = self._info.get("exif")
        if not exif_bytes:
            return None
            
        try:
            exif_dict = piexif.load(exif_bytes)
            user_comment_bytes = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
            if user_comment_bytes:
                return piexif.helper.UserComment.load(user_comment_bytes)
        except Exception:
            self._logger.debug("Could not load UserComment via piexif")
        
        return None

    def _extract_jfif_comment(self) -> str:
        """Extract JFIF comment from image info."""
        comment_bytes = self._info.get("comment", b"")
        if isinstance(comment_bytes, bytes):
            return comment_bytes.decode("utf-8", "ignore")
        return ""

    def _try_ruined_fooocus_parsing(self, user_comment: str | None) -> bool:
        """Try parsing as RuinedFooocus format."""
        if not user_comment or not user_comment.strip().startswith("{"):
            return False
            
        try:
            comment_json = json.loads(user_comment)
            if (isinstance(comment_json, dict) and 
                comment_json.get("software") == "RuinedFooocus"):
                return self._try_parser(RuinedFooocusFormat, raw=user_comment)
        except json.JSONDecodeError:
            pass
        
        return False

    def _prepare_jpeg_parser_kwargs(self, parser_class: type, user_comment: str | None) -> dict | None:
        """Prepare kwargs for JPEG/WEBP parsers."""
        if parser_class is MochiDiffusionFormat:
            caption = self._parsed_iptc_info.get("iptc_caption_abstract", "")
            return {"raw": caption} if caption else None
        elif user_comment:
            return {"raw": user_comment}
        
        return None

    def _try_fooocus_jfif_parsing(self, jfif_comment: str) -> None:
        """Try parsing JFIF comment as Fooocus format."""
        try:
            jfif_data = json.loads(jfif_comment)
            if isinstance(jfif_data, dict) and "prompt" in jfif_data:
                self._try_parser(Fooocus, info=jfif_data)
        except json.JSONDecodeError:
            self._logger.debug("JFIF Comment not valid JSON or not Fooocus")

    def _process_generic_format(self) -> None:
        """Process generic image formats using EXIF UserComment fallback."""
        self._logger.info(
            "Image format '%s' not specifically handled. Checking generic EXIF UserComment.",
            self._format_str
        )
        
        exif_bytes = self._info.get("exif")
        if not exif_bytes:
            return
            
        try:
            exif_dict = piexif.load(exif_bytes)
            user_comment_bytes = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
            if user_comment_bytes:
                user_comment = piexif.helper.UserComment.load(user_comment_bytes)
                if user_comment and not user_comment.strip().startswith("{"):
                    self._try_parser(A1111, raw=user_comment)
        except Exception as e:
            self._logger.debug("Generic EXIF UserComment check failed: %s", e)

    def _apply_fallback_if_needed(self, file_path_or_obj: str | Path | BinaryIO) -> None:
        """Apply robust fallback parsing if primary parsers failed."""
        if self._parser:
            return
            
        self._logger.debug(
            "Primary parsers failed. Attempting robust fallback with get_generation_parameters."
        )
        
        clean_text = get_generation_parameters(file_path_or_obj)
        if clean_text:
            self._logger.info("Fallback succeeded. Trying A1111 parser on clean text.")
            self._try_parser(A1111, raw=clean_text)
        else:
            self._logger.debug("Robust fallback also failed to find any text.")

    def _parse_drawthings_xmp(self, xmp_chunk: str) -> None:
        """Parse DrawThings XMP metadata."""
        if self._parser:
            return
            
        try:
            xmp_dom = minidom.parseString(xmp_chunk)
            description_nodes = xmp_dom.getElementsByTagName("rdf:Description")
            
            for desc_node in description_nodes:
                user_comment_data = self._extract_xmp_user_comment(desc_node)
                if user_comment_data:
                    if self._try_parser(DrawThings, raw=user_comment_data.strip()):
                        return
        except (minidom.ExpatError, json.JSONDecodeError) as e:
            self._logger.warning("DrawThings PNG XMP parse error: %s", e)
        except Exception as e:
            self._logger.warning("DrawThings PNG XMP unexpected error: %s", e, exc_info=True)

    def _extract_xmp_user_comment(self, desc_node) -> str | None:
        """Extract UserComment data from XMP description node."""
        user_comment_nodes = desc_node.getElementsByTagName("exif:UserComment")
        if not user_comment_nodes or not user_comment_nodes[0].childNodes:
            return None
            
        first_child = user_comment_nodes[0].childNodes[0]
        
        if first_child.nodeType == first_child.TEXT_NODE:
            return first_child.data
        elif first_child.nodeName == "rdf:Alt":
            li_nodes = first_child.getElementsByTagName("rdf:li")
            if (li_nodes and li_nodes[0].childNodes and 
                li_nodes[0].childNodes[0].nodeType == li_nodes[0].TEXT_NODE):
                return li_nodes[0].childNodes[0].data
        
        return None

    def _parse_novelai_lsb(self, img: Image.Image) -> None:
        """Parse NovelAI LSB steganography."""
        if self._parser:
            return
            
        try:
            extractor = NovelAI.LSBExtractor(img)
            if not extractor.lsb_bytes_list:
                self._logger.debug("NovelAI LSB: Extractor found no data")
                return
                
            magic_bytes = extractor.get_next_n_bytes(len(self.NOVELAI_MAGIC))
            if magic_bytes and magic_bytes.decode("utf-8", "ignore") == self.NOVELAI_MAGIC:
                self._try_parser(NovelAI, extractor=extractor)
            else:
                self._logger.debug("NovelAI LSB: Magic bytes not found")
        except Exception as e:
            self._logger.warning("NovelAI LSB check error: %s", e, exc_info=True)

    def _try_parser(self, parser_class: type[BaseFormat], **kwargs: Any) -> bool:
        """
        Attempt to parse using a specific parser class.
        
        Args:
            parser_class: The parser class to try
            **kwargs: Arguments to pass to the parser
            
        Returns:
            True if parsing succeeded, False otherwise
        """
        # Prepare kwargs for logging (truncate long raw data)
        log_kwargs = self._prepare_kwargs_for_logging(kwargs)
        self._logger.debug(
            "Attempting parser: %s with kwargs: [%s]",
            parser_class.__name__,
            ", ".join(log_kwargs)
        )
        
        try:
            # Add standard kwargs if not provided
            self._add_standard_kwargs(kwargs)
            
            # Consolidate info for parser
            parser_info = self._prepare_parser_info(kwargs, parser_class)
            if parser_info:
                kwargs["info"] = parser_info
            elif "info" in kwargs and not parser_info:
                del kwargs["info"]
            
            # Try parsing
            parser_instance = parser_class(**kwargs)
            parse_status = parser_instance.parse()
            
            return self._handle_parse_result(parser_instance, parse_status, parser_class)
            
        except TypeError as e:
            self._logger.error(
                "TypeError with %s init: %s. Passed kwargs: %s",
                parser_class.__name__, e, list(kwargs.keys()), exc_info=True
            )
            self._update_error_if_needed(f"Init error for {parser_class.__name__}: {e}")
            return False
        except Exception as e:
            self._logger.error(
                "Unexpected exception during %s.parse(): %s",
                parser_class.__name__, e, exc_info=True
            )
            self._update_error_if_needed(f"Runtime error in {parser_class.__name__}: {e}")
            return False

    def _prepare_kwargs_for_logging(self, kwargs: dict) -> list[str]:
        """Prepare kwargs for logging, truncating long values."""
        log_kwargs = []
        temp_kwargs = kwargs.copy()
        
        # Truncate long raw data for logging
        if "raw" in temp_kwargs and isinstance(temp_kwargs["raw"], str):
            if len(temp_kwargs["raw"]) > 70:
                temp_kwargs["raw"] = temp_kwargs["raw"][:67] + "..."
        
        for key, value in temp_kwargs.items():
            if key == "logger_obj":
                continue
            elif key == "info" and isinstance(value, dict):
                log_kwargs.append("info=<dict>")
            else:
                formatted_value = f"'{value}'" if isinstance(value, str) else str(value)
                log_kwargs.append(f"{key}={formatted_value}")
        
        return log_kwargs

    def _add_standard_kwargs(self, kwargs: dict) -> None:
        """Add standard kwargs if not already provided."""
        if "width" not in kwargs and self._width > 0:
            kwargs["width"] = self._width
        if "height" not in kwargs and self._height > 0:
            kwargs["height"] = self._height
        kwargs["logger_obj"] = self._logger

    def _prepare_parser_info(self, kwargs: dict, parser_class: type) -> dict:
        """Prepare consolidated info dict for parser."""
        parser_info = kwargs.get("info", {}).copy()
        if not isinstance(parser_info, dict):
            parser_info = {}
        
        # Add software tag if available
        if self._exif_software_tag:
            parser_info["software_tag"] = self._exif_software_tag
        
        # Add IPTC info for MochiDiffusionFormat
        if parser_class is MochiDiffusionFormat and self._parsed_iptc_info:
            parser_info.update(self._parsed_iptc_info)
        
        return parser_info

    def _handle_parse_result(self, parser_instance: BaseFormat, 
                           parse_status: BaseFormat.Status, 
                           parser_class: type) -> bool:
        """Handle the result of a parse attempt."""
        if parse_status == BaseFormat.Status.READ_SUCCESS:
            self._parser = parser_instance
            self._tool = getattr(parser_instance, "tool", parser_class.__name__)
            self._status = BaseFormat.Status.READ_SUCCESS
            self._error = ""
            self._logger.info("Successfully parsed as %s", self._tool)
            return True
        
        # Handle non-success status
        if parse_status != BaseFormat.Status.FORMAT_DETECTION_ERROR:
            parser_error = getattr(parser_instance, "error", "Unknown parser error")
            self._update_error_if_needed(parser_error)
        
        status_name = getattr(parse_status, "name", str(parse_status))
        self._logger.debug(
            "%s parsing attempt: Status %s. Error: %s",
            parser_class.__name__,
            status_name,
            getattr(parser_instance, "error", "N/A")
        )
        return False

    def _update_error_if_needed(self, error_msg: str) -> None:
        """Update error message if current status allows it."""
        if error_msg and (self._status == BaseFormat.Status.UNREAD or not self._error):
            self._error = error_msg

    def _set_error_state(self, error_msg: str, 
                        status: BaseFormat.Status = BaseFormat.Status.FORMAT_ERROR) -> None:
        """Set error state with message and status."""
        self._status = status
        self._error = error_msg
        if status == BaseFormat.Status.FORMAT_ERROR:
            self._logger.error(error_msg)

    def _finalize_reading(self, file_display_name: str) -> None:
        """Finalize the reading process and log results."""
        if self._status == BaseFormat.Status.UNREAD:
            self._logger.warning(
                "No suitable parser for '%s' or all parsers failed/declined",
                file_display_name
            )
            self._set_error_state(
                "No suitable metadata parser or file unreadable/corrupted",
                BaseFormat.Status.FORMAT_ERROR
            )
        
        # Log final status
        tool_name = self._tool or "None"
        status_name = getattr(self._status, "name", str(self._status))
        self._logger.info(
            "Final Reading Status for '%s': %s, Tool: %s",
            file_display_name, status_name, tool_name
        )
        
        # Log error details if present
        final_error = self._get_final_error_message()
        if self._status != BaseFormat.Status.READ_SUCCESS and final_error:
            self._logger.warning("Error details for '%s': %s", file_display_name, final_error)

    def _get_final_error_message(self) -> str:
        """Get the most relevant error message for logging."""
        if self._parser and hasattr(self._parser, "error") and self._parser.error:
            if self._status != BaseFormat.Status.READ_SUCCESS or not self._error:
                return self._parser.error
        return self._error

    def _get_display_name(self, file_path_or_obj: str | Path | TextIO | BinaryIO) -> str:
        """Get a display-friendly name for the file."""
        if hasattr(file_path_or_obj, "name") and file_path_or_obj.name:
            try:
                return Path(file_path_or_obj.name).name
            except TypeError:
                return str(file_path_or_obj.name)
        
        if isinstance(file_path_or_obj, (str, Path)):
            return Path(file_path_or_obj).name
        
        return "UnnamedFileObject"

    # ============================================================================
    # PROPERTIES - Clean, consistent access to parsed data
    # ============================================================================

    @property
    def height(self) -> str:
        """Get image height as string."""
        if self._parser:
            parser_height = str(getattr(self._parser, "height", "0"))
            if parser_height.isdigit() and int(parser_height) > 0:
                return parser_height
        return str(self._width) if self._width > 0 else "0"

    @property
    def width(self) -> str:
        """Get image width as string."""
        if self._parser:
            parser_width = str(getattr(self._parser, "width", "0"))
            if parser_width.isdigit() and int(parser_width) > 0:
                return parser_width
        return str(self._width) if self._width > 0 else "0"

    @property
    def info(self) -> dict[str, Any]:
        """Get a copy of the raw PIL image info."""
        return self._info.copy()

    @property
    def positive(self) -> str:
        """Get positive prompt text."""
        if self._parser:
            return str(getattr(self._parser, "positive", ""))
        
        if self._positive_sdxl:
            return self._positive_sdxl.get("positive", "")
        return self._positive

    @property
    def negative(self) -> str:
        """Get negative prompt text."""
        if self._parser:
            return str(getattr(self._parser, "negative", ""))
        
        if self._negative_sdxl:
            return self._negative_sdxl.get("negative", "")
        return self._negative

    @property
    def positive_sdxl(self) -> dict[str, Any]:
        """Get SDXL-specific positive prompt data."""
        if self._parser:
            return getattr(self._parser, "positive_sdxl", {})
        return self._positive_sdxl.copy()

    @property
    def negative_sdxl(self) -> dict[str, Any]:
        """Get SDXL-specific negative prompt data."""
        if self._parser:
            return getattr(self._parser, "negative_sdxl", {})
        return self._negative_sdxl.copy()

    @property
    def setting(self) -> str:
        """Get generation settings."""
        if self._parser:
            return str(getattr(self._parser, "setting", ""))
        return self._setting

    @property
    def raw(self) -> str:
        """Get raw metadata text."""
        if self._parser:
            parser_raw = getattr(self._parser, "raw", None)
            if parser_raw is not None:
                return str(parser_raw)
        return self._raw

    @property
    def tool(self) -> str:
        """Get the name of the tool that generated the image."""
        if self._parser:
            parser_tool = getattr(self._parser, "tool", None)
            if parser_tool and parser_tool != "Unknown":
                return str(parser_tool)
        return self._tool

    @property
    def parameter(self) -> dict[str, Any]:
        """Get generation parameters dictionary."""
        if self._parser:
            parser_params = getattr(self._parser, "parameter", None)
            if parser_params is not None:
                return parser_params.copy()
        return self._parameter.copy()

    @property
    def format(self) -> str:
        """Get image format (PNG, JPEG, etc.)."""
        return self._format_str

    @property
    def is_sdxl(self) -> bool:
        """Check if image was generated with SDXL."""
        if self._parser:
            return getattr(self._parser, "is_sdxl", False)
        return self._is_sdxl

    @property
    def status(self) -> BaseFormat.Status:
        """Get current parsing status."""
        return self._status

    @property
    def error(self) -> str:
        """Get error message if parsing failed."""
        if self._status != BaseFormat.Status.READ_SUCCESS and self._error:
            return self._error
        
        if self._parser:
            parser_error = getattr(self._parser, "error", None)
            if parser_error:
                return str(parser_error)
        
        return self._error

    @property
    def props(self) -> str:
        """Get all properties as a JSON string."""
        if self._parser and hasattr(self._parser, "props"):
            try:
                return self._parser.props
            except Exception as e:
                self._logger.warning("Error calling parser's props: %s", e)
        
        # Fallback properties
        props_dict = {
            "positive": self.positive,
            "negative": self.negative,
            "width": self.width,
            "height": self.height,
            "tool": self.tool,
            "setting": self.setting,
            "is_sdxl": self.is_sdxl,
            "status": getattr(self.status, "name", str(self.status)),
            "error": self.error,
            "format": self.format,
        }
        
        # Add SDXL data if present
        if self.positive_sdxl:
            props_dict["positive_sdxl"] = self.positive_sdxl
        if self.negative_sdxl:
            props_dict["negative_sdxl"] = self.negative_sdxl
        
        # Add parameters
        props_dict.update(self.parameter)
        
        try:
            return json.dumps(props_dict, indent=2)
        except TypeError as e:
            self._logger.error("Error serializing props to JSON: %s. Data: %s", e, props_dict)
            return json.dumps({"error": f"Failed to serialize props to JSON: {e}"})

    # ============================================================================
    # UTILITY METHODS - For backwards compatibility and convenience
    # ============================================================================

    def remove_data(self) -> None:
        """Remove metadata from image (placeholder for future implementation)."""
        # This would be implemented based on specific requirements
        # for metadata removal functionality
        pass

    def save_image(self, file_path: str) -> None:
        """Save image without metadata (placeholder for future implementation)."""
        # This would be implemented based on specific requirements
        # for saving cleaned images
        pass

    def construct_data(self) -> str:
        """Construct formatted data string from parsed information."""
        if not self._parser:
            return ""
        
        # This could be enhanced based on specific formatting requirements
        return self.raw

    def prompt_to_line(self, prompt: str) -> str:
        """Convert multiline prompt to single line."""
        return prompt.replace('\n', ' ').replace('\r', ' ').strip()