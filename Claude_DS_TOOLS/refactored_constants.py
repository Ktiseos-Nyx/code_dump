# dataset_tools/vendored_sdpr/constants.py

__author__ = "receyuki"
__filename__ = "constants.py"
# MODIFIED by Ktiseos Nyx for Dataset-Tools
__copyright__ = "Copyright 2023, Receyuki"
__email__ = "receyuki@gmail.com"

"""
Constants and configuration values for the vendored SDPR package.

This module contains all static configuration including supported formats,
UI resource paths, messages, tooltips, and styling constants.
"""

from importlib import resources
from pathlib import Path
from typing import Final

from . import resources as res

# ============================================================================
# CORE CONFIGURATION
# ============================================================================

# Resource directory path
RESOURCE_DIR: Final[str] = str(resources.files(res))

# Supported image formats for metadata extraction
SUPPORTED_FORMATS: Final[list[str]] = [".png", ".jpg", ".jpeg", ".webp"]

# Parameter placeholder for missing values
PARAMETER_PLACEHOLDER: Final[str] = "                    "

# ============================================================================
# UI RESOURCE PATHS
# ============================================================================

class ResourcePaths:
    """Centralized UI resource file paths."""
    
    # Theme and base resources
    COLOR_THEME: Final[Path] = Path(RESOURCE_DIR, "gray.json")
    ICON_FILE: Final[Path] = Path(RESOURCE_DIR, "icon.png")
    ICON_CUBE_FILE: Final[Path] = Path(RESOURCE_DIR, "icon-cube.png")
    ICO_FILE: Final[Path] = Path(RESOURCE_DIR, "icon-gui.ico")
    
    # Status icons (single state)
    INFO_FILE: Final[Path] = Path(RESOURCE_DIR, "info_24.png")
    ERROR_FILE: Final[Path] = Path(RESOURCE_DIR, "error_24.png")
    WARNING_FILE: Final[Path] = Path(RESOURCE_DIR, "warning_24.png")
    OK_FILE: Final[Path] = Path(RESOURCE_DIR, "check_circle_24.png")
    UPDATE_FILE: Final[Path] = Path(RESOURCE_DIR, "update_24.png")
    DROP_FILE: Final[Path] = Path(RESOURCE_DIR, "place_item_48.png")
    
    # Interactive icons (normal, alpha states)
    COPY_LARGE: Final[tuple[Path, Path]] = (
        Path(RESOURCE_DIR, "content_copy_24.png"),
        Path(RESOURCE_DIR, "content_copy_24_alpha.png"),
    )
    
    COPY_SMALL: Final[tuple[Path, Path]] = (
        Path(RESOURCE_DIR, "content_copy_20.png"),
        Path(RESOURCE_DIR, "content_copy_20_alpha.png"),
    )
    
    CLEAR_FILE: Final[tuple[Path, Path]] = (
        Path(RESOURCE_DIR, "mop_24.png"),
        Path(RESOURCE_DIR, "mop_24_alpha.png")
    )
    
    DOCUMENT_FILE: Final[tuple[Path, Path]] = (
        Path(RESOURCE_DIR, "description_24.png"),
        Path(RESOURCE_DIR, "description_24_alpha.png"),
    )
    
    EXPAND_FILE: Final[tuple[Path, Path]] = (
        Path(RESOURCE_DIR, "expand_more_24.png"),
        Path(RESOURCE_DIR, "expand_more_24_alpha.png"),
    )
    
    EDIT_FILE: Final[tuple[Path, Path]] = (
        Path(RESOURCE_DIR, "edit_24.png"),
        Path(RESOURCE_DIR, "edit_24_alpha.png")
    )
    
    EDIT_OFF_FILE: Final[tuple[Path, Path]] = (
        Path(RESOURCE_DIR, "edit_off_24.png"),
        Path(RESOURCE_DIR, "edit_off_24_alpha.png"),
    )
    
    LIGHTBULB_FILE: Final[tuple[Path, Path]] = (
        Path(RESOURCE_DIR, "lightbulb_20.png"),
        Path(RESOURCE_DIR, "lightbulb_20_alpha.png"),
    )
    
    SAVE_FILE: Final[tuple[Path, Path]] = (
        Path(RESOURCE_DIR, "save_24.png"),
        Path(RESOURCE_DIR, "save_24_alpha.png")
    )
    
    SORT_FILE: Final[tuple[Path, Path]] = (
        Path(RESOURCE_DIR, "sort_by_alpha_20.png"),
        Path(RESOURCE_DIR, "sort_by_alpha_20_alpha.png"),
    )
    
    VIEW_SEPARATE_FILE: Final[tuple[Path, Path]] = (
        Path(RESOURCE_DIR, "view_week_20.png"),
        Path(RESOURCE_DIR, "view_week_20_alpha.png"),
    )
    
    VIEW_TAB_FILE: Final[tuple[Path, Path]] = (
        Path(RESOURCE_DIR, "view_sidebar_20.png"),
        Path(RESOURCE_DIR, "view_sidebar_20_alpha.png"),
    )


# Backwards compatibility aliases
COLOR_THEME = ResourcePaths.COLOR_THEME
INFO_FILE = ResourcePaths.INFO_FILE
ERROR_FILE = ResourcePaths.ERROR_FILE
WARNING_FILE = ResourcePaths.WARNING_FILE
OK_FILE = ResourcePaths.OK_FILE
UPDATE_FILE = ResourcePaths.UPDATE_FILE
DROP_FILE = ResourcePaths.DROP_FILE
COPY_FILE_L = ResourcePaths.COPY_LARGE
COPY_FILE_S = ResourcePaths.COPY_SMALL
CLEAR_FILE = ResourcePaths.CLEAR_FILE
DOCUMENT_FILE = ResourcePaths.DOCUMENT_FILE
EXPAND_FILE = ResourcePaths.EXPAND_FILE
EDIT_FILE = ResourcePaths.EDIT_FILE
EDIT_OFF_FILE = ResourcePaths.EDIT_OFF_FILE
LIGHTBULB_FILE = ResourcePaths.LIGHTBULB_FILE
SAVE_FILE = ResourcePaths.SAVE_FILE
SORT_FILE = ResourcePaths.SORT_FILE
VIEW_SEPARATE_FILE = ResourcePaths.VIEW_SEPARATE_FILE
VIEW_TAB_FILE = ResourcePaths.VIEW_TAB_FILE
ICON_FILE = ResourcePaths.ICON_FILE
ICON_CUBE_FILE = ResourcePaths.ICON_CUBE_FILE
ICO_FILE = ResourcePaths.ICO_FILE

# ============================================================================
# USER INTERFACE MESSAGES
# ============================================================================

class Messages:
    """User-facing messages for various application states."""
    
    # File operations
    DROP: Final[list[str]] = ["Drop image here or click to select"]
    DEFAULT: Final[list[str]] = ["Drag and drop your image file into the window"]
    SUCCESS: Final[list[str]] = ["Voilà!"]
    
    # Error states
    FORMAT_ERROR: Final[list[str]] = ["", "No data detected or unsupported format"]
    SUFFIX_ERROR: Final[list[str]] = ["Unsupported format"]
    TXT_ERROR: Final[list[str]] = [
        "Importing TXT file is only allowed in edit mode",
        "unsupported TXT format",
    ]
    COMFYUI_ERROR: Final[list[str]] = [
        "The ComfyUI workflow is overly complex, or unsupported custom nodes have been used",
        "Failed to parse ComfyUI data, click here for more info",
    ]
    
    # Success operations
    CLIPBOARD: Final[list[str]] = ["Copied to the clipboard"]
    UPDATE: Final[list[str]] = ["A new version is available, click here to download"]
    EXPORT: Final[list[str]] = ["The TXT file has been generated"]
    ALONGSIDE: Final[list[str]] = ["The TXT file has been generated alongside the image"]
    TXT_SELECT: Final[list[str]] = ["The TXT file has been generated in the selected directory"]
    TXT_IMPORTED: Final[list[str]] = ["The TXT file has been successfully imported"]
    
    # Image operations
    REMOVE: Final[list[str]] = ["A new image file has been generated"]
    SUFFIX: Final[list[str]] = ["A new image file with suffix has been generated"]
    OVERWRITE: Final[list[str]] = ["A new image file has overwritten the original image"]
    REMOVE_SELECT: Final[list[str]] = ["A new image file has been generated in the selected directory"]
    
    # Mode toggles
    EDIT: Final[list[str]] = ["Edit mode", "View mode"]
    SORT: Final[list[str]] = ["Ascending order", "Descending order", "Original order"]
    VIEW_PROMPT: Final[list[str]] = ["Vertical orientation", "Horizontal orientation"]
    VIEW_SETTING: Final[list[str]] = ["Simple mode", "Normal mode"]


# Backwards compatibility
MESSAGE = {
    "drop": Messages.DROP,
    "default": Messages.DEFAULT,
    "success": Messages.SUCCESS,
    "format_error": Messages.FORMAT_ERROR,
    "suffix_error": Messages.SUFFIX_ERROR,
    "clipboard": Messages.CLIPBOARD,
    "update": Messages.UPDATE,
    "export": Messages.EXPORT,
    "alongside": Messages.ALONGSIDE,
    "txt_select": Messages.TXT_SELECT,
    "remove": Messages.REMOVE,
    "suffix": Messages.SUFFIX,
    "overwrite": Messages.OVERWRITE,
    "remove_select": Messages.REMOVE_SELECT,
    "txt_error": Messages.TXT_ERROR,
    "txt_imported": Messages.TXT_IMPORTED,
    "edit": Messages.EDIT,
    "sort": Messages.SORT,
    "view_prompt": Messages.VIEW_PROMPT,
    "view_setting": Messages.VIEW_SETTING,
    "comfyui_error": Messages.COMFYUI_ERROR,
}

# ============================================================================
# TOOLTIPS
# ============================================================================

class Tooltips:
    """Tooltip text for UI elements."""
    
    EDIT: Final[str] = "Edit image metadata"
    SAVE: Final[str] = "Save edited image"
    CLEAR: Final[str] = "Remove metadata from the image"
    EXPORT: Final[str] = "Export metadata to a TXT file"
    COPY_RAW: Final[str] = "Copy raw metadata to the clipboard"
    COPY_PROMPT: Final[str] = "Copy prompt to the clipboard"
    COPY_SETTING: Final[str] = "Copy setting to the clipboard"
    SORT: Final[str] = "Sort prompt lines in ascending or descending order"
    VIEW_PROMPT: Final[str] = "View prompt in vertical orientation"
    VIEW_SETTING: Final[str] = "View setting in simple mode"
    VIEW_SEPARATE: Final[str] = "View Clip G, Clip L and Refiner prompt in separate textbox"
    VIEW_TAB: Final[str] = "View Clip G, Clip L and Refiner prompt in one textbox"


# Backwards compatibility
TOOLTIP = {
    "edit": Tooltips.EDIT,
    "save": Tooltips.SAVE,
    "clear": Tooltips.CLEAR,
    "export": Tooltips.EXPORT,
    "copy_raw": Tooltips.COPY_RAW,
    "copy_prompt": Tooltips.COPY_PROMPT,
    "copy_setting": Tooltips.COPY_SETTING,
    "sort": Tooltips.SORT,
    "view_prompt": Tooltips.VIEW_PROMPT,
    "view_setting": Tooltips.VIEW_SETTING,
    "view_separate": Tooltips.VIEW_SEPARATE,
    "view_tab": Tooltips.VIEW_TAB,
}

# ============================================================================
# EXTERNAL URLS
# ============================================================================

class URLs:
    """External URL endpoints for the application."""
    
    RELEASE: Final[str] = "https://api.github.com/repos/receyuki/stable-diffusion-prompt-reader/releases/latest"
    FORMAT_INFO: Final[str] = "https://github.com/receyuki/stable-diffusion-prompt-reader#supported-formats"
    COMFYUI_INFO: Final[str] = "https://github.com/receyuki/stable-diffusion-prompt-reader#comfyui"


# Backwards compatibility
URL = {
    "release": URLs.RELEASE,
    "format": URLs.FORMAT_INFO,
    "comfyui": URLs.COMFYUI_INFO,
}

# ============================================================================
# COLOR SCHEME
# ============================================================================

class Colors:
    """Application color scheme constants."""
    
    # Base colors
    DEFAULT_GRAY: Final[str] = "#8E8E93"
    
    # Accessibility variants (light, dark)
    ACCESSIBLE_GRAY: Final[tuple[str, str]] = ("#6C6C70", "#AEAEB2")
    INACCESSIBLE_GRAY: Final[tuple[str, str]] = ("gray60", "gray45")
    
    # Interactive states
    EDITABLE: Final[tuple[str, str]] = ("gray10", "#DCE4EE")
    BUTTON_HOVER: Final[tuple[str, str]] = ("gray86", "gray17")


# Backwards compatibility
DEFAULT_GRAY = Colors.DEFAULT_GRAY
ACCESSIBLE_GRAY = Colors.ACCESSIBLE_GRAY
INACCESSIBLE_GRAY = Colors.INACCESSIBLE_GRAY
EDITABLE = Colors.EDITABLE
BUTTON_HOVER = Colors.BUTTON_HOVER

# ============================================================================
# UI DIMENSIONS
# ============================================================================

class Dimensions:
    """UI element size and spacing constants."""
    
    # Timing
    TOOLTIP_DELAY: Final[float] = 1.5
    
    # Button sizes (large)
    BUTTON_WIDTH_L: Final[int] = 40
    BUTTON_HEIGHT_L: Final[int] = 40
    
    # Button sizes (small)
    BUTTON_WIDTH_S: Final[int] = 36
    BUTTON_HEIGHT_S: Final[int] = 36
    
    # Other UI elements
    LABEL_HEIGHT: Final[int] = 20
    ARROW_WIDTH_L: Final[int] = 28
    PARAMETER_WIDTH: Final[int] = 280
    
    # Status bar
    STATUS_BAR_IPAD: Final[int] = 5
    STATUS_BAR_HEIGHT: Final[int] = BUTTON_HEIGHT_L + LABEL_HEIGHT - STATUS_BAR_IPAD * 2


# Backwards compatibility
TOOLTIP_DELAY = Dimensions.TOOLTIP_DELAY
BUTTON_WIDTH_L = Dimensions.BUTTON_WIDTH_L
BUTTON_HEIGHT_L = Dimensions.BUTTON_HEIGHT_L
BUTTON_WIDTH_S = Dimensions.BUTTON_WIDTH_S
BUTTON_HEIGHT_S = Dimensions.BUTTON_HEIGHT_S
LABEL_HEIGHT = Dimensions.LABEL_HEIGHT
ARROW_WIDTH_L = Dimensions.ARROW_WIDTH_L
STATUS_BAR_IPAD = Dimensions.STATUS_BAR_IPAD
PARAMETER_WIDTH = Dimensions.PARAMETER_WIDTH
STATUS_BAR_HEIGHT = Dimensions.STATUS_BAR_HEIGHT

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_resource_path(filename: str) -> Path:
    """
    Get the full path to a resource file.
    
    Args:
        filename: Name of the resource file
        
    Returns:
        Full path to the resource file
    """
    return Path(RESOURCE_DIR, filename)


def is_supported_format(file_extension: str) -> bool:
    """
    Check if a file extension is supported for metadata extraction.
    
    Args:
        file_extension: File extension to check (with or without leading dot)
        
    Returns:
        True if the format is supported, False otherwise
    """
    if not file_extension.startswith('.'):
        file_extension = f'.{file_extension}'
    
    return file_extension.lower() in SUPPORTED_FORMATS


def get_icon_paths(icon_name: str) -> tuple[Path, Path] | Path | None:
    """
    Get icon paths by name, handling both single and dual-state icons.
    
    Args:
        icon_name: Name of the icon (e.g., 'copy_large', 'edit', 'info')
        
    Returns:
        Path or tuple of paths for the icon, None if not found
    """
    icon_mapping = {
        # Single state icons
        'info': ResourcePaths.INFO_FILE,
        'error': ResourcePaths.ERROR_FILE,
        'warning': ResourcePaths.WARNING_FILE,
        'ok': ResourcePaths.OK_FILE,
        'update': ResourcePaths.UPDATE_FILE,
        'drop': ResourcePaths.DROP_FILE,
        'icon': ResourcePaths.ICON_FILE,
        'icon_cube': ResourcePaths.ICON_CUBE_FILE,
        'ico': ResourcePaths.ICO_FILE,
        
        # Dual state icons
        'copy_large': ResourcePaths.COPY_LARGE,
        'copy_small': ResourcePaths.COPY_SMALL,
        'clear': ResourcePaths.CLEAR_FILE,
        'document': ResourcePaths.DOCUMENT_FILE,
        'expand': ResourcePaths.EXPAND_FILE,
        'edit': ResourcePaths.EDIT_FILE,
        'edit_off': ResourcePaths.EDIT_OFF_FILE,
        'lightbulb': ResourcePaths.LIGHTBULB_FILE,
        'save': ResourcePaths.SAVE_FILE,
        'sort': ResourcePaths.SORT_FILE,
        'view_separate': ResourcePaths.VIEW_SEPARATE_FILE,
        'view_tab': ResourcePaths.VIEW_TAB_FILE,
    }
    
    return icon_mapping.get(icon_name)


def get_message(message_key: str, index: int = 0) -> str:
    """
    Get a message by key and index.
    
    Args:
        message_key: Key for the message group
        index: Index within the message list (default: 0)
        
    Returns:
        The requested message, or empty string if not found
    """
    messages = MESSAGE.get(message_key, [])
    if 0 <= index < len(messages):
        return messages[index]
    return ""


def get_tooltip(tooltip_key: str) -> str:
    """
    Get a tooltip by key.
    
    Args:
        tooltip_key: Key for the tooltip
        
    Returns:
        The tooltip text, or empty string if not found
    """
    return TOOLTIP.get(tooltip_key, "")