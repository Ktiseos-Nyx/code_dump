# dataset_tools/ui_layout.py

"""
UI layout setup and configuration for the Dataset Tools main window.

This module handles the creation and arrangement of all visual components
in the main application window, including splitters, panels, and action buttons.
"""

import logging
from typing import Any, Optional

from PyQt6 import QtCore
from PyQt6 import QtWidgets as Qw

from .widgets import ImageLabel, LeftPanelWidget

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Default window dimensions
DEFAULT_WINDOW_WIDTH = 1024
DEFAULT_WINDOW_HEIGHT = 768

# Layout spacing and margins
MAIN_LAYOUT_MARGIN = 10
MAIN_LAYOUT_SPACING = 5
METADATA_PANEL_MARGIN = (10, 20, 10, 20)  # left, top, right, bottom
METADATA_PANEL_SPACING = 15
BOTTOM_BAR_MARGIN = (10, 5, 10, 5)
ACTION_BUTTON_SPACING = 10
SPLITTER_MARGIN = 15

# Splitter default proportions
MAIN_SPLITTER_RATIO = (1, 3)  # left panel : right area
METADATA_IMAGE_RATIO = (1, 2)  # metadata : image

# Settings keys for persistence
SETTINGS_MAIN_SPLITTER = "mainSplitterSizes"
SETTINGS_META_IMAGE_SPLITTER = "metaImageSplitterSizes"

# Logger
log = logging.getLogger(__name__)


# ============================================================================
# LAYOUT CONFIGURATION CLASSES
# ============================================================================

class LayoutConfig:
    """Configuration container for layout parameters."""
    
    def __init__(self,
                 window_width: int = DEFAULT_WINDOW_WIDTH,
                 main_margin: int = MAIN_LAYOUT_MARGIN,
                 main_spacing: int = MAIN_LAYOUT_SPACING,
                 metadata_margin: tuple[int, int, int, int] = METADATA_PANEL_MARGIN,
                 metadata_spacing: int = METADATA_PANEL_SPACING):
        self.window_width = window_width
        self.main_margin = main_margin
        self.main_spacing = main_spacing
        self.metadata_margin = metadata_margin
        self.metadata_spacing = metadata_spacing


class SplitterSizes:
    """Helper class for calculating and managing splitter sizes."""
    
    def __init__(self, window_width: int = DEFAULT_WINDOW_WIDTH):
        self.window_width = window_width
    
    def get_main_splitter_default(self) -> list[int]:
        """Get default sizes for main splitter (left panel | right area)."""
        left_width = self.window_width // sum(MAIN_SPLITTER_RATIO) * MAIN_SPLITTER_RATIO[0]
        right_width = self.window_width - left_width
        return [left_width, right_width]
    
    def get_metadata_image_default(self) -> list[int]:
        """Get default sizes for metadata-image splitter."""
        metadata_width = self.window_width // sum(METADATA_IMAGE_RATIO) * METADATA_IMAGE_RATIO[0]
        image_width = self.window_width - metadata_width
        return [metadata_width, image_width]


# ============================================================================
# MAIN LAYOUT SETUP FUNCTION
# ============================================================================

def setup_ui_layout(main_window: Qw.QMainWindow, 
                   config: Optional[LayoutConfig] = None) -> None:
    """
    Create and arrange all visual widgets inside the MainWindow.
    
    This function sets up the complete UI layout including splitters,
    panels, and action buttons. It also connects signals and restores
    saved splitter positions.
    
    Args:
        main_window: The main window to set up
        config: Optional layout configuration (uses defaults if None)
    """
    if config is None:
        config = LayoutConfig()
    
    log.info("Setting up main UI layout...")
    
    try:
        # Setup main container
        _setup_main_container(main_window, config)
        
        # Setup main splitter and panels
        _setup_main_splitter(main_window)
        _setup_left_panel(main_window)
        _setup_middle_right_area(main_window, config)
        
        # Setup bottom action bar
        _setup_bottom_bar(main_window)
        
        # Restore splitter positions
        _restore_splitter_positions(main_window)
        
        log.info("UI layout setup completed successfully")
        
    except Exception as e:
        log.error("Error setting up UI layout: %s", e, exc_info=True)
        raise


# ============================================================================
# LAYOUT COMPONENT SETUP FUNCTIONS
# ============================================================================

def _setup_main_container(main_window: Qw.QMainWindow, config: LayoutConfig) -> None:
    """Setup the main container widget and layout."""
    main_widget = Qw.QWidget()
    main_window.setCentralWidget(main_widget)
    
    overall_layout = Qw.QVBoxLayout(main_widget)
    overall_layout.setContentsMargins(config.main_margin, config.main_margin, 
                                    config.main_margin, config.main_margin)
    overall_layout.setSpacing(config.main_spacing)
    
    # Store reference for later use
    main_window._overall_layout = overall_layout


def _setup_main_splitter(main_window: Qw.QMainWindow) -> None:
    """Setup the main horizontal splitter."""
    main_window.main_splitter = Qw.QSplitter(QtCore.Qt.Orientation.Horizontal)
    main_window._overall_layout.addWidget(main_window.main_splitter, 1)


def _setup_left_panel(main_window: Qw.QMainWindow) -> None:
    """Setup the left panel with file browser."""
    log.debug("Setting up left panel...")
    
    main_window.left_panel = LeftPanelWidget()
    
    # Connect signals from left panel to main window methods
    _connect_left_panel_signals(main_window)
    
    main_window.main_splitter.addWidget(main_window.left_panel)


def _connect_left_panel_signals(main_window: Qw.QMainWindow) -> None:
    """Connect left panel signals to main window slots."""
    connections = [
        (main_window.left_panel.open_folder_requested, main_window.open_folder),
        (main_window.left_panel.sort_files_requested, main_window.sort_files_list),
        (main_window.left_panel.list_item_selected, main_window.on_file_selected),
    ]
    
    for signal, slot in connections:
        try:
            signal.connect(slot)
        except Exception as e:
            log.warning("Failed to connect signal %s to slot %s: %s", 
                       signal, slot, e)


def _setup_middle_right_area(main_window: Qw.QMainWindow, config: LayoutConfig) -> None:
    """Setup the middle-right area containing metadata and image panels."""
    log.debug("Setting up middle-right area...")
    
    # Create container for metadata and image
    middle_right_widget = Qw.QWidget()
    middle_right_layout = Qw.QHBoxLayout(middle_right_widget)
    middle_right_layout.setContentsMargins(0, 0, 0, 0)
    middle_right_layout.setSpacing(config.main_spacing)
    
    # Create metadata-image splitter
    main_window.metadata_image_splitter = Qw.QSplitter(QtCore.Qt.Orientation.Horizontal)
    main_window.metadata_image_splitter.setContentsMargins(
        SPLITTER_MARGIN, SPLITTER_MARGIN, SPLITTER_MARGIN, SPLITTER_MARGIN
    )
    middle_right_layout.addWidget(main_window.metadata_image_splitter)
    
    # Setup metadata and image panels
    _setup_metadata_panel(main_window, config)
    _setup_image_panel(main_window)
    
    # Add to main splitter
    main_window.main_splitter.addWidget(middle_right_widget)


def _setup_metadata_panel(main_window: Qw.QMainWindow, config: LayoutConfig) -> None:
    """Setup the metadata panel with text boxes."""
    log.debug("Setting up metadata panel...")
    
    metadata_widget = Qw.QWidget()
    metadata_layout = Qw.QVBoxLayout(metadata_widget)
    metadata_layout.setContentsMargins(*config.metadata_margin)
    metadata_layout.setSpacing(config.metadata_spacing)
    
    # Add stretch at the top for better spacing
    metadata_layout.addStretch(1)
    
    # Create metadata text boxes
    _create_metadata_text_boxes(main_window, metadata_layout)
    
    # Add stretch at the bottom
    metadata_layout.addStretch(1)
    
    main_window.metadata_image_splitter.addWidget(metadata_widget)


def _create_metadata_text_boxes(main_window: Qw.QMainWindow, 
                                layout: Qw.QVBoxLayout) -> None:
    """Create and configure the metadata text boxes."""
    
    # Configuration for text boxes
    text_box_configs = [
        ("positive_prompt", "Positive Prompt"),
        ("negative_prompt", "Negative Prompt"),
        ("generation_data", "Generation Details & Metadata"),
    ]
    
    for box_name, label_text in text_box_configs:
        _create_text_box_pair(main_window, layout, box_name, label_text)


def _create_text_box_pair(main_window: Qw.QMainWindow, 
                         layout: Qw.QVBoxLayout,
                         box_name: str, 
                         label_text: str) -> None:
    """Create a label-textbox pair and add to layout."""
    
    # Create label
    label = Qw.QLabel(label_text)
    label_attr_name = f"{box_name}_label"
    setattr(main_window, label_attr_name, label)
    layout.addWidget(label)
    
    # Create text box
    text_box = Qw.QTextEdit()
    text_box.setReadOnly(True)
    text_box.setSizePolicy(
        Qw.QSizePolicy.Policy.Expanding, 
        Qw.QSizePolicy.Policy.Preferred
    )
    
    box_attr_name = f"{box_name}_box"
    setattr(main_window, box_attr_name, text_box)
    layout.addWidget(text_box)


def _setup_image_panel(main_window: Qw.QMainWindow) -> None:
    """Setup the image preview panel."""
    log.debug("Setting up image panel...")
    
    main_window.image_preview = ImageLabel()
    main_window.metadata_image_splitter.addWidget(main_window.image_preview)


def _setup_bottom_bar(main_window: Qw.QMainWindow) -> None:
    """Setup the bottom action button bar."""
    log.debug("Setting up bottom action bar...")
    
    bottom_bar = Qw.QWidget()
    bottom_layout = Qw.QHBoxLayout(bottom_bar)
    bottom_layout.setContentsMargins(*BOTTOM_BAR_MARGIN)
    
    # Add left stretch
    bottom_layout.addStretch(1)
    
    # Create action buttons
    action_layout = _create_action_buttons(main_window)
    bottom_layout.addLayout(action_layout)
    
    # Add right stretch
    bottom_layout.addStretch(1)
    
    # Add to overall layout
    main_window._overall_layout.addWidget(bottom_bar, 0)


def _create_action_buttons(main_window: Qw.QMainWindow) -> Qw.QHBoxLayout:
    """Create and configure action buttons."""
    
    action_layout = Qw.QHBoxLayout()
    action_layout.setSpacing(ACTION_BUTTON_SPACING)
    
    # Button configurations: (attribute_name, text, slot_method)
    button_configs = [
        ("copy_metadata_button", "Copy All Metadata", "copy_metadata_to_clipboard"),
        ("settings_button", "Settings", "open_settings_dialog"),
        ("exit_button", "Exit Application", "close"),
    ]
    
    for attr_name, text, slot_name in button_configs:
        button = _create_action_button(main_window, text, slot_name)
        setattr(main_window, attr_name, button)
        action_layout.addWidget(button)
    
    return action_layout


def _create_action_button(main_window: Qw.QMainWindow, 
                         text: str, 
                         slot_name: str) -> Qw.QPushButton:
    """Create a single action button with signal connection."""
    
    button = Qw.QPushButton(text)
    
    # Connect to slot if it exists
    if hasattr(main_window, slot_name):
        slot = getattr(main_window, slot_name)
        button.clicked.connect(slot)
    else:
        log.warning("Slot method '%s' not found on main window", slot_name)
    
    return button


def _restore_splitter_positions(main_window: Qw.QMainWindow) -> None:
    """Restore saved splitter positions from settings."""
    log.debug("Restoring splitter positions...")
    
    try:
        # Get current window width or use default
        window_width = _get_window_width(main_window)
        splitter_sizes = SplitterSizes(window_width)
        
        # Restore main splitter
        _restore_main_splitter(main_window, splitter_sizes)
        
        # Restore metadata-image splitter
        _restore_metadata_image_splitter(main_window, splitter_sizes)
        
    except Exception as e:
        log.warning("Error restoring splitter positions: %s", e)


def _get_window_width(main_window: Qw.QMainWindow) -> int:
    """Get the current window width safely."""
    try:
        if main_window.isVisible():
            return main_window.width()
    except RuntimeError:
        pass
    
    return DEFAULT_WINDOW_WIDTH


def _restore_main_splitter(main_window: Qw.QMainWindow, 
                          splitter_sizes: SplitterSizes) -> None:
    """Restore main splitter sizes."""
    if not hasattr(main_window, 'settings'):
        log.warning("Main window has no settings attribute")
        return
    
    default_sizes = splitter_sizes.get_main_splitter_default()
    saved_sizes = main_window.settings.value(
        SETTINGS_MAIN_SPLITTER, 
        default_sizes, 
        type=list
    )
    
    # Convert to integers and set
    int_sizes = [int(s) for s in saved_sizes]
    main_window.main_splitter.setSizes(int_sizes)
    
    log.debug("Main splitter sizes restored: %s", int_sizes)


def _restore_metadata_image_splitter(main_window: Qw.QMainWindow, 
                                   splitter_sizes: SplitterSizes) -> None:
    """Restore metadata-image splitter sizes."""
    if not hasattr(main_window, 'settings'):
        return
    
    default_sizes = splitter_sizes.get_metadata_image_default()
    saved_sizes = main_window.settings.value(
        SETTINGS_META_IMAGE_SPLITTER, 
        default_sizes, 
        type=list
    )
    
    # Convert to integers and set
    int_sizes = [int(s) for s in saved_sizes]
    main_window.metadata_image_splitter.setSizes(int_sizes)
    
    log.debug("Metadata-image splitter sizes restored: %s", int_sizes)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_splitter_positions(main_window: Qw.QMainWindow) -> None:
    """
    Save current splitter positions to settings.
    
    This function should be called when the application is closing
    to preserve the user's layout preferences.
    
    Args:
        main_window: The main window with splitters to save
    """
    if not hasattr(main_window, 'settings'):
        log.warning("Cannot save splitter positions: no settings attribute")
        return
    
    try:
        # Save main splitter sizes
        main_sizes = main_window.main_splitter.sizes()
        main_window.settings.setValue(SETTINGS_MAIN_SPLITTER, main_sizes)
        
        # Save metadata-image splitter sizes
        meta_img_sizes = main_window.metadata_image_splitter.sizes()
        main_window.settings.setValue(SETTINGS_META_IMAGE_SPLITTER, meta_img_sizes)
        
        log.debug("Splitter positions saved: main=%s, meta_img=%s", 
                 main_sizes, meta_img_sizes)
        
    except Exception as e:
        log.error("Error saving splitter positions: %s", e)


def reset_splitter_positions(main_window: Qw.QMainWindow) -> None:
    """
    Reset splitter positions to default values.
    
    Args:
        main_window: The main window with splitters to reset
    """
    try:
        window_width = _get_window_width(main_window)
        splitter_sizes = SplitterSizes(window_width)
        
        # Reset to defaults
        main_window.main_splitter.setSizes(splitter_sizes.get_main_splitter_default())
        main_window.metadata_image_splitter.setSizes(splitter_sizes.get_metadata_image_default())
        
        log.info("Splitter positions reset to defaults")
        
    except Exception as e:
        log.error("Error resetting splitter positions: %s", e)


def get_layout_info(main_window: Qw.QMainWindow) -> dict[str, Any]:
    """
    Get current layout information for debugging.
    
    Args:
        main_window: The main window to inspect
        
    Returns:
        Dictionary containing layout information
    """
    info = {
        "window_size": (main_window.width(), main_window.height()),
        "main_splitter_sizes": main_window.main_splitter.sizes(),
        "metadata_image_splitter_sizes": main_window.metadata_image_splitter.sizes(),
    }
    
    return info