# dataset_tools/main.py

# Copyright (c) 2025 [KTISEOS NYX / 0FTH3N1GHT / EARTH & DUSK MEDIA]
# SPDX-License-Identifier: GPL-3.0

"""
Main application entry point for Dataset Tools.

This module handles command-line argument parsing, logging configuration,
and PyQt application initialization. It serves as the primary launcher
for the Dataset Tools metadata viewer and editor.
"""

import argparse
import sys
from typing import Optional, Sequence

from PyQt6 import QtWidgets

from dataset_tools import __version__, set_package_log_level
from dataset_tools import logger as app_logger
from dataset_tools.ui import MainWindow

# ============================================================================
# CONFIGURATION
# ============================================================================

# Log level mappings for CLI convenience
LOG_LEVEL_SHORTCUTS = {
    "d": "DEBUG",
    "i": "INFO", 
    "w": "WARNING",
    "e": "ERROR",
    "c": "CRITICAL",
}

DEFAULT_LOG_LEVEL = "INFO"

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description=f"Dataset Tools v{__version__} - Metadata Viewer and Editor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dataset-tools                    # Launch with default settings
  dataset-tools --log-level debug  # Launch with debug logging
  dataset-tools --log-level d      # Same as above (shortcut)
        """.strip()
    )
    
    # Generate valid log level choices
    valid_choices = _get_valid_log_level_choices()
    
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        type=str,
        choices=valid_choices,
        help=(
            "Set the logging level. "
            f"Valid choices: {', '.join(LOG_LEVEL_SHORTCUTS.values())} "
            f"or shortcuts: {', '.join(LOG_LEVEL_SHORTCUTS.keys())}. "
            "Case-insensitive."
        ),
        metavar="LEVEL",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"Dataset Tools {__version__}"
    )
    
    # Future expansion area for additional CLI options
    # parser.add_argument("--config", help="Path to configuration file")
    # parser.add_argument("--batch", action="store_true", help="Run in batch mode")
    
    return parser


def _get_valid_log_level_choices() -> list[str]:
    """Generate list of valid log level choices for argparse."""
    choices = []
    
    # Add shortcuts (d, i, w, e, c)
    choices.extend(LOG_LEVEL_SHORTCUTS.keys())
    
    # Add full names in both cases (DEBUG, debug, INFO, info, etc.)
    for level in LOG_LEVEL_SHORTCUTS.values():
        choices.extend([level.upper(), level.lower()])
    
    return choices


def normalize_log_level(raw_level: str) -> str:
    """
    Normalize a log level string to standard uppercase format.
    
    Args:
        raw_level: Raw log level from CLI (could be shortcut or full name)
        
    Returns:
        Normalized log level (e.g., "DEBUG", "INFO")
        
    Examples:
        >>> normalize_log_level("d")
        'DEBUG'
        >>> normalize_log_level("info")
        'INFO'
        >>> normalize_log_level("WARNING")
        'WARNING'
    """
    normalized = raw_level.lower()
    
    # Check if it's a shortcut
    if normalized in LOG_LEVEL_SHORTCUTS:
        return LOG_LEVEL_SHORTCUTS[normalized]
    
    # Otherwise, return uppercase version
    return raw_level.upper()


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def configure_application_logging(log_level: str) -> None:
    """
    Configure application-wide logging settings.
    
    Args:
        log_level: Desired log level (e.g., "DEBUG", "INFO")
    """
    # Update the package-level log level setting
    set_package_log_level(log_level)
    
    # Reconfigure existing logger instances
    if hasattr(app_logger, "reconfigure_all_loggers"):
        app_logger.reconfigure_all_loggers(log_level)
        app_logger.debug_message(f"Logger reconfiguration completed for level: {log_level}")
    else:
        # Fallback warning if the expected function doesn't exist
        print(
            f"WARNING: Logger module missing 'reconfigure_all_loggers' function. "
            f"Log level '{log_level}' may not be fully effective."
        )


def log_application_startup(log_level: str, args: argparse.Namespace) -> None:
    """
    Log application startup information.
    
    Args:
        log_level: Active log level
        args: Parsed command-line arguments
    """
    app_logger.info_monitor(f"Dataset Tools v{__version__} launching...")
    app_logger.info_monitor(f"Application log level set to: {log_level}")
    app_logger.debug_message(f"Parsed CLI arguments: {vars(args)}")


# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

def initialize_qt_application() -> QtWidgets.QApplication:
    """
    Initialize the PyQt application instance.
    
    Returns:
        Configured QApplication instance
    """
    # Pass sys.argv to QApplication to allow Qt to process its own arguments
    # (like -style, -platform, etc.)
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("Dataset Tools")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("KTISEOS NYX")
    
    return app


def create_main_window(args: argparse.Namespace) -> MainWindow:
    """
    Create and configure the main application window.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Configured MainWindow instance
    """
    # Future: Pass args to MainWindow if it needs CLI configuration
    window = MainWindow()
    
    app_logger.debug_message("Main window created successfully")
    return window


def run_application(app: QtWidgets.QApplication, window: MainWindow) -> int:
    """
    Start the application event loop.
    
    Args:
        app: QApplication instance
        window: Main window instance
        
    Returns:
        Application exit code
    """
    window.show()
    app_logger.info_monitor("Application window displayed, entering event loop")
    
    return app.exec()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main(cli_args: Optional[Sequence[str]] = None) -> None:
    """
    Main application entry point.
    
    Handles the complete application lifecycle from argument parsing
    through PyQt event loop execution.
    
    Args:
        cli_args: Optional CLI arguments for testing (uses sys.argv if None)
    """
    try:
        # Parse command-line arguments
        parser = create_argument_parser()
        args = parser.parse_args(args=cli_args)
        
        # Configure logging based on CLI arguments
        log_level = normalize_log_level(args.log_level)
        configure_application_logging(log_level)
        
        # Log startup information
        log_application_startup(log_level, args)
        
        # Initialize PyQt application
        qt_app = initialize_qt_application()
        
        # Create main window
        main_window = create_main_window(args)
        
        # Run the application
        exit_code = run_application(qt_app, main_window)
        
        # Clean exit
        app_logger.info_monitor(f"Application exiting with code: {exit_code}")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        app_logger.info_monitor("Application interrupted by user (Ctrl+C)")
        sys.exit(130)  # Standard exit code for SIGINT
        
    except Exception as e:
        # Log unexpected errors before exiting
        app_logger.error_message(f"Unexpected error during startup: {e}", exc_info=True)
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Entry point when running as a script:
    # - python dataset_tools/main.py
    # - python -m dataset_tools.main
    # - dataset-tools (console script)
    main()