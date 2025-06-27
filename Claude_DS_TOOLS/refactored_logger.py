# dataset_tools/vendored_sdpr/logger.py

__author__ = "receyuki"
__filename__ = "logger.py"
# MODIFIED by Ktiseos Nyx for Dataset-Tools and clarity
__copyright__ = "Copyright 2024, Receyuki & Ktiseos Nyx"
__email__ = "receyuki@gmail.com"

"""
Logger utilities for the vendored SDPR package.

This module provides a centralized logging system with caching, level management,
and configurable handlers. It's designed to work both standalone and as part of
larger applications that manage their own logging infrastructure.
"""

import logging
from typing import Optional

# Global cache for logger instances to avoid recreation
_logger_cache: dict[str, logging.Logger] = {}

# Standard log level mappings
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Default formatter for basic handlers
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_log_level_value(level_name: Optional[str]) -> int:
    """
    Convert a log level name to its numeric value.
    
    Args:
        level_name: String representation of log level (case-insensitive)
        
    Returns:
        Numeric log level value, defaults to INFO if invalid/None
        
    Examples:
        >>> get_log_level_value("DEBUG")
        10
        >>> get_log_level_value("invalid")
        20
        >>> get_log_level_value(None)
        20
    """
    if not level_name:
        return logging.INFO
    
    normalized_level = level_name.strip().upper()
    return LOG_LEVELS.get(normalized_level, logging.INFO)


def configure_basic_handler(logger: logging.Logger, 
                          formatter: Optional[logging.Formatter] = None) -> None:
    """
    Configure a basic stream handler for a logger.
    
    This adds a console handler with the specified formatter and disables
    propagation to prevent duplicate messages in larger applications.
    
    Args:
        logger: Logger instance to configure
        formatter: Custom formatter, uses default if None
    """
    if logger.handlers:
        # Logger already has handlers, don't add another
        return
    
    if formatter is None:
        formatter = logging.Formatter(DEFAULT_FORMAT)
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


def create_logger(name: str, 
                 level: Optional[str] = None,
                 force_basic_handler: bool = False,
                 formatter: Optional[logging.Formatter] = None) -> logging.Logger:
    """
    Create a new logger instance with specified configuration.
    
    Args:
        name: Logger name (typically module path)
        level: Log level as string
        force_basic_handler: Whether to add a basic console handler
        formatter: Custom formatter for the handler
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set log level if specified
    if level is not None:
        level_value = get_log_level_value(level)
        logger.setLevel(level_value)
    
    # Add basic handler if requested
    if force_basic_handler:
        configure_basic_handler(logger, formatter)
    
    return logger


def update_cached_logger(logger: logging.Logger, 
                        level: Optional[str] = None,
                        force_basic_handler: bool = False,
                        formatter: Optional[logging.Formatter] = None) -> None:
    """
    Update an existing cached logger with new configuration.
    
    Args:
        logger: Existing logger instance
        level: New log level as string
        force_basic_handler: Whether to ensure basic handler exists
        formatter: Custom formatter for any new handler
    """
    # Update log level if specified and different
    if level is not None:
        new_level_value = get_log_level_value(level)
        if logger.level == 0 or logger.level != new_level_value:
            logger.setLevel(new_level_value)
    
    # Add basic handler if forced and none exists
    if force_basic_handler and not logger.handlers:
        configure_basic_handler(logger, formatter)


def get_logger(name: str,
               level: Optional[str] = None,
               force_basic_handler: bool = False,
               formatter: Optional[logging.Formatter] = None) -> logging.Logger:
    """
    Get a logger instance, using cache when possible.
    
    This is the main entry point for obtaining loggers. It maintains a cache
    of logger instances to avoid recreation and provides consistent configuration.
    
    Args:
        name: Logger name, typically following module hierarchy
        level: Log level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        force_basic_handler: If True, ensures logger has a basic console handler
        formatter: Custom formatter for any handlers created
        
    Returns:
        Configured logger instance
        
    Examples:
        >>> logger = get_logger("MyModule", level="DEBUG")
        >>> logger = get_logger("MyModule.SubModule", force_basic_handler=True)
    """
    # Check cache first
    if name in _logger_cache:
        cached_logger = _logger_cache[name]
        update_cached_logger(cached_logger, level, force_basic_handler, formatter)
        return cached_logger
    
    # Create new logger
    logger = create_logger(name, level, force_basic_handler, formatter)
    _logger_cache[name] = logger
    
    return logger


def configure_root_logger(level: str = "INFO", 
                         formatter: Optional[logging.Formatter] = None) -> None:
    """
    Configure the root logger for standalone usage.
    
    WARNING: This affects the global logging configuration and should be used
    with caution in larger applications. Typically only used for testing or
    when this package is used standalone.
    
    Args:
        level: Log level for root logger
        formatter: Custom formatter, uses default if None
    """
    if formatter is None:
        formatter = logging.Formatter(DEFAULT_FORMAT)
    
    root_logger = logging.getLogger()
    level_value = get_log_level_value(level)
    
    # Avoid duplicate handlers
    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler) 
        for handler in root_logger.handlers
    )
    
    if not has_stream_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    
    root_logger.setLevel(level_value)


def clear_logger_cache() -> None:
    """
    Clear the logger cache.
    
    Useful for testing or when you need to reset logger configurations.
    Note: This doesn't remove the actual loggers from Python's logging
    system, just from our cache.
    """
    global _logger_cache
    _logger_cache.clear()


def get_cached_logger_names() -> list[str]:
    """
    Get a list of all cached logger names.
    
    Returns:
        List of logger names currently in cache
    """
    return list(_logger_cache.keys())


def is_logger_configured(name: str) -> bool:
    """
    Check if a logger is already configured (cached).
    
    Args:
        name: Logger name to check
        
    Returns:
        True if logger exists in cache, False otherwise
    """
    return name in _logger_cache


class LoggerConfig:
    """
    Configuration container for logger settings.
    
    This class provides a convenient way to bundle logger configuration
    parameters and apply them consistently across multiple loggers.
    """
    
    def __init__(self, 
                 level: Optional[str] = None,
                 force_basic_handler: bool = False,
                 formatter: Optional[logging.Formatter] = None,
                 format_string: Optional[str] = None):
        """
        Initialize logger configuration.
        
        Args:
            level: Log level as string
            force_basic_handler: Whether to add basic console handler
            formatter: Custom formatter instance
            format_string: Format string for creating formatter (ignored if formatter provided)
        """
        self.level = level
        self.force_basic_handler = force_basic_handler
        
        if formatter is not None:
            self.formatter = formatter
        elif format_string is not None:
            self.formatter = logging.Formatter(format_string)
        else:
            self.formatter = None
    
    def apply_to_logger(self, name: str) -> logging.Logger:
        """
        Apply this configuration to get a logger.
        
        Args:
            name: Logger name
            
        Returns:
            Configured logger instance
        """
        return get_logger(
            name=name,
            level=self.level,
            force_basic_handler=self.force_basic_handler,
            formatter=self.formatter
        )


# Convenience configurations
DEBUG_CONFIG = LoggerConfig(level="DEBUG", force_basic_handler=True)
INFO_CONFIG = LoggerConfig(level="INFO", force_basic_handler=True)
ERROR_CONFIG = LoggerConfig(level="ERROR", force_basic_handler=True)


def get_debug_logger(name: str) -> logging.Logger:
    """Get a logger configured for debug output."""
    return DEBUG_CONFIG.apply_to_logger(name)


def get_info_logger(name: str) -> logging.Logger:
    """Get a logger configured for info output."""
    return INFO_CONFIG.apply_to_logger(name)


def get_error_logger(name: str) -> logging.Logger:
    """Get a logger configured for error output."""
    return ERROR_CONFIG.apply_to_logger(name)


# Module-level logger for this package
_module_logger = get_logger(__name__)


def _demo_logger_functionality():
    """Demonstrate logger functionality for testing/development."""
    print("=== Logger Module Demonstration ===\n")
    
    # Test basic logger creation
    logger1 = get_logger("DSVendored_SDPR.Module1", level="DEBUG", force_basic_handler=True)
    logger2 = get_logger("DSVendored_SDPR.Module2", level="INFO", force_basic_handler=True)
    
    # Test caching
    logger1_cached = get_logger("DSVendored_SDPR.Module1")
    assert logger1 is logger1_cached, "Logger caching failed"
    
    # Test logging at different levels
    logger1.debug("Debug message from Module1")
    logger1.info("Info message from Module1")
    logger2.info("Info message from Module2")
    logger2.warning("Warning message from Module2")
    
    # Test submodule logger
    sub_logger = get_logger("DSVendored_SDPR.Module1.Submodule", force_basic_handler=True)
    sub_logger.error("Error from submodule")
    
    # Test convenience functions
    debug_logger = get_debug_logger("TestDebug")
    debug_logger.debug("Debug via convenience function")
    
    info_logger = get_info_logger("TestInfo")
    info_logger.info("Info via convenience function")
    
    error_logger = get_error_logger("TestError")
    error_logger.error("Error via convenience function")
    
    # Test configuration class
    custom_config = LoggerConfig(
        level="WARNING",
        force_basic_handler=True,
        format_string="[%(levelname)s] %(name)s: %(message)s"
    )
    custom_logger = custom_config.apply_to_logger("CustomConfigured")
    custom_logger.warning("Warning with custom format")
    
    # Show cache status
    print(f"\nCached loggers: {get_cached_logger_names()}")
    print(f"Module1 is configured: {is_logger_configured('DSVendored_SDPR.Module1')}")
    print(f"NonExistent is configured: {is_logger_configured('NonExistent')}")


if __name__ == "__main__":
    # Run demonstration when executed directly
    _demo_logger_functionality()