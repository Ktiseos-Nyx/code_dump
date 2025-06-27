# dataset_tools/model_tool.py

"""
Model metadata extraction tool.

This module provides a unified interface for extracting metadata from various
AI model file formats including Safetensors and GGUF files. It uses a parser
registry pattern to handle different file types appropriately.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Type

from .correct_types import EmptyField
from .logger import info_monitor as nfo
from .model_parsers import GGUFParser, ModelParserStatus, SafetensorsParser
from .model_parsers.base import BaseModelParser


class ModelTool:
    """
    Unified tool for extracting metadata from AI model files.
    
    This class acts as a coordinator, selecting the appropriate parser
    based on file extension and handling the parsing workflow with
    proper error handling and logging.
    """
    
    def __init__(self) -> None:
        """Initialize the ModelTool with supported parser mappings."""
        self._parser_registry = self._build_parser_registry()
    
    def _build_parser_registry(self) -> Dict[str, Type[BaseModelParser]]:
        """
        Build the registry mapping file extensions to parser classes.
        
        Returns:
            Dictionary mapping lowercase extensions to parser classes
        """
        return {
            ".safetensors": SafetensorsParser,
            ".sft": SafetensorsParser,  # Alternative extension
            ".gguf": GGUFParser,
        }
    
    def get_supported_extensions(self) -> list[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported extensions (including the dot)
        """
        return list(self._parser_registry.keys())
    
    def is_supported_file(self, file_path: str | Path) -> bool:
        """
        Check if a file type is supported for metadata extraction.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file extension is supported, False otherwise
        """
        extension = Path(file_path).suffix.lower()
        return extension in self._parser_registry
    
    def read_metadata_from(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Extract metadata from a model file.
        
        This is the main entry point for metadata extraction. It handles
        file validation, parser selection, and error reporting.
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Dictionary containing extracted metadata or error information
            
        Examples:
            >>> tool = ModelTool()
            >>> metadata = tool.read_metadata_from("model.safetensors")
            >>> print(metadata.keys())
        """
        # Normalize path handling
        file_path_obj = Path(file_path)
        file_path_str = str(file_path_obj)
        file_name = file_path_obj.name
        extension = file_path_obj.suffix.lower()
        
        nfo("[ModelTool] Attempting to read metadata from: %s", file_name)
        
        # Validate file existence
        if not file_path_obj.exists():
            return self._create_error_response(
                f"File not found: {file_name}",
                file_name,
                "FILE_NOT_FOUND"
            )
        
        # Check if extension is supported
        parser_class = self._parser_registry.get(extension)
        if not parser_class:
            return self._handle_unsupported_extension(extension, file_name)
        
        # Attempt parsing
        return self._parse_with_class(parser_class, file_path_str, file_name, extension)
    
    def _parse_with_class(self, 
                         parser_class: Type[BaseModelParser], 
                         file_path: str,
                         file_name: str, 
                         extension: str) -> Dict[str, Any]:
        """
        Parse a file using the specified parser class.
        
        Args:
            parser_class: Parser class to use
            file_path: Full path to the file
            file_name: Name of the file (for logging)
            extension: File extension
            
        Returns:
            Parsed metadata or error information
        """
        nfo(
            "[ModelTool] Using parser: %s for extension: %s",
            parser_class.__name__,
            extension
        )
        
        try:
            parser_instance = parser_class(file_path)
            parse_status = parser_instance.parse()
            
            return self._handle_parse_result(parser_instance, parse_status, file_name)
            
        except Exception as e:
            nfo(
                "[ModelTool] Unexpected error during parsing with %s: %s",
                parser_class.__name__,
                str(e)
            )
            return self._create_error_response(
                f"Unexpected parsing error: {e}",
                file_name,
                "PARSER_EXCEPTION"
            )
    
    def _handle_parse_result(self, 
                           parser_instance: BaseModelParser,
                           status: ModelParserStatus, 
                           file_name: str) -> Dict[str, Any]:
        """
        Handle the result of a parsing attempt.
        
        Args:
            parser_instance: The parser that was used
            status: Result status from parsing
            file_name: Name of the file (for logging/errors)
            
        Returns:
            Processed result data or error information
        """
        tool_name = getattr(parser_instance, 'tool_name', parser_instance.__class__.__name__)
        
        if status == ModelParserStatus.SUCCESS:
            nfo("[ModelTool] Successfully parsed with %s", tool_name)
            return parser_instance.get_ui_data()
        
        elif status == ModelParserStatus.FAILURE:
            error_msg = self._get_parser_error_message(parser_instance)
            nfo("[ModelTool] Parser %s failed: %s", tool_name, error_msg)
            return self._create_error_response(
                f"{tool_name} parsing failed: {error_msg}",
                file_name,
                "PARSER_FAILURE"
            )
        
        elif status == ModelParserStatus.NOT_APPLICABLE:
            nfo(
                "[ModelTool] Parser %s found file not applicable: %s",
                tool_name,
                file_name
            )
            return self._create_error_response(
                f"File not applicable for {tool_name}",
                file_name,
                "NOT_APPLICABLE"
            )
        
        else:
            # Handle UNATTEMPTED or unexpected status
            status_name = getattr(status, 'name', str(status))
            nfo(
                "[ModelTool] Parser %s returned unexpected status '%s' for %s",
                tool_name,
                status_name,
                file_name
            )
            return self._create_error_response(
                f"Unexpected parser status: {status_name}",
                file_name,
                "UNEXPECTED_STATUS"
            )
    
    def _handle_unsupported_extension(self, extension: str, file_name: str) -> Dict[str, Any]:
        """
        Handle files with unsupported extensions.
        
        Args:
            extension: The unsupported file extension
            file_name: Name of the file
            
        Returns:
            Error response dictionary
        """
        supported_extensions = ", ".join(self.get_supported_extensions())
        
        nfo(
            "[ModelTool] Unsupported model file extension '%s' for file: %s. "
            "Supported extensions: %s",
            extension,
            file_name,
            supported_extensions
        )
        
        return self._create_error_response(
            f"Unsupported model file extension: {extension}. "
            f"Supported: {supported_extensions}",
            file_name,
            "UNSUPPORTED_EXTENSION"
        )
    
    def _get_parser_error_message(self, parser_instance: BaseModelParser) -> str:
        """
        Extract error message from parser instance.
        
        Args:
            parser_instance: Parser that encountered an error
            
        Returns:
            Error message string
        """
        # Try different common error attribute names
        for attr_name in ['_error_message', 'error_message', 'error', '_error']:
            if hasattr(parser_instance, attr_name):
                error_msg = getattr(parser_instance, attr_name)
                if error_msg:
                    return str(error_msg)
        
        return "Unknown parsing error"
    
    def _create_error_response(self, 
                              error_message: str, 
                              file_name: str,
                              error_type: str) -> Dict[str, Any]:
        """
        Create a standardized error response dictionary.
        
        Args:
            error_message: Description of the error
            file_name: Name of the file that caused the error
            error_type: Type/category of the error
            
        Returns:
            Standardized error response dictionary
        """
        return {
            EmptyField.PLACEHOLDER.value: {
                "Error": error_message,
                "File": file_name,
                "ErrorType": error_type,
                "Tool": "ModelTool"
            }
        }
    
    def get_parser_for_file(self, file_path: str | Path) -> Optional[Type[BaseModelParser]]:
        """
        Get the appropriate parser class for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Parser class if supported, None otherwise
        """
        extension = Path(file_path).suffix.lower()
        return self._parser_registry.get(extension)
    
    def register_parser(self, extension: str, parser_class: Type[BaseModelParser]) -> None:
        """
        Register a new parser for a file extension.
        
        Args:
            extension: File extension (with or without leading dot)
            parser_class: Parser class to register
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        extension = extension.lower()
        self._parser_registry[extension] = parser_class
        
        nfo("[ModelTool] Registered parser %s for extension %s", 
            parser_class.__name__, extension)
    
    def unregister_parser(self, extension: str) -> bool:
        """
        Unregister a parser for a file extension.
        
        Args:
            extension: File extension to unregister
            
        Returns:
            True if parser was removed, False if it wasn't registered
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        extension = extension.lower()
        
        if extension in self._parser_registry:
            del self._parser_registry[extension]
            nfo("[ModelTool] Unregistered parser for extension %s", extension)
            return True
        
        return False
    
    def __repr__(self) -> str:
        """String representation of ModelTool."""
        extensions = ", ".join(self.get_supported_extensions())
        return f"ModelTool(supported_extensions=[{extensions}])"