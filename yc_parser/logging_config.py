"""
Centralized logging configuration for the YC Company Parser.

This module provides structured logging setup with different levels,
formatters, and handlers for comprehensive error tracking and debugging.
"""
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels for console output."""
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """Structured formatter for consistent log formatting."""
    
    def format(self, record):
        # Add structured fields to the record
        record.timestamp = datetime.now().isoformat()
        record.module_name = record.name
        record.function_name = record.funcName
        record.line_number = record.lineno
        
        # Format the message with structured information
        formatted = super().format(record)
        
        # Add exception information if present
        if record.exc_info:
            formatted += f"\nException: {self.formatException(record.exc_info)}"
        
        return formatted


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    structured_format: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up comprehensive logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output logs to console
        structured_format: Whether to use structured formatting
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured root logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Define formatters
    if structured_format:
        detailed_format = (
            "%(timestamp)s | %(levelname)-8s | %(module_name)s.%(function_name)s:%(line_number)d | %(message)s"
        )
        simple_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        
        file_formatter = StructuredFormatter(detailed_format)
        console_formatter = ColoredFormatter(simple_format) if console_output else StructuredFormatter(simple_format)
    else:
        basic_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        file_formatter = logging.Formatter(basic_format)
        console_formatter = ColoredFormatter(basic_format) if console_output else logging.Formatter(basic_format)
    
    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Add error-specific handler for critical issues
    if log_file:
        error_log_file = str(log_path).replace('.log', '_errors.log')
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
    
    # Log the logging configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {log_level}, File: {log_file}, Console: {console_output}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the logger (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, message: str, exc_info=True):
    """
    Log an exception with full traceback and context.
    
    Args:
        logger: Logger instance to use
        message: Descriptive message about the exception
        exc_info: Whether to include exception info (default: True)
    """
    logger.error(message, exc_info=exc_info)


def log_performance(logger: logging.Logger, operation: str, duration: float, 
                   additional_info: Optional[dict] = None):
    """
    Log performance metrics for operations.
    
    Args:
        logger: Logger instance to use
        operation: Name of the operation
        duration: Duration in seconds
        additional_info: Additional context information
    """
    info_str = f"Performance: {operation} completed in {duration:.2f}s"
    
    if additional_info:
        info_parts = [f"{k}={v}" for k, v in additional_info.items()]
        info_str += f" | {', '.join(info_parts)}"
    
    logger.info(info_str)


def log_data_operation(logger: logging.Logger, operation: str, count: int, 
                      success: bool, errors: Optional[list] = None):
    """
    Log data operation results with structured information.
    
    Args:
        logger: Logger instance to use
        operation: Name of the data operation
        count: Number of items processed
        success: Whether the operation was successful
        errors: List of errors encountered (optional)
    """
    status = "SUCCESS" if success else "FAILED"
    message = f"Data Operation: {operation} | Status: {status} | Count: {count}"
    
    if success:
        logger.info(message)
    else:
        logger.error(message)
    
    if errors:
        logger.error(f"Errors in {operation}: {len(errors)} errors")
        for i, error in enumerate(errors[:5]):  # Log first 5 errors
            logger.error(f"  Error {i+1}: {error}")
        if len(errors) > 5:
            logger.error(f"  ... and {len(errors) - 5} more errors")


def configure_module_loggers():
    """Configure specific loggers for different modules with appropriate levels."""
    
    # Set specific log levels for different modules
    module_configs = {
        'yc_parser.yc_client': logging.INFO,
        'yc_parser.linkedin_scraper': logging.INFO,
        'yc_parser.data_manager': logging.INFO,
        'yc_parser.company_processor': logging.INFO,
        'requests': logging.WARNING,  # Reduce noise from requests library
        'urllib3': logging.WARNING,   # Reduce noise from urllib3
        'streamlit': logging.WARNING, # Reduce noise from streamlit
    }
    
    for module_name, level in module_configs.items():
        logger = logging.getLogger(module_name)
        logger.setLevel(level)


def setup_application_logging(
    environment: str = "development",
    log_directory: str = "logs"
) -> logging.Logger:
    """
    Set up application-wide logging with environment-specific configuration.
    
    Args:
        environment: Environment name (development, production, testing)
        log_directory: Directory to store log files
        
    Returns:
        Configured root logger
    """
    # Create logs directory
    Path(log_directory).mkdir(exist_ok=True)
    
    # Environment-specific configuration
    if environment.lower() == "production":
        log_level = "INFO"
        console_output = False
        log_file = os.path.join(log_directory, "yc_parser.log")
    elif environment.lower() == "testing":
        log_level = "WARNING"
        console_output = True
        log_file = os.path.join(log_directory, "yc_parser_test.log")
    else:  # development
        log_level = "DEBUG"
        console_output = True
        log_file = os.path.join(log_directory, "yc_parser_dev.log")
    
    # Set up logging
    root_logger = setup_logging(
        log_level=log_level,
        log_file=log_file,
        console_output=console_output,
        structured_format=True
    )
    
    # Configure module-specific loggers
    configure_module_loggers()
    
    # Log startup information
    logger = get_logger(__name__)
    logger.info(f"Application logging initialized for {environment} environment")
    logger.info(f"Log level: {log_level}, Log file: {log_file}, Console: {console_output}")
    
    return root_logger


class LoggingContext:
    """Context manager for temporary logging configuration changes."""
    
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.new_level = level
        self.original_level = None
    
    def __enter__(self):
        self.original_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


def with_logging_level(logger: logging.Logger, level: int):
    """
    Context manager to temporarily change logging level.
    
    Args:
        logger: Logger instance
        level: Temporary logging level
        
    Returns:
        LoggingContext manager
    """
    return LoggingContext(logger, level)