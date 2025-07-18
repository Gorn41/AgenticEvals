"""
Logging utilities for AgenticEvals.
"""

import sys
import logging
from typing import Optional
from pathlib import Path

try:
    from loguru import logger as loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    loguru_logger = None


class LoggerAdapter:
    """Adapter to provide consistent logging interface regardless of backend."""
    
    def __init__(self, name: str, use_loguru: bool = True):
        self.name = name
        self.use_loguru = use_loguru and LOGURU_AVAILABLE
        
        if self.use_loguru:
            self.logger = loguru_logger
        else:
            self.logger = logging.getLogger(name)
    
    def debug(self, message: str, **kwargs):
        if self.use_loguru and self.logger:
            self.logger.debug(f"[{self.name}] {message}", **kwargs)
        elif not self.use_loguru and self.logger:
            self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        if self.use_loguru and self.logger:
            self.logger.info(f"[{self.name}] {message}", **kwargs)
        elif not self.use_loguru and self.logger:
            self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        if self.use_loguru and self.logger:
            self.logger.warning(f"[{self.name}] {message}", **kwargs)
        elif not self.use_loguru and self.logger:
            self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        if self.use_loguru and self.logger:
            self.logger.error(f"[{self.name}] {message}", **kwargs)
        elif not self.use_loguru and self.logger:
            self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        if self.use_loguru and self.logger:
            self.logger.critical(f"[{self.name}] {message}", **kwargs)
        elif not self.use_loguru and self.logger:
            self.logger.critical(message, **kwargs)


def setup_logging(log_level: str = "INFO", 
                  log_file: Optional[Path] = None,
                  use_loguru: bool = True) -> None:
    """Set up logging configuration."""
    
    if use_loguru and LOGURU_AVAILABLE and loguru_logger:
        # Configure loguru
        loguru_logger.remove()  # Remove default handler
        
        # Add console handler
        loguru_logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True
        )
        
        # Add file handler if specified
        if log_file:
            loguru_logger.add(
                log_file,
                level=log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation="10 MB",
                retention="7 days"
            )
    
    else:
        # Configure standard logging
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        
        handlers = [logging.StreamHandler(sys.stderr)]
        
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=handlers
        )


def get_logger(name: str, level: str = "INFO") -> LoggerAdapter:
    """Get a logger instance with a custom adapter."""
    
    use_loguru = LOGURU_AVAILABLE
    setup_logging(log_level=level, use_loguru=use_loguru)
    
    # Return the adapter
    return LoggerAdapter(name, use_loguru=use_loguru)


# Set up default logging
setup_logging() 