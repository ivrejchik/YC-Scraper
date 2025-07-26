"""
Configuration Management Module

Handles loading and validation of application configuration from YAML files.
"""
import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration settings."""
    name: str = "YC Company Parser"
    version: str = "1.0.0"
    environment: str = "production"
    debug: bool = False


@dataclass
class BatchConfig:
    """YC batch configuration settings."""
    default_season: str = "Summer"
    default_year: int = 2025
    default_batch_code: str = "S25"  # Will be auto-generated from season/year if not specified


@dataclass
class DataConfig:
    """Data storage configuration settings."""
    csv_file_path: str = "yc_{batch}_companies.csv"  # Will be formatted with batch
    backup_directory: str = "data_backups"
    backup_retention_days: int = 30
    auto_backup: bool = True
    backup_on_startup: bool = True


@dataclass
class ProcessingConfig:
    """Processing configuration settings."""
    max_workers: int = 3
    batch_size: int = 10
    enable_resume: bool = True
    resume_state_file: str = "yc_{batch}_companies_resume_state.json"  # Will be formatted with batch
    validate_on_startup: bool = True
    auto_repair_data: bool = False


@dataclass
class RateLimitingConfig:
    """Rate limiting and timeout configuration."""
    linkedin_delay_min: float = 1.0
    linkedin_delay_max: float = 2.0
    request_timeout: int = 30
    connect_timeout: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0


@dataclass
class YcApiConfig:
    """YC API configuration settings."""
    base_url: str = "https://www.ycombinator.com"
    api_base: str = "https://yc-oss.github.io/api/batches"
    api_endpoint_template: str = "{season_lower}-{year}.json"  # Will be formatted with season/year
    user_agent: str = "YC Company Parser/1.0"
    headers: Dict[str, str] = field(default_factory=lambda: {
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9"
    })


@dataclass
class LinkedInConfig:
    """LinkedIn scraping configuration settings."""
    user_agents: List[str] = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ])
    headers: Dict[str, str] = field(default_factory=lambda: {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    })


@dataclass
class LinkedInDiscoveryConfig:
    """LinkedIn discovery configuration settings."""
    default_batch: str = "S25"
    search_query_template: str = "site:linkedin.com {company_name} YC {batch}"
    max_results_per_search: int = 10
    confidence_threshold: float = 0.7
    enable_fuzzy_matching: bool = True
    fuzzy_match_threshold: float = 0.8
    search_delay_min: float = 2.0
    search_delay_max: float = 4.0
    max_retries: int = 3
    retry_delay: float = 5.0
    normalize_company_names: bool = True
    skip_existing_linkedin_urls: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_directory: str = "logs"
    log_file: str = "yc_parser.log"
    error_log_file: str = "yc_parser_errors.log"
    max_log_size_mb: int = 10
    backup_count: int = 5
    console_logging: bool = True
    console_level: str = "INFO"


@dataclass
class StreamlitConfig:
    """Streamlit application configuration."""
    page_title: str = "YC Company Parser"
    page_icon: str = ""
    layout: str = "wide"
    show_error_recovery: bool = True
    show_system_status: bool = True
    auto_refresh_interval: int = 0
    max_description_length: int = 100
    show_clickable_links: bool = True


@dataclass
class HealthCheckConfig:
    """Health check configuration settings."""
    enabled: bool = True
    port: int = 8080
    endpoint: str = "/health"
    check_interval: int = 60
    timeout: int = 10
    checks: List[str] = field(default_factory=lambda: [
        "data_file_exists",
        "csv_readable",
        "log_directory_writable",
        "network_connectivity"
    ])
    auto_start_server: bool = False


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""
    enabled: bool = False
    collect_metrics: bool = False
    metrics_file: str = "metrics.json"
    track_processing_time: bool = True
    track_memory_usage: bool = False
    track_network_requests: bool = True
    alerts_enabled: bool = False
    error_threshold: int = 5
    processing_time_threshold: int = 300


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    verify_ssl: bool = True
    allow_redirects: bool = True
    max_redirects: int = 5
    sanitize_logs: bool = True
    mask_sensitive_data: bool = True


@dataclass
class DevelopmentConfig:
    """Development-specific configuration."""
    use_mock_data: bool = False
    mock_company_count: int = 10
    verbose_logging: bool = True
    save_raw_responses: bool = False
    skip_linkedin_checks: bool = False
    fast_mode: bool = False


@dataclass
class Config:
    """Main configuration container."""
    app: AppConfig = field(default_factory=AppConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    data: DataConfig = field(default_factory=DataConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    rate_limiting: RateLimitingConfig = field(default_factory=RateLimitingConfig)
    yc_api: YcApiConfig = field(default_factory=YcApiConfig)
    linkedin: LinkedInConfig = field(default_factory=LinkedInConfig)
    linkedin_discovery: LinkedInDiscoveryConfig = field(default_factory=LinkedInDiscoveryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    streamlit: StreamlitConfig = field(default_factory=StreamlitConfig)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default locations.
        
    Returns:
        Dictionary containing configuration settings.
    """
    # Determine config file path
    if config_path is None:
        config_path = os.environ.get('YC_PARSER_CONFIG', 'config.yaml')
    
    config_file = Path(config_path)
    
    # Load default configuration
    default_config = Config()
    
    # Convert to dictionary for easier manipulation
    config_dict = {
        'app': {
            'name': default_config.app.name,
            'version': default_config.app.version,
            'environment': default_config.app.environment,
            'debug': default_config.app.debug
        },
        'batch': {
            'default_season': default_config.batch.default_season,
            'default_year': default_config.batch.default_year,
            'default_batch_code': default_config.batch.default_batch_code
        },
        'data': {
            'csv_file_path': default_config.data.csv_file_path,
            'backup_directory': default_config.data.backup_directory,
            'backup_retention_days': default_config.data.backup_retention_days,
            'auto_backup': default_config.data.auto_backup,
            'backup_on_startup': default_config.data.backup_on_startup
        },
        'processing': {
            'max_workers': default_config.processing.max_workers,
            'batch_size': default_config.processing.batch_size,
            'enable_resume': default_config.processing.enable_resume,
            'resume_state_file': default_config.processing.resume_state_file,
            'validate_on_startup': default_config.processing.validate_on_startup,
            'auto_repair_data': default_config.processing.auto_repair_data
        },
        'rate_limiting': {
            'linkedin_delay_min': default_config.rate_limiting.linkedin_delay_min,
            'linkedin_delay_max': default_config.rate_limiting.linkedin_delay_max,
            'request_timeout': default_config.rate_limiting.request_timeout,
            'connect_timeout': default_config.rate_limiting.connect_timeout,
            'max_retries': default_config.rate_limiting.max_retries,
            'retry_delay': default_config.rate_limiting.retry_delay,
            'backoff_factor': default_config.rate_limiting.backoff_factor
        },
        'yc_api': {
            'base_url': default_config.yc_api.base_url,
            'api_base': default_config.yc_api.api_base,
            'api_endpoint_template': default_config.yc_api.api_endpoint_template,
            'user_agent': default_config.yc_api.user_agent,
            'headers': default_config.yc_api.headers
        },
        'linkedin': {
            'user_agents': default_config.linkedin.user_agents,
            'headers': default_config.linkedin.headers
        },
        'linkedin_discovery': {
            'default_batch': default_config.linkedin_discovery.default_batch,
            'search_query_template': default_config.linkedin_discovery.search_query_template,
            'max_results_per_search': default_config.linkedin_discovery.max_results_per_search,
            'confidence_threshold': default_config.linkedin_discovery.confidence_threshold,
            'enable_fuzzy_matching': default_config.linkedin_discovery.enable_fuzzy_matching,
            'fuzzy_match_threshold': default_config.linkedin_discovery.fuzzy_match_threshold,
            'search_delay_min': default_config.linkedin_discovery.search_delay_min,
            'search_delay_max': default_config.linkedin_discovery.search_delay_max,
            'max_retries': default_config.linkedin_discovery.max_retries,
            'retry_delay': default_config.linkedin_discovery.retry_delay,
            'normalize_company_names': default_config.linkedin_discovery.normalize_company_names,
            'skip_existing_linkedin_urls': default_config.linkedin_discovery.skip_existing_linkedin_urls
        },
        'logging': {
            'level': default_config.logging.level,
            'format': default_config.logging.format,
            'log_directory': default_config.logging.log_directory,
            'log_file': default_config.logging.log_file,
            'error_log_file': default_config.logging.error_log_file,
            'max_log_size_mb': default_config.logging.max_log_size_mb,
            'backup_count': default_config.logging.backup_count,
            'console_logging': default_config.logging.console_logging,
            'console_level': default_config.logging.console_level
        },
        'streamlit': {
            'page_title': default_config.streamlit.page_title,
            'page_icon': default_config.streamlit.page_icon,
            'layout': default_config.streamlit.layout,
            'show_error_recovery': default_config.streamlit.show_error_recovery,
            'show_system_status': default_config.streamlit.show_system_status,
            'auto_refresh_interval': default_config.streamlit.auto_refresh_interval,
            'max_description_length': default_config.streamlit.max_description_length,
            'show_clickable_links': default_config.streamlit.show_clickable_links
        },
        'health_check': {
            'enabled': default_config.health_check.enabled,
            'port': default_config.health_check.port,
            'endpoint': default_config.health_check.endpoint,
            'check_interval': default_config.health_check.check_interval,
            'timeout': default_config.health_check.timeout,
            'checks': default_config.health_check.checks,
            'auto_start_server': default_config.health_check.auto_start_server
        },
        'monitoring': {
            'enabled': default_config.monitoring.enabled,
            'collect_metrics': default_config.monitoring.collect_metrics,
            'metrics_file': default_config.monitoring.metrics_file,
            'track_processing_time': default_config.monitoring.track_processing_time,
            'track_memory_usage': default_config.monitoring.track_memory_usage,
            'track_network_requests': default_config.monitoring.track_network_requests,
            'alerts_enabled': default_config.monitoring.alerts_enabled,
            'error_threshold': default_config.monitoring.error_threshold,
            'processing_time_threshold': default_config.monitoring.processing_time_threshold
        },
        'security': {
            'verify_ssl': default_config.security.verify_ssl,
            'allow_redirects': default_config.security.allow_redirects,
            'max_redirects': default_config.security.max_redirects,
            'sanitize_logs': default_config.security.sanitize_logs,
            'mask_sensitive_data': default_config.security.mask_sensitive_data
        },
        'development': {
            'use_mock_data': default_config.development.use_mock_data,
            'mock_company_count': default_config.development.mock_company_count,
            'verbose_logging': default_config.development.verbose_logging,
            'save_raw_responses': default_config.development.save_raw_responses,
            'skip_linkedin_checks': default_config.development.skip_linkedin_checks,
            'fast_mode': default_config.development.fast_mode
        }
    }
    
    # Load from YAML file if it exists
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            
            if yaml_config:
                # Deep merge YAML config with defaults
                config_dict = _deep_merge(config_dict, yaml_config)
                logger.info(f"Configuration loaded from {config_file}")
            else:
                logger.warning(f"Configuration file {config_file} is empty, using defaults")
                
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            logger.info("Using default configuration")
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            logger.info("Using default configuration")
    else:
        logger.info(f"Configuration file {config_file} not found, using defaults")
    
    # Store config path for reference
    config_dict['_config_path'] = str(config_file)
    
    return config_dict


def _deep_merge(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base_dict: Base dictionary to merge into
        update_dict: Dictionary with updates to apply
        
    Returns:
        Merged dictionary
    """
    result = base_dict.copy()
    
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def generate_batch_code(season: str, year: int) -> str:
    """
    Generate batch code from season and year.
    
    Args:
        season: "Summer" or "Winter"
        year: Year (e.g., 2025, 2024)
        
    Returns:
        Batch code like "S25", "W24"
    """
    season_code = "S" if season.lower() == "summer" else "W"
    year_short = str(year)[-2:]  # Last 2 digits
    return f"{season_code}{year_short}"


def parse_batch_code(batch_code: str) -> tuple[str, int]:
    """
    Parse batch code into season and year.
    
    Args:
        batch_code: Batch code like "S25", "W24"
        
    Returns:
        Tuple of (season, year)
    """
    if len(batch_code) < 3:
        raise ValueError(f"Invalid batch code format: {batch_code}")
    
    season_code = batch_code[0].upper()
    year_short = batch_code[1:]
    
    season = "Summer" if season_code == "S" else "Winter"
    year = int(f"20{year_short}")
    
    return season, year


def format_config_paths(config: Dict[str, Any], batch_code: str) -> Dict[str, Any]:
    """
    Format configuration paths with batch code.
    
    Args:
        config: Configuration dictionary
        batch_code: Batch code like "S25", "W24"
        
    Returns:
        Configuration with formatted paths
    """
    config = config.copy()
    
    # Format data file path
    if 'data' in config and 'csv_file_path' in config['data']:
        config['data']['csv_file_path'] = config['data']['csv_file_path'].format(
            batch=batch_code.lower()
        )
    
    # Format resume state file path
    if 'processing' in config and 'resume_state_file' in config['processing']:
        config['processing']['resume_state_file'] = config['processing']['resume_state_file'].format(
            batch=batch_code.lower()
        )
    
    return config


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration settings.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Validate required sections
    required_sections = ['app', 'data', 'processing', 'rate_limiting', 'logging']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required configuration section: {section}")
    
    # Validate specific settings
    if 'processing' in config:
        processing = config['processing']
        if processing.get('max_workers', 0) < 1:
            errors.append("processing.max_workers must be at least 1")
        if processing.get('batch_size', 0) < 1:
            errors.append("processing.batch_size must be at least 1")
    
    if 'rate_limiting' in config:
        rate_limiting = config['rate_limiting']
        if rate_limiting.get('linkedin_delay_min', 0) < 0:
            errors.append("rate_limiting.linkedin_delay_min must be non-negative")
        if rate_limiting.get('linkedin_delay_max', 0) < rate_limiting.get('linkedin_delay_min', 0):
            errors.append("rate_limiting.linkedin_delay_max must be >= linkedin_delay_min")
    
    if 'health_check' in config:
        health_check = config['health_check']
        port = health_check.get('port', 8080)
        if not (1 <= port <= 65535):
            errors.append("health_check.port must be between 1 and 65535")
    
    return errors


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    errors = validate_config(config)
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration loaded and validated successfully")
        print(f"Environment: {config['app']['environment']}")
        print(f"Debug mode: {config['app']['debug']}")
        print(f"Data file: {config['data']['csv_file_path']}")
        print(f"Log directory: {config['logging']['log_directory']}")
        print(f"Health check enabled: {config['health_check']['enabled']}")