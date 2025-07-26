"""
Health Check and Monitoring Module for YC Company Parser

Provides health check endpoints and monitoring capabilities for the application.
"""

import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import requests

from .config import load_config
from .data_manager import DataManager
from .logging_config import get_logger


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    name: str
    status: str  # "pass", "fail", "warn"
    message: str
    timestamp: str
    duration_ms: float
    details: Optional[Dict[str, Any]] = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: str
    total_companies: int
    companies_with_linkedin: int
    yc_s25_mentions: int
    data_file_size_mb: float
    last_processing_time: Optional[float]
    error_count: int
    uptime_seconds: float


class HealthChecker:
    """Performs health checks and collects system metrics."""
    
    def __init__(self):
        self.config = load_config()
        self.logger = get_logger(__name__)
        self.start_time = time.time()
        self.error_count = 0
        self.last_metrics: Optional[SystemMetrics] = None
        
    def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all configured health checks."""
        checks = []
        
        # Get enabled checks from config
        enabled_checks = self.config.get('health_check', {}).get('checks', [
            'data_file_exists',
            'csv_readable', 
            'log_directory_writable',
            'network_connectivity'
        ])
        
        for check_name in enabled_checks:
            try:
                result = self._run_single_check(check_name)
                checks.append(result)
            except Exception as e:
                self.logger.error(f"Health check '{check_name}' failed with exception: {e}")
                checks.append(HealthCheckResult(
                    name=check_name,
                    status="fail",
                    message=f"Check failed with exception: {str(e)}",
                    timestamp=datetime.now().isoformat(),
                    duration_ms=0.0
                ))
        
        return checks
    
    def _run_single_check(self, check_name: str) -> HealthCheckResult:
        """Run a single health check."""
        start_time = time.time()
        
        try:
            if check_name == 'data_file_exists':
                return self._check_data_file_exists(start_time)
            elif check_name == 'csv_readable':
                return self._check_csv_readable(start_time)
            elif check_name == 'log_directory_writable':
                return self._check_log_directory_writable(start_time)
            elif check_name == 'network_connectivity':
                return self._check_network_connectivity(start_time)
            else:
                return HealthCheckResult(
                    name=check_name,
                    status="fail",
                    message=f"Unknown health check: {check_name}",
                    timestamp=datetime.now().isoformat(),
                    duration_ms=(time.time() - start_time) * 1000
                )
                
        except Exception as e:
            return HealthCheckResult(
                name=check_name,
                status="fail",
                message=f"Check failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def _check_data_file_exists(self, start_time: float) -> HealthCheckResult:
        """Check if the main data file exists."""
        csv_path = Path(self.config.get('data', {}).get('csv_file_path', 'yc_s25_companies.csv'))
        
        if csv_path.exists():
            file_size = csv_path.stat().st_size
            return HealthCheckResult(
                name="data_file_exists",
                status="pass",
                message=f"Data file exists: {csv_path}",
                timestamp=datetime.now().isoformat(),
                duration_ms=(time.time() - start_time) * 1000,
                details={"file_path": str(csv_path), "file_size_bytes": file_size}
            )
        else:
            return HealthCheckResult(
                name="data_file_exists",
                status="warn",
                message=f"Data file not found: {csv_path}",
                timestamp=datetime.now().isoformat(),
                duration_ms=(time.time() - start_time) * 1000,
                details={"file_path": str(csv_path)}
            )
    
    def _check_csv_readable(self, start_time: float) -> HealthCheckResult:
        """Check if the CSV file can be read successfully."""
        try:
            data_manager = DataManager()
            df = data_manager.load_existing_data()
            
            return HealthCheckResult(
                name="csv_readable",
                status="pass",
                message=f"CSV file readable with {len(df)} records",
                timestamp=datetime.now().isoformat(),
                duration_ms=(time.time() - start_time) * 1000,
                details={"record_count": len(df), "columns": list(df.columns) if not df.empty else []}
            )
            
        except FileNotFoundError:
            return HealthCheckResult(
                name="csv_readable",
                status="warn",
                message="CSV file not found (empty dataset)",
                timestamp=datetime.now().isoformat(),
                duration_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return HealthCheckResult(
                name="csv_readable",
                status="fail",
                message=f"CSV read failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def _check_log_directory_writable(self, start_time: float) -> HealthCheckResult:
        """Check if the log directory is writable."""
        log_dir = Path(self.config.get('logging', {}).get('log_directory', 'logs'))
        
        try:
            # Create directory if it doesn't exist
            log_dir.mkdir(exist_ok=True)
            
            # Test write access
            test_file = log_dir / f"health_check_{int(time.time())}.tmp"
            test_file.write_text("health check test")
            test_file.unlink()
            
            return HealthCheckResult(
                name="log_directory_writable",
                status="pass",
                message=f"Log directory writable: {log_dir}",
                timestamp=datetime.now().isoformat(),
                duration_ms=(time.time() - start_time) * 1000,
                details={"log_directory": str(log_dir)}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="log_directory_writable",
                status="fail",
                message=f"Log directory not writable: {str(e)}",
                timestamp=datetime.now().isoformat(),
                duration_ms=(time.time() - start_time) * 1000,
                details={"log_directory": str(log_dir)}
            )
    
    def _check_network_connectivity(self, start_time: float) -> HealthCheckResult:
        """Check network connectivity to YC website."""
        timeout = self.config.get('health_check', {}).get('timeout', 10)
        
        try:
            response = requests.get(
                "https://www.ycombinator.com",
                timeout=timeout,
                headers={'User-Agent': 'YC-Parser-Health-Check/1.0'}
            )
            
            if response.status_code == 200:
                return HealthCheckResult(
                    name="network_connectivity",
                    status="pass",
                    message="Network connectivity OK",
                    timestamp=datetime.now().isoformat(),
                    duration_ms=(time.time() - start_time) * 1000,
                    details={"status_code": response.status_code, "response_time_ms": response.elapsed.total_seconds() * 1000}
                )
            else:
                return HealthCheckResult(
                    name="network_connectivity",
                    status="warn",
                    message=f"Unexpected HTTP status: {response.status_code}",
                    timestamp=datetime.now().isoformat(),
                    duration_ms=(time.time() - start_time) * 1000,
                    details={"status_code": response.status_code}
                )
                
        except requests.exceptions.Timeout:
            return HealthCheckResult(
                name="network_connectivity",
                status="fail",
                message=f"Network request timed out after {timeout}s",
                timestamp=datetime.now().isoformat(),
                duration_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return HealthCheckResult(
                name="network_connectivity",
                status="fail",
                message=f"Network connectivity failed: {str(e)}",
                timestamp=datetime.now().isoformat(),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            data_manager = DataManager()
            df = data_manager.load_existing_data()
            
            # Calculate metrics
            total_companies = len(df)
            companies_with_linkedin = len(df[(df['linkedin_url'] != '') & (df['linkedin_url'].notna())]) if not df.empty else 0
            yc_s25_mentions = len(df[df['yc_s25_on_linkedin'] == True]) if not df.empty else 0
            
            # Get data file size
            csv_path = Path(self.config.get('data', {}).get('csv_file_path', 'yc_s25_companies.csv'))
            data_file_size_mb = csv_path.stat().st_size / (1024 * 1024) if csv_path.exists() else 0.0
            
            # Calculate uptime
            uptime_seconds = time.time() - self.start_time
            
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                total_companies=total_companies,
                companies_with_linkedin=companies_with_linkedin,
                yc_s25_mentions=yc_s25_mentions,
                data_file_size_mb=round(data_file_size_mb, 2),
                last_processing_time=None,  # Would need to be tracked separately
                error_count=self.error_count,
                uptime_seconds=round(uptime_seconds, 1)
            )
            
            self.last_metrics = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            self.error_count += 1
            
            # Return minimal metrics on error
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                total_companies=0,
                companies_with_linkedin=0,
                yc_s25_mentions=0,
                data_file_size_mb=0.0,
                last_processing_time=None,
                error_count=self.error_count,
                uptime_seconds=round(time.time() - self.start_time, 1)
            )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a comprehensive health summary."""
        checks = self.run_all_checks()
        metrics = self.collect_metrics()
        
        # Determine overall status
        failed_checks = [c for c in checks if c.status == "fail"]
        warning_checks = [c for c in checks if c.status == "warn"]
        
        if failed_checks:
            overall_status = "unhealthy"
        elif warning_checks:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "checks": [asdict(check) for check in checks],
            "metrics": asdict(metrics),
            "summary": {
                "total_checks": len(checks),
                "passed_checks": len([c for c in checks if c.status == "pass"]),
                "failed_checks": len(failed_checks),
                "warning_checks": len(warning_checks)
            }
        }


class HealthCheckServer:
    """Simple HTTP server for health check endpoints."""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.health_checker = HealthChecker()
        self.logger = get_logger(__name__)
        self.running = False
        self.server_thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the health check server."""
        if self.running:
            self.logger.warning("Health check server is already running")
            return
        
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            
            class HealthHandler(BaseHTTPRequestHandler):
                def __init__(self, health_checker, *args, **kwargs):
                    self.health_checker = health_checker
                    super().__init__(*args, **kwargs)
                
                def do_GET(self):
                    if self.path == '/health':
                        self.handle_health_check()
                    elif self.path == '/metrics':
                        self.handle_metrics()
                    elif self.path == '/status':
                        self.handle_status()
                    else:
                        self.send_error(404, "Not Found")
                
                def handle_health_check(self):
                    """Handle health check endpoint."""
                    try:
                        health_summary = self.health_checker.get_health_summary()
                        
                        # Set HTTP status based on health
                        if health_summary['status'] == 'healthy':
                            status_code = 200
                        elif health_summary['status'] == 'degraded':
                            status_code = 200  # Still operational
                        else:
                            status_code = 503  # Service unavailable
                        
                        self.send_response(status_code)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        
                        response = json.dumps(health_summary, indent=2)
                        self.wfile.write(response.encode())
                        
                    except Exception as e:
                        self.send_error(500, f"Health check failed: {str(e)}")
                
                def handle_metrics(self):
                    """Handle metrics endpoint."""
                    try:
                        metrics = self.health_checker.collect_metrics()
                        
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        
                        response = json.dumps(asdict(metrics), indent=2)
                        self.wfile.write(response.encode())
                        
                    except Exception as e:
                        self.send_error(500, f"Metrics collection failed: {str(e)}")
                
                def handle_status(self):
                    """Handle simple status endpoint."""
                    try:
                        checks = self.health_checker.run_all_checks()
                        failed_checks = [c for c in checks if c.status == "fail"]
                        
                        if failed_checks:
                            status = "FAIL"
                            status_code = 503
                        else:
                            status = "OK"
                            status_code = 200
                        
                        self.send_response(status_code)
                        self.send_header('Content-Type', 'text/plain')
                        self.end_headers()
                        
                        self.wfile.write(status.encode())
                        
                    except Exception as e:
                        self.send_error(500, f"Status check failed: {str(e)}")
                
                def log_message(self, format, *args):
                    """Override to use our logger."""
                    pass  # Suppress default HTTP server logging
            
            # Create handler with health_checker bound
            def handler_factory(*args, **kwargs):
                return HealthHandler(self.health_checker, *args, **kwargs)
            
            # Start server in background thread
            def run_server():
                try:
                    server = HTTPServer(('localhost', self.port), handler_factory)
                    self.logger.info(f"Health check server started on port {self.port}")
                    self.running = True
                    server.serve_forever()
                except Exception as e:
                    self.logger.error(f"Health check server failed: {e}")
                    self.running = False
            
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start health check server: {e}")
    
    def stop(self):
        """Stop the health check server."""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=5)
        self.logger.info("Health check server stopped")


def save_metrics_to_file(metrics: SystemMetrics, file_path: str = "metrics.json"):
    """Save metrics to a JSON file for monitoring."""
    try:
        # Load existing metrics
        metrics_file = Path(file_path)
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []
        
        # Add new metrics
        all_metrics.append(asdict(metrics))
        
        # Keep only last 100 entries to prevent file from growing too large
        if len(all_metrics) > 100:
            all_metrics = all_metrics[-100:]
        
        # Save back to file
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
            
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to save metrics to file: {e}")


def run_health_check_cli() -> int:
    """Run health checks from command line."""
    health_checker = HealthChecker()
    
    try:
        print("🏥 Running health checks...")
        health_summary = health_checker.get_health_summary()
        
        # Display results
        print(f"\nOverall Status: {health_summary['status'].upper()}")
        print(f"Timestamp: {health_summary['timestamp']}")
        print(f"Checks: {health_summary['summary']['passed_checks']}/{health_summary['summary']['total_checks']} passed")
        
        print("\nDetailed Results:")
        for check in health_summary['checks']:
            status_icon = "✅" if check['status'] == "pass" else "⚠️" if check['status'] == "warn" else "❌"
            print(f"{status_icon} {check['name']}: {check['message']}")
        
        print(f"\nSystem Metrics:")
        metrics = health_summary['metrics']
        print(f"  Total companies: {metrics['total_companies']}")
        print(f"  Companies with LinkedIn: {metrics['companies_with_linkedin']}")
        print(f"  YC S25 mentions: {metrics['yc_s25_mentions']}")
        print(f"  Data file size: {metrics['data_file_size_mb']} MB")
        print(f"  Uptime: {metrics['uptime_seconds']} seconds")
        print(f"  Error count: {metrics['error_count']}")
        
        # Return appropriate exit code
        if health_summary['status'] == 'healthy':
            return 0
        elif health_summary['status'] == 'degraded':
            return 1
        else:
            return 2
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 3


if __name__ == "__main__":
    import sys
    sys.exit(run_health_check_cli())