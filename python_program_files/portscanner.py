from flask import Flask, request, jsonify, render_template, Blueprint
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import nmap
import socket
from datetime import datetime
import concurrent.futures
import os
import validators
from urllib.parse import urlparse
from functools import wraps
import multiprocessing
import time
import threading
import ipaddress

portscanner = Blueprint('portscanner', __name__, url_prefix='/portscanner')

class Config:
    SCAN_TIMEOUT = 300  # 5 minutes max for any scan
    MAX_CONCURRENT_SCANS = multiprocessing.cpu_count() * 2
    DEFAULT_PORTS = '1-1024'
    
    SCAN_TYPES = {
        'intense_scan': {'command': '-T4 -A -v -Pn', 'description': 'Intense scan'},
        'service_version': {'command': '-sV -Pn', 'description': 'Service version detection'},
        'os_detection': {'command': '-O -Pn', 'description': 'OS detection'},
        'tcp_connect': {'command': '-sT -Pn', 'description': 'TCP connect scan'},
        'syn_scan': {'command': '-sS -Pn', 'description': 'SYN scan (requires root)'},
        'udp_scan': {'command': '-sU -Pn', 'description': 'UDP scan'},
        'aggressive_scan': {'command': '-A -Pn', 'description': 'Aggressive scan with version detection'},
        'list_scan': {'command': '-sL -Pn', 'description': 'Lists targets without scanning'},
        'null_scan': {'command': '-sN -Pn', 'description': 'Null scan with no flags'},
        'xmas_scan': {'command': '-sX -Pn', 'description': 'Xmas scan with FIN, PSH, URG flags'},
        'fin_scan': {'command': '-sF -Pn', 'description': 'FIN scan with only FIN flag'},
        'full_port_scan': {'command': '-p- -Pn', 'description': 'Scans all 65,535 ports'},
        'script_scan': {'command': '-sC -Pn', 'description': 'Default script scan'},
        'version_intensity': {'command': '--version-intensity 9 -Pn', 'description': 'Intense version detection'},
        'timing_aggressive': {'command': '-T4 -Pn', 'description': 'Aggressive timing template'},
        'timing_insane': {'command': '-T5 -Pn', 'description': 'Insane timing template'},
        'traceroute': {'command': '--traceroute -Pn', 'description': 'Trace path to host'},
        'fragment_scan': {'command': '-f -Pn', 'description': 'Fragment packets'},
        'idle_scan': {'command': '-sI -Pn', 'description': 'Idle scan'},
        'ack_scan': {'command': '-sA -Pn', 'description': 'ACK scan'},
        'window_scan': {'command': '-sW -Pn', 'description': 'Window scan'},
        'quick_scan': {'command': '-T4 -F', 'description': 'Quick scan'},
        'sctp_init_scan': {'command': '-sY -Pn', 'description': 'SCTP INIT scan'},
        'sctp_cookie_scan': {'command': '-sZ -Pn', 'description': 'SCTP COOKIE-ECHO scan'}
    }

    # Common ports for quick scan (expanded to include most commonly used services)
    QUICK_SCAN_PORTS = [
        20, 21, 22, 23, 25, 53, 80, 81, 88, 110, 111, 123, 135, 137, 138, 139, 143, 161, 162, 
        389, 443, 445, 465, 500, 515, 587, 636, 993, 995, 1080, 1433, 1521, 1723, 2049, 2082, 
        2083, 2086, 2087, 2096, 2097, 2222, 3306, 3389, 5060, 5222, 5432, 5900, 5938, 6379, 
        8000, 8080, 8443, 8888, 9100, 10000
    ]
    
    # Ultra-fast socket connection timeout for quick scan
    QUICK_CONNECT_TIMEOUT = 0.1  # 100ms

# Configure rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "20 per minute"]
)

def requires_root(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        root_required_scans = ['syn_scan', 'null_scan', 'xmas_scan', 'fin_scan', 'idle_scan']
        scan_type = request.json.get('scanType')
        if scan_type in root_required_scans and os.geteuid() != 0:
            return jsonify({
                'error': f'The {scan_type} requires root privileges',
                'success': False
            }), 403
        return f(*args, **kwargs)
    return decorated_function

class DirectScanner:
    """Ultra-fast socket-based scanner for quick scan operations"""
    
    @staticmethod
    def scan_port(ip, port, timeout=Config.QUICK_CONNECT_TIMEOUT):
        """Scan a single port using direct socket connection"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return port, result == 0
    
    @staticmethod
    def get_service_name(port):
        """Get service name for common ports"""
        try:
            return socket.getservbyport(port)
        except:
            return "unknown"
    
    @staticmethod
    def parallel_port_scan(ip, ports, max_workers=100):
        """Scan multiple ports in parallel"""
        open_ports = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_port = {
                executor.submit(DirectScanner.scan_port, ip, port): port 
                for port in ports
            }
            
            for future in concurrent.futures.as_completed(future_to_port):
                port = future_to_port[future]
                try:
                    port, is_open = future.result()
                    if is_open:
                        service = DirectScanner.get_service_name(port)
                        open_ports[port] = {
                            'state': 'open',
                            'service': service,
                            'reason': 'syn-ack'
                        }
                except Exception:
                    pass
        
        return open_ports
    
    @staticmethod
    def scan_target(target, ports=Config.QUICK_SCAN_PORTS):
        """Perform a fast scan of a target"""
        try:
            # Resolve domain to IP if needed
            try:
                socket.inet_aton(target)
                ip = target
            except socket.error:
                ip = socket.gethostbyname(target)
                
            start_time = time.time()
            open_ports = DirectScanner.parallel_port_scan(ip, ports)
            scan_time = time.time() - start_time
            
            result = {
                'scan_info': {
                    'target': target,
                    'scan_type': 'quick_scan',
                    'description': 'Ultra-fast port scan',
                    'command_used': 'direct socket scanning',
                    'start_time': datetime.now().isoformat(),
                    'elapsed': scan_time,
                    'total_hosts': 1,
                    'up_hosts': 1,
                    'down_hosts': 0
                },
                'hosts': {
                    ip: {
                        'state': 'up',
                        'protocols': {
                            'tcp': open_ports
                        },
                        'hostnames': [{'name': target, 'type': ''}] if target != ip else []
                    }
                },
                'success': True
            }
            
            return result
        except Exception as e:
            return {"error": str(e), "success": False}

class PortScanner:
    def __init__(self):
        self.nmap = nmap.PortScanner()
        self.scan_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=Config.MAX_CONCURRENT_SCANS
        )
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=max(2, multiprocessing.cpu_count() - 1)
        )
        self.active_scans = {}
        self.direct_scanner = DirectScanner()

    def validate_target(self, target):
        """Validate and normalize target input (supports IP, domain, URL, CIDR, and ranges)"""
        try:
            # Remove protocol and path if URL
            if '//' in target:
                parsed = urlparse(target)
                target = parsed.netloc or parsed.path
            
            # Remove any remaining path components and query parameters
            target = target.split('/')[0].split('?')[0]
            
            # Check if CIDR notation
            if '/' in target and target.split('/')[1].isdigit():
                try:
                    ipaddress.ip_network(target, strict=False)
                    return target, None
                except ValueError:
                    pass
            
            # Check if IP range (e.g., 192.168.1.1-10)
            if '-' in target and not any(c.isalpha() for c in target):
                base_parts = target.split('-')[0].split('.')
                if len(base_parts) == 4 and all(part.isdigit() for part in base_parts):
                    return target, None
            
            # Check if IP address
            try:
                socket.inet_aton(target)
                return target, None
            except socket.error:
                pass
            
            # Check if domain
            if validators.domain(target) or target.startswith('localhost'):
                try:
                    socket.gethostbyname(target)
                    return target, None
                except socket.gaierror:
                    return None, "Domain cannot be resolved"
            
            return None, "Invalid target format. Please provide a valid IP address, domain, URL, CIDR, or range."
            
        except Exception as e:
            return None, f"Validation error: {str(e)}"

    def perform_port_scan(self, target, scan_type):
        """Perform intelligent port scan based on scan type"""
        try:
            if scan_type not in Config.SCAN_TYPES:
                return {"error": "Invalid scan type", "success": False}
            
            # Start timing
            start_time = time.time()
            
            # For quick scan, use the ultra-fast direct scanner instead of nmap
            if scan_type == 'quick_scan':
                # Use our ultra-fast direct socket scanner for quick scan
                return DirectScanner.scan_target(target)
            
            # For all other scan types, use optimized nmap
            scan_args = Config.SCAN_TYPES[scan_type]['command']
            
            # Determine ports to scan
            if scan_type == 'full_port_scan':
                ports = '1-65535'
            else:
                ports = Config.DEFAULT_PORTS
            
            # Add performance optimization flags for nmap scans
            if '--host-timeout' not in scan_args:
                scan_args += f' --host-timeout={Config.SCAN_TIMEOUT}s'
            
            if scan_type in ['intense_scan', 'aggressive_scan', 'timing_aggressive', 'timing_insane']:
                if '--min-rate' not in scan_args:
                    scan_args += ' --min-rate=1000'
                if '--min-parallelism' not in scan_args:
                    scan_args += ' --min-parallelism=10'
            
            try:
                # Run the nmap scan with optimized settings
                scan_results = self.nmap.scan(
                    hosts=target,
                    ports=ports,
                    arguments=scan_args,
                    timeout=Config.SCAN_TIMEOUT
                )
                
                if not scan_results or 'scan' not in scan_results:
                    return {"error": "Scan produced no results", "success": False}

            except nmap.PortScannerError as e:
                return {"error": f"Scan failed: {str(e)}", "success": False}
            except Exception as e:
                return {"error": f"Unexpected error: {str(e)}", "success": False}

            # Calculate actual scan duration
            scan_duration = time.time() - start_time
            
            processed_results = {
                'scan_info': {
                    'target': target,
                    'scan_type': scan_type,
                    'description': Config.SCAN_TYPES[scan_type]['description'],
                    'command_used': f"nmap {scan_args} -p {ports} {target}",
                    'start_time': datetime.now().isoformat(),
                    'elapsed': scan_duration,
                    'total_hosts': self.nmap.scanstats().get('totalhosts', '0'),
                    'up_hosts': self.nmap.scanstats().get('uphosts', '0'),
                    'down_hosts': self.nmap.scanstats().get('downhosts', '0')
                },
                'hosts': {},
                'success': True
            }

            for host in self.nmap.all_hosts():
                host_data = {
                    'state': self.nmap[host].state(),
                    'protocols': {},
                    'hostnames': self.nmap[host].hostnames()
                }

                # Add OS detection results if available
                if hasattr(self.nmap[host], 'osmatch') and self.nmap[host].osmatch():
                    host_data['os_matches'] = self.nmap[host].osmatch()

                # Add traceroute if available
                if hasattr(self.nmap[host], 'traceroute'):
                    host_data['traceroute'] = self.nmap[host].traceroute()

                # Process each protocol
                for proto in self.nmap[host].all_protocols():
                    ports = self.nmap[host][proto].keys()
                    host_data['protocols'][proto] = {}
                    
                    for port in ports:
                        port_info = self.nmap[host][proto][port]
                        port_data = {
                            'state': port_info.get('state'),
                            'service': port_info.get('name'),
                            'product': port_info.get('product', ''),
                            'version': port_info.get('version', ''),
                            'extrainfo': port_info.get('extrainfo', ''),
                            'reason': port_info.get('reason', ''),
                            'cpe': port_info.get('cpe', [])
                        }

                        # Add script output if available
                        if 'script' in port_info:
                            port_data['scripts'] = port_info['script']

                        host_data['protocols'][proto][port] = port_data

                processed_results['hosts'][host] = host_data

            # Add performance metrics
            processed_results['scan_info']['performance'] = {
                'actual_duration': scan_duration,
                'ports_per_second': len(ports.split(',')) if ',' in ports else 
                                   (int(ports.split('-')[1]) - int(ports.split('-')[0]) + 1 
                                    if '-' in ports else 1) / max(0.1, scan_duration)
            }

            return processed_results

        except Exception as e:
            return {"error": str(e), "success": False}

    def async_scan(self, target, scan_type):
        """Submit scan to the process pool for background processing"""
        future = self.process_pool.submit(self.perform_port_scan, target, scan_type)
        scan_id = id(future)
        self.active_scans[scan_id] = {
            'future': future,
            'target': target,
            'scan_type': scan_type,
            'start_time': datetime.now().isoformat()
        }
        return scan_id
    
    def get_scan_status(self, scan_id):
        """Check status of an asynchronous scan"""
        if scan_id not in self.active_scans:
            return {"error": "Scan ID not found", "success": False}
        
        future = self.active_scans[scan_id]['future']
        if future.done():
            try:
                result = future.result()
                # Clean up after getting the result
                del self.active_scans[scan_id]
                return result
            except Exception as e:
                del self.active_scans[scan_id]
                return {"error": str(e), "success": False}
        else:
            # Return status for in-progress scan
            return {
                "status": "in_progress",
                "target": self.active_scans[scan_id]['target'],
                "scan_type": self.active_scans[scan_id]['scan_type'],
                "start_time": self.active_scans[scan_id]['start_time'],
                "elapsed": (datetime.now() - datetime.fromisoformat(self.active_scans[scan_id]['start_time'])).total_seconds(),
                "success": True
            }

# Initialize scanner
scanner = PortScanner()

@portscanner.route('/')
def index():
    return render_template('portscanner.html')

@portscanner.route('/api/scan-types', methods=['GET'])
def get_scan_types():
    """Return available scan types and their descriptions"""
    return jsonify({
        'scan_types': Config.SCAN_TYPES,
        'success': True
    })

@portscanner.route('/api/scan', methods=['POST'])
@limiter.limit("10 per minute")
@requires_root
def port_scan():
    """Port scanner endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided", "success": False}), 400

        target = data.get('target', '').strip()
        scan_type = data.get('scanType', 'quick_scan')
        async_mode = data.get('async', False)

        if not target:
            return jsonify({"error": "No target specified", "success": False}), 400

        normalized_target, error = scanner.validate_target(target)
        if error:
            return jsonify({"error": error, "success": False}), 400

        if scan_type not in Config.SCAN_TYPES:
            return jsonify({"error": "Invalid scan type", "success": False}), 400

        if async_mode:
            # Start scan asynchronously
            scan_id = scanner.async_scan(normalized_target, scan_type)
            return jsonify({
                "scan_id": scan_id,
                "status": "started",
                "message": "Scan started in background",
                "success": True
            })
        else:
            # Run scan synchronously
            results = scanner.perform_port_scan(normalized_target, scan_type)
            if not results.get('success', False):
                return jsonify(results), 400

            return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@portscanner.route('/api/scan-status/<int:scan_id>', methods=['GET'])
def check_scan_status(scan_id):
    """Check status of an asynchronous scan"""
    try:
        status = scanner.get_scan_status(scan_id)
        if not status.get('success', False) and status.get('error'):
            return jsonify(status), 400
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@portscanner.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded", "success": False}), 429

def create_app():
    app = Flask(__name__)
    app.register_blueprint(portscanner)
    limiter.init_app(app)
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=False, host='0.0.0.0', port=5000)
