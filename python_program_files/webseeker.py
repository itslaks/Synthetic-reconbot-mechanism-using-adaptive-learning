from flask import Flask, request, jsonify, render_template, Blueprint
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
import requests
import nmap
import re
import dns.resolver
import time
import logging
import socket
import whois
import ssl
import OpenSSL
import concurrent.futures
from datetime import datetime
from functools import lru_cache
from urllib.parse import urlparse
import os
import json

webseeker = Blueprint('webseeker', __name__, url_prefix='/webseeker')

# Configuration
class Config:
    VIRUSTOTAL_API_KEY = "" #user your own virustotal api key
    IPINFO_API_KEY = ""  #user your own ipinfo api key
    ABUSEIPDB_API_KEY = ""   #user your own abuseipdb api key
    CACHE_DURATION = 3600  # 1 hour
    SCAN_TIMEOUT = 300  # 5 minutes
    MAX_CONCURRENT_SCANS = 3
    ALLOWED_SCAN_TYPES = ['quick', 'comprehensive', 'stealth']
    PORT_RANGES = {
        'quick': '20-25,53,80,110,143,443,465,587,993,995,3306,3389,5432,8080,8443',
        'comprehensive': '1-1024',
        'stealth': '21-23,25,53,80,110,111,135,139,143,443,445,993,995,1723,3306,3389,5900,8080'
    }

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Rate limiting configuration
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per minute"],
    storage_uri="memory://"
)

class SecurityScanner:
    def __init__(self):
        self.nmap = nmap.PortScanner()
        self.scan_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=Config.MAX_CONCURRENT_SCANS
        )

    def validate_target(self, target):
        """Validate and normalize target input"""
        try:
            target = re.sub(r'^https?://', '', target)
            target = target.split('/')[0]
            
            if not re.match(r'^[\w\-\.]+\.[a-zA-Z]{2,}$', target):
                return None, "Invalid domain format"
            
            try:
                socket.gethostbyname(target)
            except socket.gaierror:
                return None, "Domain cannot be resolved"
            
            return target, None
            
        except Exception as e:
            logger.error(f"Target validation error: {str(e)}")
            return None, f"Validation error: {str(e)}"

    def get_ip_info(self, target):
        """Fetch enhanced IP and geolocation information"""
        try:
            ip = socket.gethostbyname(target)
            
            # Basic IP info from ipinfo.io
            response = requests.get(
                f"https://ipinfo.io/{ip}/json?token={Config.IPINFO_API_KEY}",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # Enhanced reverse DNS lookup
            try:
                answers = dns.resolver.resolve(dns.reversename.from_address(ip), "PTR")
                reverse_dns = str(answers[0]) if answers else None
            except Exception:
                reverse_dns = socket.getfqdn(ip)
                
            data['reverse_dns'] = reverse_dns
            
            # Enhanced WHOIS information
            try:
                whois_data = whois.whois(target)
                data['whois'] = {
                    'registrar': whois_data.registrar,
                    'creation_date': str(whois_data.creation_date[0]) if isinstance(whois_data.creation_date, list) else str(whois_data.creation_date),
                    'expiration_date': str(whois_data.expiration_date[0]) if isinstance(whois_data.expiration_date, list) else str(whois_data.expiration_date),
                    'name_servers': whois_data.name_servers if isinstance(whois_data.name_servers, list) else [whois_data.name_servers] if whois_data.name_servers else [],
                    'status': whois_data.status if isinstance(whois_data.status, list) else [whois_data.status] if whois_data.status else [],
                    'emails': whois_data.emails if isinstance(whois_data.emails, list) else [whois_data.emails] if whois_data.emails else []
                }
            except Exception:
                data['whois'] = {
                    'registrar': None,
                    'creation_date': None,
                    'expiration_date': None,
                    'name_servers': [],
                    'status': [],
                    'emails': []
                }
                
            # Get ASN details
            try:
                asn_response = requests.get(f"https://ipapi.co/{ip}/json/", timeout=5)
                if asn_response.status_code == 200:
                    asn_data = asn_response.json()
                    data.update({
                        'asn': asn_data.get('asn'),
                        'network': asn_data.get('network'),
                        'network_type': asn_data.get('network_type'),
                        'success': True
                    })
            except Exception:
                pass
                
            # Add abuse contact information
            try:
                abuse_response = requests.get(
                    f"https://api.abuseipdb.com/api/v2/check",
                    params={'ipAddress': ip, 'maxAgeInDays': 90},
                    headers={'Key': Config.ABUSEIPDB_API_KEY},
                    timeout=5
                )
                if abuse_response.status_code == 200:
                    abuse_data = abuse_response.json().get('data', {})
                    data['abuse'] = {
                        'total_reports': abuse_data.get('totalReports'),
                        'confidence_score': abuse_data.get('abuseConfidenceScore'),
                        'last_reported': abuse_data.get('lastReportedAt'),
                        'is_public': abuse_data.get('isPublic'),
                        'abuse_types': abuse_data.get('reports', [])
                    }
            except Exception:
                data['abuse'] = None
                
            return data
                
        except Exception as e:
            logger.error(f"IP info error for {target}: {str(e)}")
            return {"error": str(e)}

    def check_ssl(self, target):
        """Check SSL/TLS certificate information"""
        try:
            context = ssl.create_default_context()
            with socket.create_connection((target, 443)) as sock:
                with context.wrap_socket(sock, server_hostname=target) as ssock:
                    cert = ssock.getpeercert(True)
                    x509 = OpenSSL.crypto.load_certificate(
                        OpenSSL.crypto.FILETYPE_ASN1, cert
                    )
            
            issuer = {}
            for key, value in x509.get_issuer().get_components():
                issuer[key.decode('utf-8')] = value.decode('utf-8')
            
            subject = {}
            for key, value in x509.get_subject().get_components():
                subject[key.decode('utf-8')] = value.decode('utf-8')
            
            return {
                'issuer': issuer,
                'subject': subject,
                'version': x509.get_version(),
                'serial_number': str(x509.get_serial_number()),
                'not_before': x509.get_notBefore().decode('utf-8'),
                'not_after': x509.get_notAfter().decode('utf-8'),
                'expired': x509.has_expired(),
                'cipher': ssock.cipher(),
                'protocol': ssock.version()
            }
        except Exception as e:
            logger.error(f"SSL check error for {target}: {str(e)}")
            return {"error": str(e)}

    def scan_virustotal(self, target):
        """Query VirusTotal API"""
        try:
            headers = {
                "x-apikey": Config.VIRUSTOTAL_API_KEY,
                "accept": "application/json"
            }
            
            # Get domain report
            domain_response = requests.get(
                f"https://www.virustotal.com/api/v3/domains/{target}",
                headers=headers,
                timeout=15
            )
            domain_response.raise_for_status()
            domain_data = domain_response.json()

            # Get URL report
            url_response = requests.get(
                f"https://www.virustotal.com/api/v3/urls/{target}",
                headers=headers,
                timeout=15
            )
            url_data = url_response.json() if url_response.status_code == 200 else {}

            return {
                'domain_report': domain_data.get('data', {}).get('attributes', {}),
                'url_report': url_data.get('data', {}).get('attributes', {}),
                'success': True
            }
        except Exception as e:
            logger.error(f"VirusTotal error for {target}: {str(e)}")
            return {
                "data": {
                    "attributes": {
                        "last_analysis_stats": {
                            "harmless": 0,
                            "malicious": 0,
                            "suspicious": 0,
                            "unrated": 0
                        }
                    }
                },
                "success": False,
                "error": str(e)
            }

    def perform_port_scan(self, target, scan_type='quick'):
        """Perform Nmap port scan"""
        try:
            port_range = Config.PORT_RANGES.get(scan_type, Config.PORT_RANGES['quick'])
            
            scan_args = {
                'quick': '-T4 -sV --version-intensity 5',
                'comprehensive': '-T4 -sV -sC -A --version-all',
                'stealth': '-T2 -sS -Pn --min-rate 100'
            }
            
            self.nmap.scan(target, port_range, arguments=scan_args[scan_type])

            processed_results = {
                'ports': [],
                'scan_stats': {
                    'start_time': datetime.now().isoformat(),
                    'elapsed': self.nmap.scanstats().get('elapsed', '0'),
                    'scan_type': scan_type
                },
                'os_detection': {},
                'success': True
            }

            if len(self.nmap.all_hosts()) > 0:
                host = self.nmap.all_hosts()[0]
                
                # Add OS detection if available
                if hasattr(self.nmap[host], 'osclass'):
                    processed_results['os_detection'] = self.nmap[host]['osclass']
                
                for proto in self.nmap[host].all_protocols():
                    ports = self.nmap[host][proto].keys()
                    for port in ports:
                        port_info = self.nmap[host][proto][port]
                        processed_results['ports'].append({
                            'port': str(port),
                            'protocol': proto,
                            'state': port_info.get('state', ''),
                            'service': port_info.get('name', ''),
                            'version': port_info.get('version', ''),
                            'product': port_info.get('product', ''),
                            'extrainfo': port_info.get('extrainfo', ''),
                            'cpe': port_info.get('cpe', [])
                        })

            return processed_results

        except Exception as e:
            logger.error(f"Port scan error for {target}: {str(e)}")
            return {"error": str(e), "success": False}

    def perform_full_scan(self, target, scan_type):
        """Perform all security checks"""
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    'ip_info': executor.submit(self.get_ip_info, target),
                    'virustotal': executor.submit(self.scan_virustotal, target),
                    'port_scan': executor.submit(self.perform_port_scan, target, scan_type),
                    'ssl_cert': executor.submit(self.check_ssl, target)
                }

                results = {
                    'timestamp': datetime.now().isoformat(),
                    'target': target,
                    'scan_type': scan_type,
                    'success': True
                }
                
                for key, future in futures.items():
                    try:
                        results[key] = future.result(timeout=Config.SCAN_TIMEOUT)
                    except concurrent.futures.TimeoutError:
                        results[key] = {"error": "Scan timed out", "success": False}
                    except Exception as e:
                        results[key] = {"error": str(e), "success": False}

                return results

        except Exception as e:
            logger.error(f"Full scan error for {target}: {str(e)}")
            return {"error": f"Scan failed: {str(e)}", "success": False}

# Initialize scanner
scanner = SecurityScanner()

@webseeker.route('/')
def index():
    return render_template("webseeker.html")

@webseeker.route('/api/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def analyze():
    try:
        data = request.get_json()
        target = data.get('domain', '').strip()
        scan_type = data.get('scanType', 'quick')

        if not target:
            return jsonify({"error": "No target specified", "success": False}), 400
        
        if scan_type not in Config.ALLOWED_SCAN_TYPES:
            return jsonify({"error": "Invalid scan type", "success": False}), 400

        normalized_target, error = scanner.validate_target(target)
        if error:
            return jsonify({"error": error, "success": False}), 400

        results = scanner.perform_full_scan(normalized_target, scan_type)
        
        if not results.get("success", False):
            return jsonify({"error": results.get("error", "Unknown error"), "success": False}), 500

        return jsonify(results)

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": "Internal server error", "success": False}), 500

@webseeker.route('/api/quick-check', methods=['GET'])
@limiter.limit("30 per minute")
def quick_check():
    """Lightweight domain check endpoint"""
    try:
        target = request.args.get('domain', '').strip()
        if not target:
            return jsonify({"error": "No target specified", "success": False}), 400

        normalized_target, error = scanner.validate_target(target)
        if error:
            return jsonify({"error": error, "success": False}), 400

        # Quick IP and VirusTotal check only
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            ip_future = executor.submit(scanner.get_ip_info, normalized_target)
            vt_future = executor.submit(scanner.scan_virustotal, normalized_target)

            results = {
                'timestamp': datetime.now().isoformat(),
                'target': normalized_target,
                'ip_info': ip_future.result(timeout=10),
                'virustotal': vt_future.result(timeout=10),
                'success': True
            }

        return jsonify(results)

    except Exception as e:
        logger.error(f"Quick check error: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@webseeker.route('/api/ssl-check', methods=['GET'])
@limiter.limit("20 per minute")
def ssl_check():
    """Standalone SSL certificate checker"""
    try:
        target = request.args.get('domain', '').strip()
        if not target:
            return jsonify({"error": "No target specified", "success": False}), 400

        normalized_target, error = scanner.validate_target(target)
        if error:
            return jsonify({"error": error, "success": False}), 400

        ssl_info = scanner.check_ssl(normalized_target)
        ssl_info['success'] = 'error' not in ssl_info
        return jsonify(ssl_info)

    except Exception as e:
        logger.error(f"SSL check error: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@webseeker.route('/api/port-scan', methods=['POST'])
@limiter.limit("5 per minute")
def port_scan():
    """Standalone port scanner endpoint"""
    try:
        data = request.get_json()
        target = data.get('domain', '').strip()
        scan_type = data.get('scanType', 'quick')
        custom_ports = data.get('ports', None)

        if not target:
            return jsonify({"error": "No target specified", "success": False}), 400

        normalized_target, error = scanner.validate_target(target)
        if error:
            return jsonify({"error": error, "success": False}), 400

        # If custom ports provided, validate them
        if custom_ports:
            if not re.match(r'^[\d,\-]+$', custom_ports):
                return jsonify({"error": "Invalid port format", "success": False}), 400
            Config.PORT_RANGES['custom'] = custom_ports
            scan_type = 'custom'

        results = scanner.perform_port_scan(normalized_target, scan_type)
        return jsonify(results)

    except Exception as e:
        logger.error(f"Port scan error: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@webseeker.route('/api/abuse-check', methods=['GET'])
@limiter.limit("15 per minute")
def abuse_check():
    """Check IP address against AbuseIPDB"""
    try:
        target = request.args.get('domain', '').strip()
        if not target:
            return jsonify({"error": "No target specified", "success": False}), 400

        normalized_target, error = scanner.validate_target(target)
        if error:
            return jsonify({"error": error, "success": False}), 400

        ip = socket.gethostbyname(normalized_target)
        response = requests.get(
            "https://api.abuseipdb.com/api/v2/check",
            params={
                'ipAddress': ip,
                'maxAgeInDays': 90,
                'verbose': True
            },
            headers={'Key': Config.ABUSEIPDB_API_KEY},
            timeout=10
        )
        response.raise_for_status()
        
        return jsonify({
            'target': normalized_target,
            'ip': ip,
            'data': response.json().get('data', {}),
            'success': True
        })

    except Exception as e:
        logger.error(f"Abuse check error: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@webseeker.route('/api/whois', methods=['GET'])
@limiter.limit("15 per minute")
def whois_lookup():
    """Standalone WHOIS lookup endpoint"""
    try:
        target = request.args.get('domain', '').strip()
        if not target:
            return jsonify({"error": "No target specified", "success": False}), 400

        normalized_target, error = scanner.validate_target(target)
        if error:
            return jsonify({"error": error, "success": False}), 400

        whois_data = whois.whois(normalized_target)
        
        # Process and clean WHOIS data
        processed_data = {
            'domain': normalized_target,
            'registrar': whois_data.registrar,
            'creation_date': str(whois_data.creation_date[0]) if isinstance(whois_data.creation_date, list) else str(whois_data.creation_date),
            'expiration_date': str(whois_data.expiration_date[0]) if isinstance(whois_data.expiration_date, list) else str(whois_data.expiration_date),
            'name_servers': whois_data.name_servers if isinstance(whois_data.name_servers, list) else [whois_data.name_servers] if whois_data.name_servers else [],
            'status': whois_data.status if isinstance(whois_data.status, list) else [whois_data.status] if whois_data.status else [],
            'emails': whois_data.emails if isinstance(whois_data.emails, list) else [whois_data.emails] if whois_data.emails else [],
            'dnssec': whois_data.dnssec,
            'success': True
        }

        return jsonify(processed_data)

    except Exception as e:
        logger.error(f"WHOIS lookup error: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@webseeker.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded", "success": False}), 429

@webseeker.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({"error": "Internal server error", "success": False}), 500

if __name__ == '__main__':
    logger.info("Starting security scanner server...")
    webseeker.run(debug=False, host='0.0.0.0', port=5000)