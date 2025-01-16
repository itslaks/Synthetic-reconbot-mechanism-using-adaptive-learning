from flask import Flask, request, jsonify, render_template, Blueprint
import re
import dns.resolver
import requests
from urllib.parse import urlparse
import tldextract
import concurrent.futures
import whois
from datetime import datetime
import socket
import ssl
import OpenSSL
from bs4 import BeautifulSoup
from typing import Dict, List, Union

enscan = Blueprint('enscan', __name__, template_folder='templates')

@enscan.route('/')
def index():
    return render_template('enscan.html')

def is_valid_domain(domain: str) -> bool:
    pattern = r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
    return bool(re.match(pattern, domain))

def get_ssl_info(domain: str) -> Dict:
    try:
        context = ssl.create_default_context()
        with socket.create_connection((domain, 443)) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert()
                return {
                    "issuer": dict(x[0] for x in cert['issuer']),
                    "subject": dict(x[0] for x in cert['subject']),
                    "version": cert['version'],
                    "expires": cert['notAfter']
                }
    except Exception as e:
        return {"error": str(e)}

def get_security_headers(domain: str) -> Dict:
    try:
        response = requests.get(f"https://{domain}", timeout=5)
        headers = response.headers
        return {
            "X-Frame-Options": headers.get("X-Frame-Options", "Not set"),
            "X-XSS-Protection": headers.get("X-XSS-Protection", "Not set"),
            "Content-Security-Policy": headers.get("Content-Security-Policy", "Not set"),
            "Strict-Transport-Security": headers.get("Strict-Transport-Security", "Not set")
        }
    except Exception as e:
        return {"error": str(e)}

def get_domain_info(domain: str) -> Dict:
    try:
        w = whois.whois(domain)
        return {
            "registrar": w.registrar,
            "creation_date": w.creation_date.strftime("%Y-%m-%d") if isinstance(w.creation_date, datetime) else str(w.creation_date),
            "expiration_date": w.expiration_date.strftime("%Y-%m-%d") if isinstance(w.expiration_date, datetime) else str(w.expiration_date),
            "name_servers": w.name_servers if hasattr(w, 'name_servers') else None,
            "status": w.status if hasattr(w, 'status') else None,
            "emails": w.emails if hasattr(w, 'emails') else None
        }
    except Exception as e:
        return {"error": str(e)}

def check_spf_dmarc(domain: str) -> Dict:
    spf = dns_query(domain, 'TXT')
    dmarc = dns_query(f"_dmarc.{domain}", 'TXT')
    
    return {
        "spf_record": next((r for r in spf if "v=spf1" in str(r)), "No SPF record found"),
        "dmarc_record": next((r for r in dmarc if "v=DMARC1" in str(r)), "No DMARC record found")
    }

def dns_query(domain: str, record_type: str) -> Union[List[str], str]:
    resolver = dns.resolver.Resolver(configure=False)
    resolver.nameservers = ['8.8.8.8', '8.8.4.4']
    resolver.timeout = 5
    resolver.lifetime = 5
    
    try:
        answers = resolver.resolve(domain, record_type)
        return [str(record) for record in answers]
    except Exception as e:
        return str(e)

def dns_enumeration(domain: str) -> Dict:
    if not is_valid_domain(domain):
        return {"error": "Invalid domain format"}

    results = {}
    record_types = ['A', 'AAAA', 'NS', 'MX', 'TXT', 'SOA', 'CNAME', 'PTR', 'SRV']
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(record_types)) as executor:
        future_to_record = {executor.submit(dns_query, domain, record_type): record_type 
                          for record_type in record_types}
        for future in concurrent.futures.as_completed(future_to_record):
            record_type = future_to_record[future]
            results[record_type] = future.result()

    results.update({
        "domain_info": get_domain_info(domain),
        "ssl_info": get_ssl_info(domain),
        "security_headers": get_security_headers(domain),
        "email_security": check_spf_dmarc(domain)
    })

    return results

@enscan.route('/api/scan', methods=['POST'])
def scan():
    try:
        data = request.json
        input_value = data.get('input', '').strip()

        if not input_value:
            return jsonify({"error": "Empty input provided"}), 400

        result = dns_enumeration(input_value)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def create_app():
    app = Flask(__name__)
    app.register_blueprint(enscan)
    return app

if __name__ == '__main__':
    socket.setdefaulttimeout(10)
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)