from flask import Blueprint, request, render_template, jsonify
import json
from collections import OrderedDict

osint = Blueprint('osint', __name__, template_folder='templates')

# Load JSON data
with open('data.json', 'r') as f:
    user_data = json.load(f)

# Define popular platforms to be displayed first
popular_platforms = ['Instagram', 'LinkedIn', 'Facebook', 'GitHub', 'LeetCode', 'Apple']

@osint.route('/')
def index():
    return render_template('osint.html')

@osint.route('/search', methods=['POST'])
def search():
    username = request.form['username']
    results = OrderedDict()

    # First, check popular platforms
    for platform in popular_platforms:
        if platform.lower() in user_data and 'url' in user_data[platform.lower()]:
            try:
                url = user_data[platform.lower()]['url'].format(username)
                results[platform] = url
            except KeyError:
                continue

    # Then, check all other platforms
    for site, site_data in sorted(user_data.items()):
        if site.capitalize() not in results and 'url' in site_data:
            try:
                url = site_data['url'].format(username)
                results[site.capitalize()] = url
            except KeyError:
                continue

    if results:
        return jsonify(results)
    else:
        return jsonify({"error": "No profiles found"}), 404