import http.server
import socketserver
import urllib.request
import json
import logging

PORT = 8000

class ProxyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/proxy?url='):
            url = self.path.split('/proxy?url=')[1]
            try:
                # Add a realistic User-Agent to bypass Yahoo's basic blocks
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
                })
                with urllib.request.urlopen(req) as response:
                    self.send_response(200)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    for k, v in response.headers.items():
                        if k.lower() not in ['content-length', 'content-encoding', 'transfer-encoding', 'connection']:
                            self.send_header(k, v)
                    self.end_headers()
                    self.wfile.write(response.read())
            except Exception as e:
                self.send_response(500)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(str(e).encode())
        else:
            super().do_GET()

with socketserver.TCPServer(("", PORT), ProxyHTTPRequestHandler) as httpd:
    print(f"Serving at http://localhost:{PORT}")
    print("Local CORS proxy is ACTIVE. yfinance calls will be routed through /proxy?url=")
    httpd.serve_forever()
