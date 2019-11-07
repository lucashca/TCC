import http.server
import socketserver
import json
import sys

import cgi

from loadModels import ModelsControlers


controler = ModelsControlers()

class my_handler(http.server.SimpleHTTPRequestHandler):

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    def do_HEAD(self):
        self._set_headers()
    def do_GET(self):
        self._set_headers()
        self.wfile.write(json.dumps({'hello': 'world', 'received': 'ok'}).encode("utf8"))
    
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        print("Post")
        ctype, pdict = cgi.parse_header(self.headers.get('content-type'))    
        if ctype != 'application/json':
            self.send_response(400)
            self.end_headers()
            return       
        length = int(self.headers.get('content-length'))
        message = json.loads(self.rfile.read(length))
        message['received'] = 'ok'
        lat = float(message['latitude'])
        lng = float(message['longitude'])
        elevation = float(message['elevation'])
        ca = float(message['ca'])
        print(lat,lng,elevation,ca)
        mg = controler.modelMgPredict(lat,lng,elevation,ca)[0]
        na = controler.modelNaPredict(lat,lng,elevation,ca)[0]
        k = controler.modelKPredict(lat,lng,elevation,ca)[0]
      
        print(ca,mg,na,k)
        # send the message back
        self._set_headers()
        self.wfile.write(json.dumps({'Ca': ca, 'Mg':mg,'Na':na,'K':k,'received': 'ok'}).encode("utf8"))

PORT = int(sys.argv[1])

try:
   httpd = socketserver.ThreadingTCPServer(('', PORT), my_handler)
 
   print ("servidor web rodando na porta ", PORT)
   httpd.serve_forever()
 
except KeyboardInterrupt:
   print("Voce pressionou ^C, encerrando...")
   httpd.socket.close()