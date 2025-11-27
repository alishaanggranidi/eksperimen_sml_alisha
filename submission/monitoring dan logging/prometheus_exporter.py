from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')  # req masuk 
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')  # req latency
THROUGHPUT = Counter('http_requests_throughput', 'Total number of requests per second')  # throughput
ERROR_COUNT = Counter('http_requests_error_total', 'Total number of failed requests')  # error
SUCCESS_COUNT = Counter('http_requests_success_total', 'Total number of successful requests')  # success
REQUEST_SIZE = Histogram('http_request_size_bytes', 'Size of HTTP request payload (bytes)')  # request size
RESPONSE_SIZE = Histogram('http_response_size_bytes', 'Size of HTTP response payload (bytes)')  # response size

CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')
DISK_USAGE = Gauge('system_disk_usage', 'Disk Usage Percentage')
NET_BYTES_SENT = Gauge('system_net_bytes_sent', 'Total Bytes Sent')
NET_BYTES_RECV = Gauge('system_net_bytes_recv', 'Total Bytes Received')

# endpoint
@app.route('/metrics', methods=['GET'])
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage('/').percent)
    net_io = psutil.net_io_counters()
    NET_BYTES_SENT.set(net_io.bytes_sent)
    NET_BYTES_RECV.set(net_io.bytes_recv)

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# endpoint predict
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    data = request.get_json()

    try:
        time.sleep(0.1) 
        
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        SUCCESS_COUNT.inc() 

        dummy_response = {"prediction": [13500.50], "status": "success (simulation)"}
        
        print("Prediksi Simulasi Berhasil!")
        return jsonify(dummy_response)

    except Exception as e:
        ERROR_COUNT.inc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)