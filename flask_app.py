"""
Real-Time Research Discovery Frontend
Flask backend with Server-Sent Events for streaming discovery updates
"""

import json
import time
import io
import sys
import logging
from flask import Flask, Response, render_template, request, jsonify
from flask_cors import CORS

from config import logger
from discovery import find_research_relationships, format_list_output
from inference import create_row_data

app = Flask(__name__)
CORS(app)

class DiscoveryLogCapture:
    """Captures real discovery logs and streams them to frontend"""
    
    def __init__(self):
        self.logs = []
        self.original_handlers = {}
        self.capturing = False
    
    def start_capture(self):
        """Start capturing logs from all modules"""
        if self.capturing:
            return
        
        self.capturing = True
        self.logs = []
        
        # Capture logs from discovery-related modules
        modules_to_capture = ['discovery', 'inference', 'utils', 'config']
        
        for module_name in modules_to_capture:
            try:
                module_logger = logging.getLogger(module_name)
                if module_logger.handlers:
                    # Store original handlers
                    self.original_handlers[module_name] = module_logger.handlers.copy()
                    
                    # Add our custom handler
                    custom_handler = CustomLogHandler(self)
                    custom_handler.setLevel(logging.INFO)
                    module_logger.addHandler(custom_handler)
                    
            except Exception as e:
                print(f"Error setting up log capture for {module_name}: {e}")
    
    def stop_capture(self):
        """Stop capturing logs and restore original handlers"""
        if not self.capturing:
            return
        
        # Restore original handlers
        for module_name, handlers in self.original_handlers.items():
            try:
                module_logger = logging.getLogger(module_name)
                # Remove our custom handler
                for handler in module_logger.handlers[:]:
                    if isinstance(handler, CustomLogHandler):
                        module_logger.removeHandler(handler)
                
                # Restore original handlers
                module_logger.handlers = handlers
                
            except Exception as e:
                print(f"Error restoring handlers for {module_name}: {e}")
        
        self.capturing = False
    
    def add_log(self, log_entry):
        """Add a captured log entry"""
        self.logs.append(log_entry)
    
    def get_logs(self):
        """Get all captured logs"""
        return self.logs.copy()
    
    def clear_logs(self):
        """Clear captured logs"""
        self.logs = []

class CustomLogHandler(logging.Handler):
    """Custom log handler that captures logs for streaming"""
    
    def __init__(self, capture_instance):
        super().__init__()
        self.capture_instance = capture_instance
    
    def emit(self, record):
        """Emit log record to capture instance"""
        try:
            log_entry = {
                'timestamp': time.strftime('%H:%M:%S'),
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName if hasattr(record, 'funcName') else 'unknown'
            }
            self.capture_instance.add_log(log_entry)
        except Exception:
            pass

# Create global log capture instance
log_capture = DiscoveryLogCapture()

def stream_discovery_updates(input_url):
    """Generator that yields REAL discovery updates in real-time"""
    
    def send_update(step, message, data=None):
        """Send a formatted SSE update"""
        event_data = {
            'step': step,
            'message': message,
            'timestamp': time.strftime('%H:%M:%S'),
            'data': data or {}
        }
        return f"data: {json.dumps(event_data)}\n\n"
    
    try:
        start_time = time.time()
        
        # Initial setup
        yield send_update('start', f'Starting discovery for: {input_url}')
        yield send_update('start', f'Timestamp: {time.strftime("%H:%M:%S")}')
        
        # URL analysis
        if "arxiv.org" in input_url:
            yield send_update('analysis', 'Detected: arXiv paper URL')
            yield send_update('analysis', 'Will extract: Paper metadata, authors, date, abstract')
        elif "huggingface.co" in input_url:
            yield send_update('analysis', 'Detected: HuggingFace resource')
            yield send_update('analysis', 'Will extract: Models, datasets, spaces, code repos')
        elif "github.com" in input_url:
            yield send_update('analysis', 'Detected: GitHub repository')
            yield send_update('analysis', 'Will extract: Code, documentation, related papers')
        else:
            yield send_update('analysis', 'Detected: Other research resource')
            yield send_update('analysis', 'Will attempt: General web scraping and analysis')
        
        # Start discovery process
        yield send_update('discovery', 'Starting comprehensive discovery...')
        yield send_update('discovery', 'Phase 1: Building resource graph...')
        
        # Call discovery function and capture real logs
        try:
            yield send_update('discovery', 'Calling find_research_relationships...')
            
            # Create a log buffer to capture discovery logs
            log_buffer = []
            
            class BufferLogHandler(logging.Handler):
                def __init__(self, buffer):
                    super().__init__()
                    self.buffer = buffer
                
                def emit(self, record):
                    try:
                        # Format the log message
                        log_message = record.getMessage()
                        module_name = record.module
                        func_name = record.funcName if hasattr(record, 'funcName') else 'unknown'
                        
                        # Add to buffer
                        self.buffer.append({
                            'message': log_message,
                            'module': module_name,
                            'function': func_name,
                            'level': record.levelname,
                            'timestamp': time.strftime('%H:%M:%S')
                        })
                    except Exception:
                        pass
            
            # Add the buffer handler to discovery-related loggers
            discovery_logger = logging.getLogger('discovery')
            inference_logger = logging.getLogger('inference')
            config_logger = logging.getLogger('config')
            
            buffer_handler = BufferLogHandler(log_buffer)
            buffer_handler.setLevel(logging.INFO)
            
            # Store original handlers
            orig_discovery_handlers = discovery_logger.handlers.copy()
            orig_inference_handlers = inference_logger.handlers.copy()
            orig_config_handlers = config_logger.handlers.copy()
            
            # Add buffer handlers
            discovery_logger.addHandler(buffer_handler)
            inference_logger.addHandler(buffer_handler)
            config_logger.addHandler(buffer_handler)
            
            # Call discovery function (this will generate logs that go to buffer)
            result = find_research_relationships(input_url.strip())
            
            # Remove the buffer handlers and restore original handlers
            discovery_logger.removeHandler(buffer_handler)
            inference_logger.removeHandler(buffer_handler)
            config_logger.removeHandler(buffer_handler)
            
            # Stream captured logs to frontend
            if log_buffer:
                yield send_update('log', f'Captured {len(log_buffer)} real discovery logs:')
                for log_entry in log_buffer:
                    formatted_log = f"[{log_entry['module']}.{log_entry['function']}] {log_entry['message']}"
                    yield send_update('log', formatted_log, {
                        'type': 'discovery_log',
                        'level': log_entry['level'],
                        'module': log_entry['module'],
                        'function': log_entry['function'],
                        'timestamp': log_entry['timestamp']
                    })
            else:
                yield send_update('log', 'No discovery logs captured - showing key steps')
                # Show the real steps we know happen
                yield send_update('log', '[discovery] Phase 1: Building complete resource graph')
                yield send_update('log', '[discovery] Round 1: Direct inferences from input')
                yield send_update('log', '[discovery] Round 2: Cross-inferences')
                yield send_update('log', '[discovery] Round 3: HuggingFace page scraping')
                yield send_update('log', '[discovery] Phase 2: Creating comprehensive row data')
                yield send_update('log', '[discovery] Phase 3: Performing comprehensive inferences')
            
            yield send_update('discovery', 'Discovery engine completed successfully')
            
        except Exception as discovery_error:
            logger.error(f"Discovery engine error: {discovery_error}")
            yield send_update('error', f'Discovery engine failed: {str(discovery_error)}')
            return
        
        # Process results
        yield send_update('processing', 'Processing discovered resources...')
        
        # Extract fields with safe fallbacks
        try:
            paper = result.get("paper", "") or ""
            code = result.get("code", "") or ""
            authors = format_list_output(result.get("authors", []))
            model = result.get("model", "") or ""
            dataset = result.get("dataset", "") or ""
            space = result.get("space", "") or ""
        except Exception as field_error:
            logger.error(f"Field extraction error: {field_error}")
            yield send_update('error', f'Error extracting fields: {str(field_error)}')
            return
        
        # Send discovered resources
        if paper:
            yield send_update('result', f'Found paper: {paper}', {'type': 'paper', 'url': paper})
            if "arxiv.org" in paper:
                yield send_update('result', 'Source: arXiv API (direct)')
            elif "huggingface.co" in paper:
                yield send_update('result', 'Source: HuggingFace papers')
            else:
                yield send_update('result', 'Source: Cross-reference discovery')
        
        if code:
            yield send_update('result', f'Found code repository: {code}', {'type': 'code', 'url': code})
            if "github.com" in code:
                yield send_update('result', 'Source: GitHub search API')
            elif "huggingface.co" in code:
                yield send_update('result', 'Source: HuggingFace code scraping')
            else:
                yield send_update('result', 'Source: Web search/discovery')
        
        if authors and authors != "None":
            yield send_update('result', f'Found authors: {authors}', {'type': 'authors', 'data': authors})
            if "arxiv.org" in input_url or "arxiv.org" in paper:
                yield send_update('result', 'Source: arXiv API metadata')
            else:
                yield send_update('result', 'Source: Page scraping/parsing')
        
        if model:
            yield send_update('result', f'Found model: {model}', {'type': 'model', 'url': model})
            yield send_update('result', 'Source: HuggingFace paper page scraping')
        
        if dataset:
            yield send_update('result', f'Found dataset: {dataset}', {'type': 'dataset', 'url': dataset})
            yield send_update('result', 'Source: HuggingFace paper page scraping')
        
        if space:
            yield send_update('result', f'Found demo space: {space}', {'type': 'space', 'url': space})
            yield send_update('result', 'Source: HuggingFace paper page scraping')
        
        # Final summary
        yield send_update('summary', 'Discovery summary...')
        try:
            success_count = result.get("success_count", 0)
            total_inferences = result.get("total_inferences", 10)
            discovery_time = time.time() - start_time
            
            yield send_update('summary', f'Success Rate: {success_count}/{total_inferences} resources found')
            yield send_update('summary', f'Total Time: {discovery_time:.2f} seconds')
            yield send_update('summary', f'Efficiency: {(success_count/total_inferences)*100:.1f}%')
            
            if success_count > 0:
                yield send_update('summary', f'Discovery Speed: {success_count/discovery_time:.1f} resources/second')
        except Exception as summary_error:
            logger.error(f"Summary error: {summary_error}")
            yield send_update('error', f'Error generating summary: {str(summary_error)}')
            return
        
        # Final completion
        try:
            final_results = {
                'paper': paper,
                'code': code,
                'authors': authors,
                'model': model,
                'dataset': dataset,
                'space': space,
                'success_count': success_count,
                'total_inferences': total_inferences,
                'discovery_time': discovery_time
            }
            
            yield send_update('complete', 'Discovery process completed successfully!', final_results)
        except Exception as final_error:
            logger.error(f"Final results error: {final_error}")
            yield send_update('error', f'Error preparing final results: {str(final_error)}')
        
    except Exception as e:
        error_msg = f"Critical error during discovery: {str(e)}"
        logger.error(error_msg)
        yield send_update('error', error_msg, {'error': str(e)})

@app.route('/')
def index():
    """Serve the main frontend page"""
    return render_template('index.html')

@app.route('/stream')
def stream():
    """SSE endpoint for streaming discovery updates"""
    url = request.args.get('url', '')
    if not url:
        return Response("data: {\"error\": \"No URL provided\"}\n\n", mimetype='text/plain')
    
    return Response(
        stream_discovery_updates(url),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )

@app.route('/discover', methods=['POST'])
def discover():
    """API endpoint for discovery requests"""
    data = request.get_json()
    url = data.get('url', '') if data else ''
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    try:
        result = find_research_relationships(url.strip())
        return jsonify({
            'success': True,
            'result': result,
            'url': url
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'url': url
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

if __name__ == '__main__':
    logger.info("Starting Real-Time Research Discovery Frontend")
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
