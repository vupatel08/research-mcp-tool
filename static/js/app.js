// Research Discovery Engine - Smooth Real-Time Frontend

class DiscoveryEngine {
    constructor() {
        this.eventSource = null;
        this.isDiscovering = false;
        this.currentStep = '';
        this.logCount = 0;
        this.results = {
            paper: '',
            code: '',
            authors: '',
            model: '',
            dataset: '',
            space: ''
        };

        this.initializeElements();
        this.bindEvents();
        this.resetUI();
    }

    initializeElements() {
        // Input elements
        this.urlInput = document.getElementById('urlInput');
        this.discoverBtn = document.getElementById('discoverBtn');

        // Status elements
        this.statusText = document.getElementById('statusText');
        this.progressBar = document.getElementById('progressFill');

        // Log element
        this.discoveryLog = document.getElementById('discoveryLog');

        // Result elements
        this.resultElements = {
            paper: {
                item: document.getElementById('paperResult'),
                content: document.getElementById('paperContent')
            },
            code: {
                item: document.getElementById('codeResult'),
                content: document.getElementById('codeContent')
            },
            authors: {
                item: document.getElementById('authorsResult'),
                content: document.getElementById('authorsContent')
            },
            model: {
                item: document.getElementById('modelResult'),
                content: document.getElementById('modelContent')
            },
            dataset: {
                item: document.getElementById('datasetResult'),
                content: document.getElementById('datasetContent')
            },
            space: {
                item: document.getElementById('spaceResult'),
                content: document.getElementById('spaceContent')
            }
        };

        // Summary elements removed
    }

    bindEvents() {
        // Main discover button
        this.discoverBtn.addEventListener('click', () => this.startDiscovery());

        // Enter key in input
        this.urlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !this.isDiscovering) {
                this.startDiscovery();
            }
        });

        // Example buttons
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const url = e.target.dataset.url;
                this.urlInput.value = url;
                this.startDiscovery();
            });
        });

        // Auto-focus input on page load
        this.urlInput.focus();
    }

    resetUI() {
        this.statusText.textContent = 'Ready to discover research resources';
        this.progressBar.style.width = '0%';
        this.logCount = 0;

        // Clear results
        Object.keys(this.results).forEach(key => {
            this.results[key] = '';
            this.resultElements[key].content.textContent = '-';
            this.resultElements[key].item.classList.remove('found');
        });

        // Summary removed
    }

    startDiscovery() {
        const url = this.urlInput.value.trim();
        if (!url || this.isDiscovering) return;

        this.isDiscovering = true;
        this.discoverBtn.disabled = true;
        this.discoverBtn.textContent = 'Discovering...';
        this.discoverBtn.classList.add('loading');

        this.resetUI();
        this.clearLog();

        this.statusText.textContent = 'Starting discovery...';
        this.progressBar.style.width = '5%';

        // Start SSE connection
        this.connectSSE(url);
    }

    connectSSE(url) {
        // Close existing connection
        if (this.eventSource) {
            this.eventSource.close();
        }

        const encodedUrl = encodeURIComponent(url);
        this.eventSource = new EventSource(`/stream?url=${encodedUrl}`);

        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleDiscoveryUpdate(data);
            } catch (error) {
                console.error('Error parsing SSE data:', error);
                this.addLogEntry('error', 'Error parsing server response');
            }
        };

        this.eventSource.onerror = (error) => {
            console.error('SSE Error:', error);
            this.handleDiscoveryComplete();
            this.addLogEntry('error', 'Connection error occurred');
        };
    }

    handleDiscoveryUpdate(data) {
        const { step, message, timestamp, data: updateData } = data;

        // Update status and progress
        this.updateStatus(step, message);
        this.updateProgress(step);

        // Add log entry with smooth animation
        this.addLogEntry(step, message, updateData);

        // Handle result data
        if (updateData && updateData.type) {
            this.updateResult(updateData.type, updateData);
        }

        // Handle completion
        if (step === 'complete' && updateData) {
            this.handleFinalResults(updateData);
            setTimeout(() => this.handleDiscoveryComplete(), 1000);
        }

        // Handle errors
        if (step === 'error') {
            setTimeout(() => this.handleDiscoveryComplete(), 1000);
        }
    }

    updateStatus(step, message) {
        let statusText = '';

        switch (step) {
            case 'start':
                statusText = 'Initializing discovery...';
                break;
            case 'analysis':
                statusText = 'Analyzing URL...';
                break;
            case 'discovery':
                statusText = 'Discovering resources...';
                break;
            case 'log':
                statusText = `Processing discovery logs... (${this.logCount} entries)`;
                break;
            case 'processing':
                statusText = 'Processing results...';
                break;
            case 'result':
                statusText = 'Found resources!';
                break;
            case 'summary':
                statusText = 'Generating summary...';
                break;
            case 'complete':
                statusText = 'Discovery completed successfully!';
                break;
            case 'error':
                statusText = 'Discovery failed';
                break;
            default:
                statusText = message;
        }

        this.statusText.textContent = statusText;
    }

    updateProgress(step) {
        let progress = 5;

        switch (step) {
            case 'start':
                progress = 10;
                break;
            case 'analysis':
                progress = 15;
                break;
            case 'discovery':
                progress = 25;
                break;
            case 'log':
                progress = 40 + (this.logCount * 0.5); // Gradual increase with logs
                break;
            case 'processing':
                progress = 70;
                break;
            case 'result':
                progress = 85;
                break;
            case 'summary':
                progress = 95;
                break;
            case 'complete':
                progress = 100;
                break;
            case 'error':
                progress = 100;
                break;
        }

        this.progressBar.style.width = `${Math.min(progress, 100)}%`;
    }

    addLogEntry(step, message, data = null) {
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${step}`;

        // Handle different log types
        if (step === 'log' && data && data.type === 'discovery_log') {
            // Real discovery log entry
            logEntry.textContent = message;
            logEntry.setAttribute('data-level', data.level || 'INFO');
            logEntry.setAttribute('data-module', data.module || 'unknown');

            // Add timestamp prefix for real logs
            if (data.timestamp) {
                const timestampSpan = document.createElement('span');
                timestampSpan.textContent = `[${data.timestamp}] `;
                timestampSpan.style.color = 'var(--text-muted)';
                timestampSpan.style.fontSize = '10px';
                logEntry.insertBefore(timestampSpan, logEntry.firstChild);
            }

            this.logCount++;
        } else {
            // Regular log entry
            logEntry.textContent = message;
        }

        // Add smooth entrance animation
        logEntry.style.opacity = '0';
        logEntry.style.transform = 'translateY(10px)';

        this.discoveryLog.appendChild(logEntry);

        // Trigger animation
        requestAnimationFrame(() => {
            logEntry.style.opacity = '1';
            logEntry.style.transform = 'translateY(0)';
        });

        // Auto-scroll to bottom smoothly
        this.smoothScrollToBottom();
    }

    smoothScrollToBottom() {
        const scrollHeight = this.discoveryLog.scrollHeight;
        const height = this.discoveryLog.clientHeight;
        const maxScrollTop = scrollHeight - height;

        // Smooth scroll animation
        const start = this.discoveryLog.scrollTop;
        const change = maxScrollTop - start;
        const duration = 300;
        let currentTime = 0;

        const animateScroll = () => {
            currentTime += 16; // ~60fps
            const val = this.easeInOutQuad(currentTime, start, change, duration);
            this.discoveryLog.scrollTop = val;
            if (currentTime < duration) {
                requestAnimationFrame(animateScroll);
            }
        };

        if (change > 0) {
            animateScroll();
        }
    }

    easeInOutQuad(t, b, c, d) {
        t /= d / 2;
        if (t < 1) return c / 2 * t * t + b;
        t--;
        return -c / 2 * (t * (t - 2) - 1) + b;
    }

    updateResult(type, data) {
        if (!this.resultElements[type]) return;

        const { item, content } = this.resultElements[type];

        if (data.url) {
            this.results[type] = data.url;
            content.innerHTML = `<a href="${data.url}" target="_blank" rel="noopener">${data.url}</a>`;
        } else if (data.data) {
            this.results[type] = data.data;
            content.textContent = data.data;
        }

        // Mark as found with smooth animation
        item.classList.add('found');
    }

    handleFinalResults(data) {
        // Update any missing results
        Object.keys(this.results).forEach(key => {
            if (data[key] && !this.results[key]) {
                this.updateResult(key, { url: data[key] });
            }
        });
    }



    handleDiscoveryComplete() {
        this.isDiscovering = false;
        this.discoverBtn.disabled = false;
        this.discoverBtn.textContent = 'Discover';
        this.discoverBtn.classList.remove('loading');

        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }

        // Focus input for next discovery
        setTimeout(() => {
            this.urlInput.focus();
        }, 500);
    }

    clearLog() {
        // Fade out existing entries
        const entries = this.discoveryLog.querySelectorAll('.log-entry:not(.welcome)');
        entries.forEach(entry => {
            entry.style.opacity = '0';
            entry.style.transform = 'translateY(-10px)';
        });

        // Clear after animation
        setTimeout(() => {
            this.discoveryLog.innerHTML = '';
        }, 200);
    }
}

// Initialize the discovery engine when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DiscoveryEngine();
});

// Add smooth CSS transitions
const style = document.createElement('style');
style.textContent = `
    .log-entry {
        transition: opacity 0.3s ease, transform 0.3s ease;
    }
    
    .result-item {
        transition: all 0.3s ease;
    }
    
    .summary-item span:last-child {
        transition: transform 0.2s ease;
    }
`;
document.head.appendChild(style);