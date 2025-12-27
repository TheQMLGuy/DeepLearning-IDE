// Web-based APIs with Pyodide integration for in-browser Python execution
// This replaces Electron APIs for GitHub Pages deployment
// Types are defined in vite-env.d.ts

// ============================================
// PYODIDE - In-Browser Python Runtime
// ============================================

let pyodideInstance: any = null;
let pyodideLoadingPromise: Promise<void> | null = null;
let outputCallback: ((data: string) => void) | null = null;

const loadPyodide = async (): Promise<void> => {
    if (pyodideInstance) return;
    if (pyodideLoadingPromise) return pyodideLoadingPromise;

    pyodideLoadingPromise = (async () => {
        console.log('[Pyodide] Loading Python runtime...');
        outputCallback?.('\r\n🔄 Loading Python environment...\r\n');

        // Load Pyodide from CDN
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js';
        document.head.appendChild(script);

        await new Promise<void>((resolve, reject) => {
            script.onload = () => resolve();
            script.onerror = () => reject(new Error('Failed to load Pyodide'));
        });

        // Initialize Pyodide
        pyodideInstance = await (window as any).loadPyodide({
            indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/'
        });

        // Setup stdout/stderr capture
        await pyodideInstance.runPythonAsync(`
import sys
from io import StringIO

class OutputCapture:
    def __init__(self):
        self.buffer = StringIO()
    
    def write(self, text):
        self.buffer.write(text)
        # Trigger JS callback for real-time output
        import js
        if hasattr(js, '_pythonOutput'):
            js._pythonOutput(text)
    
    def flush(self):
        pass
    
    def getvalue(self):
        return self.buffer.getvalue()

sys.stdout = OutputCapture()
sys.stderr = OutputCapture()
        `);

        // Setup JS callback for Python output
        (window as any)._pythonOutput = (text: string) => {
            outputCallback?.(text);
        };

        console.log('[Pyodide] Python runtime ready!');
        outputCallback?.('✅ Python environment ready!\r\n\r\n$ ');
    })();

    return pyodideLoadingPromise;
};

const runPython = async (code: string): Promise<string> => {
    await loadPyodide();

    try {
        // Run the code - output is streamed via _pythonOutput callback
        await pyodideInstance.runPythonAsync(code);
        return ''; // Output was already sent via real-time callback
    } catch (error: any) {
        return `❌ Error: ${error.message || error}`;
    }
};

// ============================================
// VIRTUAL FILE SYSTEM
// ============================================

interface VirtualFile {
    name: string;
    content?: string;
    isDirectory: boolean;
    path: string;
    children?: VirtualFile[];
}

// Sample files for demo
const defaultFiles: VirtualFile[] = [
    {
        name: 'main.py',
        isDirectory: false,
        path: '/project/main.py',
        content: `# Welcome to AI IDE! 🚀
# This is a web-based Python IDE with in-browser execution

print("Hello, World!")
print("=" * 40)

# Let's try some Python
numbers = [1, 2, 3, 4, 5]
squared = [x ** 2 for x in numbers]
print(f"Numbers: {numbers}")
print(f"Squared: {squared}")

# Calculate sum
total = sum(squared)
print(f"Sum of squares: {total}")
`
    },
    {
        name: 'data_analysis.py',
        isDirectory: false,
        path: '/project/data_analysis.py',
        content: `# Data Analysis Example
# Note: NumPy and Pandas work in Pyodide!

print("📊 Data Analysis Demo")
print("=" * 40)

# Basic statistics without numpy
data = [23, 45, 67, 89, 12, 34, 56, 78, 90, 21]
print(f"Data: {data}")
print(f"Min: {min(data)}")
print(f"Max: {max(data)}")
print(f"Mean: {sum(data) / len(data):.2f}")

# Sorting
sorted_data = sorted(data)
print(f"Sorted: {sorted_data}")

# Find median
n = len(sorted_data)
median = sorted_data[n // 2] if n % 2 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
print(f"Median: {median}")
`
    },
    {
        name: 'algorithms.py',
        isDirectory: false,
        path: '/project/algorithms.py',
        content: `# Algorithm Examples
print("🔢 Algorithm Demos")
print("=" * 40)

# Fibonacci sequence
def fibonacci(n):
    """Generate first n Fibonacci numbers"""
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib[:n]

print(f"Fibonacci(10): {fibonacci(10)}")

# Prime number checker
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [x for x in range(2, 30) if is_prime(x)]
print(f"Primes up to 30: {primes}")

# Factorial
def factorial(n):
    return 1 if n <= 1 else n * factorial(n - 1)

print(f"Factorial of 10: {factorial(10):,}")
`
    },
    {
        name: 'examples',
        isDirectory: true,
        path: '/project/examples',
        children: [
            {
                name: 'hello.py',
                isDirectory: false,
                path: '/project/examples/hello.py',
                content: `# Simple Hello World
print("👋 Hello from the examples folder!")
print("This is a simple demo file.")

name = "AI IDE User"
print(f"Welcome, {name}!")
`
            }
        ]
    },
    {
        name: 'README.md',
        isDirectory: false,
        path: '/project/README.md',
        content: `# AI IDE - Web Edition

A fully web-based Python IDE that runs entirely in your browser!

## Features
- 🐍 Real Python execution (via Pyodide)
- 📝 Monaco code editor
- 📁 Virtual file system
- 🎨 Dark theme

## Getting Started
1. Click on a file to open it
2. Edit the code in the editor
3. Press the "Run" button to execute

Enjoy coding! 🚀
`
    }
];

// In-memory file storage with localStorage persistence
const STORAGE_KEY = 'ai-ide-files';

let virtualFiles: VirtualFile[] = [];

const loadFilesFromStorage = (): VirtualFile[] => {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
            return JSON.parse(stored);
        }
    } catch (e) {
        console.log('[VFS] No stored files, using defaults');
    }
    return [...defaultFiles];
};

const saveFilesToStorage = () => {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(virtualFiles));
    } catch (e) {
        console.warn('[VFS] Could not save to localStorage');
    }
};

// Initialize virtual files
virtualFiles = loadFilesFromStorage();

const findFile = (path: string): VirtualFile | undefined => {
    const search = (files: VirtualFile[]): VirtualFile | undefined => {
        for (const file of files) {
            if (file.path === path) return file;
            if (file.children) {
                const found = search(file.children);
                if (found) return found;
            }
        }
        return undefined;
    };
    return search(virtualFiles);
};

const virtualFileSystem = {
    selectDirectory: async (): Promise<string> => {
        return '/project';
    },

    readDir: async (path: string): Promise<{ name: string; isDirectory: boolean; path: string }[]> => {
        if (path === '/project') {
            return virtualFiles.map(f => ({
                name: f.name,
                isDirectory: f.isDirectory,
                path: f.path
            }));
        }

        const dir = findFile(path);
        if (dir?.children) {
            return dir.children.map(f => ({
                name: f.name,
                isDirectory: f.isDirectory,
                path: f.path
            }));
        }

        return [];
    },

    readFile: async (path: string): Promise<string> => {
        const file = findFile(path);
        return file?.content || `# File not found: ${path}`;
    },

    saveFile: async (path: string, content: string): Promise<{ success: boolean }> => {
        const existingFile = findFile(path);
        if (existingFile) {
            existingFile.content = content;
        } else {
            // Create new file
            const name = path.split('/').pop() || 'untitled.py';
            virtualFiles.push({
                name,
                isDirectory: false,
                path,
                content
            });
        }
        saveFilesToStorage();
        return { success: true };
    }
};

// ============================================
// TERMINAL API
// ============================================

const terminalAPI = {
    init: () => {
        console.log('[Terminal] Initialized');
        // Start loading Pyodide in background
        loadPyodide().catch(console.error);
    },

    send: async (data: string) => {
        const trimmed = data.trim();

        // Handle different commands
        if (trimmed.startsWith('python ') || trimmed.includes('.py')) {
            // Extract file path and run
            const match = trimmed.match(/python\s+["']?([^"'\s]+)["']?/);
            if (match) {
                const filePath = match[1];

                const file = findFile(filePath) || findFile('/project/' + filePath);
                if (file && file.content) {
                    const error = await runPython(file.content);
                    if (error) {
                        outputCallback?.(error + '\n');
                    }
                } else {
                    outputCallback?.(`File not found: ${filePath}\n`);
                }
            }
        } else if (trimmed.startsWith('pip install')) {
            // Mock pip install
            const packages = trimmed.replace('pip install', '').replace('-q', '').trim();
            outputCallback?.(`\r\n📦 Installing ${packages}...\r\n`);
            await new Promise(r => setTimeout(r, 500));
            outputCallback?.(`✅ Successfully installed ${packages}\r\n\r\n$ `);
        } else if (trimmed === 'clear' || trimmed === 'cls') {
            outputCallback?.('\x1b[2J\x1b[H$ ');
        } else if (trimmed === 'help') {
            outputCallback?.(`\r\n
📚 AI IDE - Web Terminal Help
================================
Commands:
  python <file>  - Run a Python file
  pip install    - (mocked) Install packages
  clear          - Clear the terminal
  help           - Show this help

Note: This runs Python in your browser using Pyodide!
\r\n$ `);
        } else if (trimmed.length > 0) {
            // Try to run as Python code directly
            outputCallback?.('\r\n');
            const result = await runPython(trimmed);
            if (result) {
                outputCallback?.(`${result}\r\n`);
            }
            outputCallback?.('$ ');
        } else {
            outputCallback?.('\r\n$ ');
        }
    },

    onData: (callback: (data: string) => void) => {
        outputCallback = callback;
        return () => { outputCallback = null; };
    }
};

// ============================================
// APP CONTROL API
// ============================================

const appControlAPI = {
    setMode: (mode: string) => {
        console.log('[AppControl] Mode set to:', mode);
    }
};

// ============================================
// KAGGLE API (Mock for web)
// ============================================

const kaggleAPI = {
    search: async (query: string) => {
        console.log('[Kaggle] Searching for:', query);

        // Return mock results based on query
        const mockDatasets = [
            { id: 'heptapod/titanic', name: 'Titanic Dataset', size: '60KB' },
            { id: 'uciml/iris', name: 'Iris Flower Dataset', size: '5KB' },
            { id: 'zalando-research/fashionmnist', name: 'Fashion MNIST', size: '30MB' },
            { id: 'crowdflower/twitter-airline-sentiment', name: 'Twitter Airline Sentiment', size: '3MB' },
            { id: 'datasnaek/youtube-new', name: 'YouTube Trending Videos', size: '200MB' },
        ];

        const filtered = mockDatasets.filter(d =>
            d.name.toLowerCase().includes(query.toLowerCase()) ||
            d.id.toLowerCase().includes(query.toLowerCase())
        );

        return {
            success: true,
            data: filtered.length > 0 ? filtered : mockDatasets.slice(0, 3)
        };
    }
};

// ============================================
// ANALYSIS API (Mock for web)
// ============================================

const analysisAPI = {
    checkDeps: async (_path: string) => {
        // In web mode, we don't need to check dependencies
        // Pyodide handles this differently
        return { missing: [] };
    },

    recommend: async (_path: string) => {
        return {
            success: true,
            recommendation: 'For web-based analysis, try using the built-in Python libraries or micropip to install packages.'
        };
    }
};

// ============================================
// INITIALIZE
// ============================================

console.log('[Web Mode] Initializing web APIs...');

window.fileSystem = virtualFileSystem;
window.terminal = terminalAPI;
window.appControl = appControlAPI;
window.kaggle = kaggleAPI;
window.analysis = analysisAPI;

// Start loading Pyodide in background
window.pyodideReady = loadPyodide();

console.log('[Web Mode] Web APIs ready!');

export { };
