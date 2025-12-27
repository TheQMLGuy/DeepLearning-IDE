// Web-based APIs with Polyglot Python+R execution
// Automatic variable sharing between Python and R

// ============================================
// PYODIDE - Python Runtime
// ============================================

let pyodideInstance: any = null;
let pyodideLoadingPromise: Promise<void> | null = null;
let outputCallback: ((data: string) => void) | null = null;

// Shared variable store
const sharedVars: Record<string, any> = {};

const loadPyodide = async (): Promise<void> => {
    if (pyodideInstance) return;
    if (pyodideLoadingPromise) return pyodideLoadingPromise;

    pyodideLoadingPromise = (async () => {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js';
        document.head.appendChild(script);

        await new Promise<void>((resolve, reject) => {
            script.onload = () => resolve();
            script.onerror = () => reject(new Error('Failed to load Pyodide'));
        });

        pyodideInstance = await (window as any).loadPyodide({
            indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/'
        });

        await pyodideInstance.runPythonAsync(`
import sys
import json

class OutputCapture:
    def write(self, text):
        import js
        if hasattr(js, '_pythonOutput'):
            js._pythonOutput(text)
    def flush(self):
        pass

sys.stdout = OutputCapture()
sys.stderr = OutputCapture()
        `);

        (window as any)._pythonOutput = (text: string) => {
            outputCallback?.(text);
        };

        outputCallback?.('✓ Ready\n');
    })();

    return pyodideLoadingPromise;
};

// ============================================
// WEBR - R Runtime  
// ============================================

let webrInstance: any = null;
let webrLoadingPromise: Promise<void> | null = null;

const loadWebR = async (): Promise<void> => {
    if (webrInstance) return;
    if (webrLoadingPromise) return webrLoadingPromise;

    webrLoadingPromise = (async () => {
        const { WebR } = await import('https://webr.r-wasm.org/latest/webr.mjs' as any);
        webrInstance = new WebR();
        await webrInstance.init();
    })();

    return webrLoadingPromise;
};

// ============================================
// POLYGLOT ENGINE - Auto variable sharing
// ============================================

// Detect language per line
const detectLineLanguage = (line: string): 'python' | 'r' | 'comment' | 'empty' => {
    const trimmed = line.trim();
    if (!trimmed) return 'empty';
    if (trimmed.startsWith('#')) return 'comment';

    // Strong R indicators
    if (/<-/.test(line)) return 'r';
    if (/^\s*library\s*\(/.test(line)) return 'r';
    if (/^\s*c\s*\(/.test(line)) return 'r';
    if (/\$\w+/.test(line)) return 'r';
    if (/^\s*(cat|print)\s*\(.*\\n/.test(line)) return 'r'; // R's cat with \n

    // Strong Python indicators  
    if (/^\s*(import|from)\s+/.test(line)) return 'python';
    if (/^\s*def\s+\w+\s*\(/.test(line)) return 'python';
    if (/^\s*class\s+\w+/.test(line)) return 'python';
    if (/print\s*\(f?["']/.test(line)) return 'python';
    if (/\[.*for.*in.*\]/.test(line)) return 'python'; // list comprehension
    if (/:\s*$/.test(line)) return 'python'; // ends with colon

    return 'python'; // default
};

// Parse code and group into language blocks
const parseIntoBlocks = (code: string): Array<{ lang: 'python' | 'r', code: string, startLine: number }> => {
    const lines = code.split('\n');
    const blocks: Array<{ lang: 'python' | 'r', code: string, startLine: number }> = [];

    let currentLang: 'python' | 'r' | null = null;
    let currentCode: string[] = [];
    let startLine = 1;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmed = line.trim().toLowerCase();

        // Check for explicit markers
        if (trimmed === '#%% python' || trimmed === '# python') {
            if (currentCode.length > 0 && currentLang) {
                blocks.push({ lang: currentLang, code: currentCode.join('\n'), startLine });
            }
            currentLang = 'python';
            currentCode = [];
            startLine = i + 2;
            continue;
        }
        if (trimmed === '#%% r' || trimmed === '# r') {
            if (currentCode.length > 0 && currentLang) {
                blocks.push({ lang: currentLang, code: currentCode.join('\n'), startLine });
            }
            currentLang = 'r';
            currentCode = [];
            startLine = i + 2;
            continue;
        }

        // Auto-detect if no explicit marker
        const lineLang = detectLineLanguage(line);
        if (lineLang !== 'comment' && lineLang !== 'empty') {
            if (currentLang === null) {
                currentLang = lineLang;
                startLine = i + 1;
            } else if (lineLang !== currentLang) {
                // Language switch detected
                if (currentCode.length > 0) {
                    blocks.push({ lang: currentLang, code: currentCode.join('\n'), startLine });
                }
                currentLang = lineLang;
                currentCode = [];
                startLine = i + 1;
            }
        }

        currentCode.push(line);
    }

    // Add final block
    if (currentCode.length > 0 && currentLang) {
        const codeStr = currentCode.join('\n').trim();
        if (codeStr) {
            blocks.push({ lang: currentLang, code: codeStr, startLine });
        }
    }

    return blocks;
};

// Extract variables from Python and inject into R
const syncPythonToR = async (): Promise<void> => {
    if (!pyodideInstance || !webrInstance) return;

    try {
        // Get all Python variables
        const varsJson = await pyodideInstance.runPythonAsync(`
import json
_vars = {}
for name, val in list(globals().items()):
    if not name.startswith('_') and name not in ['sys', 'json', 'OutputCapture']:
        try:
            if isinstance(val, (int, float)):
                _vars[name] = val
            elif isinstance(val, (list, tuple)) and all(isinstance(x, (int, float)) for x in val):
                _vars[name] = list(val)
            elif isinstance(val, str):
                _vars[name] = val
        except:
            pass
json.dumps(_vars)
        `);

        if (varsJson && varsJson !== '{}') {
            const vars = JSON.parse(varsJson);
            for (const [name, value] of Object.entries(vars)) {
                sharedVars[name] = value;
                if (typeof value === 'number') {
                    await webrInstance.evalR(`${name} <- ${value}`);
                } else if (Array.isArray(value)) {
                    await webrInstance.evalR(`${name} <- c(${value.join(', ')})`);
                } else if (typeof value === 'string') {
                    await webrInstance.evalR(`${name} <- "${value}"`);
                }
            }
        }
    } catch (e) {
        // Silent fail
    }
};

// Run Python block
const runPythonBlock = async (code: string): Promise<void> => {
    await loadPyodide();
    try {
        await pyodideInstance.runPythonAsync(code);
    } catch (error: any) {
        outputCallback?.(`Error: ${error.message?.split('\n').pop() || error}\n`);
    }
};

// Run R block - capture output properly
const runRBlock = async (code: string): Promise<void> => {
    await loadWebR();
    await syncPythonToR();

    try {
        // Wrap code in capture.output to get all console output
        const wrappedCode = `capture.output({ ${code} }, type = "output")`;
        const result = await webrInstance.evalR(wrappedCode);

        try {
            const output = await result.toArray();
            if (output && output.length > 0) {
                // Filter out empty lines and join
                const cleanOutput = output.filter((line: string) => line !== '').join('\n');
                if (cleanOutput) {
                    outputCallback?.(cleanOutput + '\n');
                }
            }
        } catch {
            // Try alternative: just get the result directly
            const str = await result.toString();
            if (str) outputCallback?.(str + '\n');
        }
    } catch (error: any) {
        outputCallback?.(`R: ${error.message?.split('\n').slice(-2).join(' ') || error}\n`);
    }
};

// Execute polyglot code
const executePolyglot = async (code: string): Promise<void> => {
    const blocks = parseIntoBlocks(code);

    for (const block of blocks) {
        if (block.lang === 'python') {
            await runPythonBlock(block.code);
        } else {
            await runRBlock(block.code);
        }
    }
};

// Get language annotations for editor
const getLanguageAnnotations = (code: string): Array<{ line: number, lang: 'py' | 'R' }> => {
    const lines = code.split('\n');
    const annotations: Array<{ line: number, lang: 'py' | 'R' }> = [];

    let currentLang: 'py' | 'R' = 'py';

    for (let i = 0; i < lines.length; i++) {
        const trimmed = lines[i].trim().toLowerCase();

        if (trimmed === '#%% python' || trimmed === '# python') {
            currentLang = 'py';
        } else if (trimmed === '#%% r' || trimmed === '# r') {
            currentLang = 'R';
        } else {
            const lang = detectLineLanguage(lines[i]);
            if (lang === 'r') currentLang = 'R';
            else if (lang === 'python') currentLang = 'py';
        }

        if (lines[i].trim() && !lines[i].trim().startsWith('#')) {
            annotations.push({ line: i + 1, lang: currentLang });
        }
    }

    return annotations;
};

// ============================================
// VIRTUAL FILE SYSTEM
// ============================================

interface VirtualFile {
    name: string;
    content?: string;
    isDirectory: boolean;
    path: string;
}

const STORAGE_KEY = 'ai-ide-files';
let virtualFiles: VirtualFile[] = [];

const defaultFiles: VirtualFile[] = [
    {
        name: 'script.py',
        isDirectory: false,
        path: '/project/script.py',
        content: `# Mixed Python + R Code
# Variables are automatically shared!

#%% python  
numbers = [1, 2, 3, 4, 5]
squared = [x ** 2 for x in numbers]
print(f"Numbers: {numbers}")
print(f"Squared: {squared}")

#%% r
cat("Mean:", mean(squared), "\\n")
cat("SD:", sd(squared), "\\n")

#%% python
total = sum(squared)
print(f"Total: {total}")
`
    }
];

const loadFilesFromStorage = (): VirtualFile[] => {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) return JSON.parse(stored);
    } catch (e) { }
    return [...defaultFiles];
};

const saveFilesToStorage = () => {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(virtualFiles));
    } catch (e) { }
};

virtualFiles = loadFilesFromStorage();

const findFile = (path: string): VirtualFile | undefined => {
    return virtualFiles.find(f => f.path === path || f.path === '/project/' + path || f.name === path);
};

const virtualFileSystem = {
    selectDirectory: async (): Promise<string> => '/project',
    readDir: async (path: string) => virtualFiles.filter(f => f.path.startsWith(path)).map(f => ({
        name: f.name, isDirectory: f.isDirectory, path: f.path
    })),
    readFile: async (path: string): Promise<string> => findFile(path)?.content || '',
    saveFile: async (path: string, content: string) => {
        const existing = findFile(path);
        if (existing) {
            existing.content = content;
        } else {
            virtualFiles.push({ name: path.split('/').pop() || path, isDirectory: false, path: '/project/' + path, content });
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
        loadPyodide().catch(console.error);
    },

    send: async (data: string) => {
        const trimmed = data.trim();

        if (trimmed.startsWith('python ') || trimmed.includes('.py')) {
            const match = trimmed.match(/python\s+["']?([^"'\s]+)["']?/);
            if (match) {
                const file = findFile(match[1]);
                if (file?.content) {
                    await executePolyglot(file.content);
                } else {
                    outputCallback?.(`File not found\n`);
                }
            }
        } else if (trimmed.length > 0) {
            await executePolyglot(trimmed);
        }
    },

    onData: (callback: (data: string) => void) => {
        outputCallback = callback;
        return () => { outputCallback = null; };
    }
};

// ============================================
// OTHER APIS
// ============================================

const kaggleAPI = {
    search: async (query: string) => {
        try {
            const res = await fetch('./datasets.json');
            const data = await res.json();
            const all = [
                ...data.kaggle.map((d: any) => ({ ...d, source: 'kaggle' })),
                ...data.tensorflow.map((d: any) => ({ ...d, source: 'tensorflow' })),
                ...data.pytorch.map((d: any) => ({ ...d, source: 'pytorch' })),
                ...data.huggingface.map((d: any) => ({ ...d, source: 'huggingface' }))
            ];
            return {
                success: true, data: all.filter((d: any) =>
                    d.name.toLowerCase().includes(query.toLowerCase()) || d.id.toLowerCase().includes(query.toLowerCase())
                ).slice(0, 20)
            };
        } catch {
            return { success: false, data: [] };
        }
    }
};

// ============================================
// INITIALIZE
// ============================================

window.fileSystem = virtualFileSystem;
window.terminal = terminalAPI;
window.appControl = { setMode: () => { } };
window.kaggle = kaggleAPI;
window.analysis = { checkDeps: async () => ({ missing: [] }), recommend: async () => ({ success: true, recommendation: '' }) };
window.pyodideReady = loadPyodide();

(window as any).runR = runRBlock;
(window as any).loadWebR = loadWebR;
(window as any).getLanguageAnnotations = getLanguageAnnotations;

export { };
