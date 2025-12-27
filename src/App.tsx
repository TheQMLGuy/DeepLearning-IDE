import { useState, useEffect } from 'react'
import { Play, RotateCcw, Activity, Loader2, Copy, Check } from 'lucide-react'
import CodeEditor from './components/editor/CodeEditor'
import './App.css'

function App() {
  const [editorCode, setEditorCode] = useState<string>(`# Welcome to AI IDE! 🚀
# Write Python code and click Run to execute

print("Hello, World!")
print("=" * 40)

# Example: List comprehension
numbers = [1, 2, 3, 4, 5]
squared = [x ** 2 for x in numbers]
print(f"Numbers: {numbers}")
print(f"Squared: {squared}")

# Calculate sum
total = sum(squared)
print(f"Sum of squares: {total}")
`)
  const [output, setOutput] = useState<string>('')
  const [isRunning, setIsRunning] = useState(false)
  const [pyodideReady, setPyodideReady] = useState(false)
  const [copied, setCopied] = useState(false)

  // Initialize Pyodide
  useEffect(() => {
    if (window.pyodideReady) {
      window.pyodideReady.then(() => {
        setPyodideReady(true)
        setOutput('✓ Python ready\n')
      }).catch((err) => {
        setOutput(`Error loading Python: ${err}\n`)
      })
    }

    // Listen to terminal output
    const cleanup = window.terminal.onData((data: string) => {
      // Clean up terminal escape codes and format properly
      const cleanData = data
        .replace(/\r\n/g, '\n')
        .replace(/\r/g, '')
        .replace(/\$ $/, '')
        .replace(/^🐍 Running.*\n\n/, '')
        .replace(/^\n+/, '')
        .replace(/\n+$/, '\n')

      if (cleanData.trim()) {
        setOutput(prev => prev + cleanData)
      }
    })

    window.terminal.init()

    return cleanup
  }, [])

  const handleRunCode = async () => {
    if (!pyodideReady || isRunning) return

    setIsRunning(true)
    setOutput('')

    // Save and run the code
    await window.fileSystem.saveFile('script.py', editorCode)
    window.terminal.send(`python "script.py"\n`)

    // Wait a moment then stop running indicator
    setTimeout(() => setIsRunning(false), 500)
  }

  const handleClear = () => {
    setOutput('')
  }

  const handleCopyOutput = () => {
    navigator.clipboard.writeText(output)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="pycharm-container">
      {/* Header */}
      <header className="pycharm-header">
        <div className="header-left">
          <Activity size={20} color="#3b82f6" />
          <span className="app-title">AI IDE</span>
          <span className="web-badge">Web</span>
        </div>

        <div className="header-right">
          {!pyodideReady && (
            <div className="loading-indicator">
              <Loader2 size={14} className="spin" />
              <span>Loading Python...</span>
            </div>
          )}

          <button
            onClick={handleRunCode}
            disabled={!pyodideReady || isRunning}
            className="run-button"
          >
            {isRunning ? (
              <Loader2 size={14} className="spin" />
            ) : (
              <Play size={14} fill="white" />
            )}
            <span>Run</span>
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="pycharm-main">
        {/* Editor */}
        <div className="editor-panel">
          <div className="panel-header">
            <span className="file-tab active">
              <span className="file-icon">🐍</span>
              script.py
            </span>
          </div>
          <div className="editor-content">
            <CodeEditor
              initialValue={editorCode}
              onChange={(val) => setEditorCode(val || '')}
            />
          </div>
        </div>

        {/* Output Panel - PyCharm Style */}
        <div className="output-panel">
          <div className="panel-header">
            <div className="panel-tabs">
              <span className="panel-tab active">Run</span>
            </div>
            <div className="panel-actions">
              <button onClick={handleCopyOutput} className="panel-action-btn" title="Copy output">
                {copied ? <Check size={14} /> : <Copy size={14} />}
              </button>
              <button onClick={handleClear} className="panel-action-btn" title="Clear output">
                <RotateCcw size={14} />
              </button>
            </div>
          </div>
          <div className="output-content">
            <pre className="output-text">{output || 'Click Run to execute your code'}</pre>
          </div>
        </div>
      </div>

      {/* Status Bar */}
      <footer className="pycharm-footer">
        <span>Python 3.11 (Pyodide)</span>
        <span className="footer-separator">|</span>
        <span>UTF-8</span>
        <div className="footer-right">
          <span>{pyodideReady ? '● Ready' : '○ Loading...'}</span>
        </div>
      </footer>
    </div>
  )
}

export default App
