import { useState, useEffect, useRef, useCallback } from 'react'
import { Play, RotateCcw, Loader2, Copy, Check, Database, BarChart2, Table2, Boxes, Search, RefreshCw, GripHorizontal } from 'lucide-react'
import CodeEditor from './components/editor/CodeEditor'
import './App.css'

interface Dataset {
  id: string;
  name: string;
  category?: string;
  size?: string;
  source: string;
}

const sampleDataFrame = [
  { id: 1, name: 'Alice', age: 28, salary: 55000, dept: 'Engineering' },
  { id: 2, name: 'Bob', age: 34, salary: 62000, dept: 'Marketing' },
  { id: 3, name: 'Charlie', age: 25, salary: 48000, dept: 'Engineering' },
  { id: 4, name: 'Diana', age: 31, salary: 71000, dept: 'Sales' },
  { id: 5, name: 'Eve', age: 29, salary: 58000, dept: 'Engineering' },
]

// Detect language per line
const detectLanguage = (line: string): 'py' | 'R' | null => {
  const trimmed = line.trim()
  if (!trimmed || trimmed.startsWith('#')) return null
  if (/<-/.test(line) || /^\s*library\s*\(/.test(line) || /^\s*c\s*\(/.test(line)) return 'R'
  if (/print\s*\(f?["']/.test(line) || /^\s*(import|from|def|class)\s+/.test(line) || /\[.*for.*in.*\]/.test(line)) return 'py'
  return 'py'
}

function App() {
  const [editorCode, setEditorCode] = useState<string>(`# Python + R Notebook
# Variables automatically shared!

#%% python
numbers = [1, 2, 3, 4, 5]
squared = [x ** 2 for x in numbers]
print(f"Numbers: {numbers}")
print(f"Squared: {squared}")

#%% r
cat("Mean:", mean(squared), "\\n")
cat("SD:", sd(squared), "\\n")

#%% python
print(f"Sum: {sum(squared)}")
`)
  const [output, setOutput] = useState<string>('')
  const [isRunning, setIsRunning] = useState(false)
  const [pyodideReady, setPyodideReady] = useState(false)
  const [rReady, setRReady] = useState(false)
  const [copied, setCopied] = useState(false)

  // Panel states
  const [showArchitecture, setShowArchitecture] = useState(false)
  const [showDatabase, setShowDatabase] = useState(false)
  const [showPandas, setShowPandas] = useState(false)
  const [showR, setShowR] = useState(false)

  // Resizable output panel
  const [outputHeight, setOutputHeight] = useState(200)
  const resizeRef = useRef<HTMLDivElement>(null)
  const isResizing = useRef(false)

  // Dataset state
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [datasetSearch, setDatasetSearch] = useState('')
  const [datasetFilter, setDatasetFilter] = useState<string>('all')
  const [loadingDatasets, setLoadingDatasets] = useState(false)

  // DataFrame state
  const [dataFrame, setDataFrame] = useState(sampleDataFrame)
  const [selectedCell, setSelectedCell] = useState<{ row: number, col: string } | null>(null)

  // Language annotations for gutter
  const [langAnnotations, setLangAnnotations] = useState<Array<{ line: number, lang: 'py' | 'R' }>>([])

  // Update language annotations when code changes
  useEffect(() => {
    const lines = editorCode.split('\n')
    const annotations: Array<{ line: number, lang: 'py' | 'R' }> = []
    let currentLang: 'py' | 'R' = 'py'

    for (let i = 0; i < lines.length; i++) {
      const trimmed = lines[i].trim().toLowerCase()
      if (trimmed === '#%% python' || trimmed === '# python') currentLang = 'py'
      else if (trimmed === '#%% r' || trimmed === '# r') currentLang = 'R'
      else {
        const detected = detectLanguage(lines[i])
        if (detected) currentLang = detected
      }

      if (lines[i].trim() && !lines[i].trim().startsWith('#')) {
        annotations.push({ line: i + 1, lang: currentLang })
      }
    }
    setLangAnnotations(annotations)
  }, [editorCode])

  // Load datasets
  const loadDatasets = async () => {
    setLoadingDatasets(true)
    try {
      const res = await fetch('./datasets.json')
      const data = await res.json()
      setDatasets([
        ...data.kaggle.map((d: any) => ({ ...d, source: 'kaggle' })),
        ...data.tensorflow.map((d: any) => ({ ...d, source: 'tensorflow' })),
        ...data.pytorch.map((d: any) => ({ ...d, source: 'pytorch' })),
        ...data.huggingface.map((d: any) => ({ ...d, source: 'huggingface' }))
      ])
    } catch (e) { console.error(e) }
    setLoadingDatasets(false)
  }

  // Initialize both Python AND R on startup
  useEffect(() => {
    loadDatasets()

    // Load Python
    if (window.pyodideReady) {
      window.pyodideReady.then(() => {
        setPyodideReady(true)
        setOutput('✓ Python ready\n')
      }).catch((err) => setOutput(`Error: ${err}\n`))
    }

    // Auto-load R (in background)
    if ((window as any).loadWebR) {
      (window as any).loadWebR()
        .then(() => {
          setRReady(true)
          setOutput(prev => prev + '✓ R ready\n')
        })
        .catch(console.error)
    }

    const cleanup = window.terminal.onData((data: string) => {
      const cleanData = data.replace(/\r\n/g, '\n').replace(/\r/g, '\n')
      if (cleanData.trim() || cleanData.includes('\n')) {
        setOutput(prev => prev + cleanData)
      }
    })

    window.terminal.init()
    return cleanup
  }, [])

  // Resize handler
  const handleMouseDown = useCallback(() => {
    isResizing.current = true
    document.body.style.cursor = 'row-resize'
    document.body.style.userSelect = 'none'
  }, [])

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isResizing.current) return
    const container = document.querySelector('.center-panel')
    if (!container) return
    const rect = container.getBoundingClientRect()
    const newHeight = rect.bottom - e.clientY
    setOutputHeight(Math.max(100, Math.min(500, newHeight)))
  }, [])

  const handleMouseUp = useCallback(() => {
    isResizing.current = false
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
  }, [])

  useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [handleMouseMove, handleMouseUp])

  const handleRunCode = async () => {
    if (!pyodideReady || isRunning) return
    setIsRunning(true)
    setOutput('')
    await window.fileSystem.saveFile('script.py', editorCode)
    window.terminal.send(`python "script.py"\n`)
    setTimeout(() => setIsRunning(false), 800)
  }

  const runRCode = async (code: string) => {
    if (!(window as any).runR) return
    setOutput(prev => prev + '\n')
    try {
      await (window as any).runR(code)
    } catch (e: any) {
      setOutput(prev => prev + `R: ${e.message}\n`)
    }
  }

  const handleClear = () => setOutput('')
  const handleCopyOutput = () => {
    navigator.clipboard.writeText(output)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleCellEdit = (rowIdx: number, col: string, value: string) => {
    const newData = [...dataFrame]
    const numValue = parseFloat(value)
      ; (newData[rowIdx] as any)[col] = isNaN(numValue) ? value : numValue
    setDataFrame(newData)
  }

  const addPyTorchLayer = (code: string) => setEditorCode(prev => prev + `\n${code}`)

  const filteredDatasets = datasets.filter(d => {
    const matchSearch = d.name.toLowerCase().includes(datasetSearch.toLowerCase())
    const matchFilter = datasetFilter === 'all' || d.source === datasetFilter
    return matchSearch && matchFilter
  })

  const generateDatasetCode = (d: Dataset) => {
    if (d.source === 'pytorch') return `# ${d.name}\nfrom torchvision import datasets\ntrain = datasets.${d.id}(root='./data', train=True, download=True)\nprint(f"Samples: {len(train)}")`
    if (d.source === 'tensorflow') return `# ${d.name}\nimport tensorflow as tf\n(x_train, y_train), _ = tf.keras.datasets.${d.id}.load_data()\nprint(f"Shape: {x_train.shape}")`
    return `# ${d.name} from ${d.source}\n# ID: ${d.id}\nprint("Dataset: ${d.name}")`
  }

  return (
    <div className="ide-container">
      {/* Header */}
      <header className="ide-header">
        <div className="header-left">
          <span className="app-title">🔥 AI IDE</span>
          <span className="badge pytorch">Py</span>
          <span className="badge r-badge">R</span>
        </div>

        <div className="header-center">
          <button className={`toggle-btn ${showArchitecture ? 'active' : ''}`} onClick={() => setShowArchitecture(!showArchitecture)}>
            <Boxes size={14} /><span>Arch</span>
          </button>
          <button className={`toggle-btn ${showDatabase ? 'active' : ''}`} onClick={() => setShowDatabase(!showDatabase)}>
            <Database size={14} /><span>Data</span>
          </button>
          <button className={`toggle-btn ${showPandas ? 'active' : ''}`} onClick={() => setShowPandas(!showPandas)}>
            <Table2 size={14} /><span>DF</span>
          </button>
          <button className={`toggle-btn ${showR ? 'active' : ''}`} onClick={() => setShowR(!showR)}>
            <BarChart2 size={14} /><span>R</span>
          </button>
        </div>

        <div className="header-right">
          {!pyodideReady && <span className="status-dot loading">Py</span>}
          {!rReady && <span className="status-dot loading">R</span>}
          {pyodideReady && <span className="status-dot ready">Py</span>}
          {rReady && <span className="status-dot ready">R</span>}
          <button onClick={handleRunCode} disabled={!pyodideReady || isRunning} className="run-button">
            {isRunning ? <Loader2 size={14} className="spin" /> : <Play size={14} fill="white" />}
            <span>Run</span>
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="ide-main">
        {/* Left - Pandas */}
        {showPandas && (
          <div className="side-panel left-panel">
            <div className="panel-header"><Table2 size={14} /><span>DataFrame</span></div>
            <div className="panel-content dataframe-panel">
              <div className="df-info">{dataFrame.length} × {Object.keys(dataFrame[0] || {}).length}</div>
              <div className="dataframe-container">
                <table className="dataframe">
                  <thead><tr><th>#</th>{Object.keys(dataFrame[0] || {}).map(col => <th key={col}>{col}</th>)}</tr></thead>
                  <tbody>
                    {dataFrame.map((row, idx) => (
                      <tr key={idx}>
                        <td className="row-idx">{idx}</td>
                        {Object.entries(row).map(([col, val]) => (
                          <td key={col} className={selectedCell?.row === idx && selectedCell?.col === col ? 'selected' : ''}
                            onClick={() => setSelectedCell({ row: idx, col })}
                            onDoubleClick={() => { const v = prompt(`${col}:`, String(val)); if (v !== null) handleCellEdit(idx, col, v) }}>
                            {typeof val === 'number' ? val.toLocaleString() : val}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="df-actions">
                <button onClick={() => setEditorCode(prev => prev + `\nprint(df.describe())`)}>describe</button>
                <button onClick={() => setEditorCode(prev => prev + `\nprint(df.head())`)}>head</button>
              </div>
            </div>
          </div>
        )}

        {/* Left - R */}
        {showR && (
          <div className="side-panel left-panel r-panel">
            <div className="panel-header"><BarChart2 size={14} /><span>R</span>{rReady && <span className="ready-dot">●</span>}</div>
            <div className="panel-content">
              <div className="r-section">
                <h4>Stats</h4>
                <button className="r-btn" onClick={() => runRCode('x <- c(1,2,3,4,5); mean(x)')}>mean()</button>
                <button className="r-btn" onClick={() => runRCode('x <- c(1,2,3,4,5); sd(x)')}>sd()</button>
                <button className="r-btn" onClick={() => runRCode('x <- c(1,2,3,4,5); summary(x)')}>summary()</button>
              </div>
              <div className="r-section">
                <h4>Tests</h4>
                <button className="r-btn" onClick={() => runRCode('t.test(rnorm(10), rnorm(10))')}>t.test()</button>
                <button className="r-btn" onClick={() => runRCode('cor(c(1,2,3,4,5), c(2,4,5,4,5))')}>cor()</button>
              </div>
            </div>
          </div>
        )}

        {/* Center */}
        <div className="center-panel">
          <div className="editor-area">
            {/* Language Gutter + Editor */}
            <div className="editor-panel">
              <div className="panel-header">
                <span className="file-tab active">📄 script.py</span>
                <div className="lang-legend">
                  <span className="lang-tag py">py</span>
                  <span className="lang-tag r">R</span>
                </div>
              </div>
              <div className="editor-with-gutter">
                <div className="language-gutter">
                  {editorCode.split('\n').map((line, i) => {
                    const anno = langAnnotations.find(a => a.line === i + 1)
                    return (
                      <div key={i} className="gutter-line">
                        {anno && <span className={`gutter-lang ${anno.lang}`}>{anno.lang}</span>}
                      </div>
                    )
                  })}
                </div>
                <div className="editor-content">
                  <CodeEditor initialValue={editorCode} onChange={(val) => setEditorCode(val || '')} />
                </div>
              </div>
            </div>

            {/* Architecture */}
            {showArchitecture && (
              <div className="architecture-panel">
                <div className="panel-header"><Boxes size={14} /><span>PyTorch</span></div>
                <div className="architecture-content">
                  <div className="layer-palette">
                    <div className="layer-group">
                      <button className="layer-btn linear" onClick={() => addPyTorchLayer('self.fc = nn.Linear(in, out)')}>Linear</button>
                      <button className="layer-btn conv" onClick={() => addPyTorchLayer('self.conv = nn.Conv2d(in, out, 3)')}>Conv2d</button>
                      <button className="layer-btn pool" onClick={() => addPyTorchLayer('self.pool = nn.MaxPool2d(2)')}>MaxPool</button>
                      <button className="layer-btn rnn" onClick={() => addPyTorchLayer('self.lstm = nn.LSTM(in, hidden)')}>LSTM</button>
                      <button className="layer-btn act" onClick={() => addPyTorchLayer('x = torch.relu(x)')}>ReLU</button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Resizable Output */}
          <div className="resize-handle" ref={resizeRef} onMouseDown={handleMouseDown}>
            <GripHorizontal size={12} />
          </div>
          <div className="output-panel" style={{ height: outputHeight }}>
            <div className="panel-header">
              <span className="panel-tab">Output</span>
              <div className="panel-actions">
                <button onClick={handleCopyOutput} className="panel-action-btn">{copied ? <Check size={12} /> : <Copy size={12} />}</button>
                <button onClick={handleClear} className="panel-action-btn"><RotateCcw size={12} /></button>
              </div>
            </div>
            <div className="output-content">
              <pre className="output-text">{output || 'Click Run to execute'}</pre>
            </div>
          </div>
        </div>

        {/* Right - Database */}
        {showDatabase && (
          <div className="side-panel right-panel database-panel">
            <div className="panel-header"><Database size={14} /><span>Data</span><button className="icon-btn" onClick={loadDatasets}><RefreshCw size={11} /></button></div>
            <div className="panel-content">
              <div className="search-box"><Search size={12} /><input placeholder="Search..." value={datasetSearch} onChange={e => setDatasetSearch(e.target.value)} /></div>
              <div className="filter-tabs">
                {['all', 'kaggle', 'pytorch', 'tensorflow'].map(f => (
                  <button key={f} className={`filter-tab ${datasetFilter === f ? 'active' : ''}`} onClick={() => setDatasetFilter(f)}>{f}</button>
                ))}
              </div>
              <div className="dataset-list">
                {loadingDatasets ? <div className="loading"><Loader2 size={14} className="spin" /></div> :
                  filteredDatasets.slice(0, 15).map(d => (
                    <button key={d.id + d.source} className="dataset-btn" onClick={() => setEditorCode(generateDatasetCode(d))}>
                      <span className="ds-name">{d.name}</span>
                      <span className="ds-meta">{d.source}</span>
                    </button>
                  ))
                }
              </div>
            </div>
          </div>
        )}
      </div>

      <footer className="ide-footer">
        <span>Python {pyodideReady ? '✓' : '...'}</span>
        <span className="footer-separator">|</span>
        <span>R {rReady ? '✓' : '...'}</span>
      </footer>
    </div>
  )
}

export default App
