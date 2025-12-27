import { useState, useEffect, useCallback } from 'react'
import { Database, BarChart2, Table2, Boxes, Search, RefreshCw, Loader2, Brain } from 'lucide-react'
import Notebook from './components/notebook/Notebook'
import './App.css'

interface Dataset {
  id: string
  name: string
  category?: string
  size?: string
  source: string
}

const sampleDataFrame = [
  { id: 1, name: 'Alice', age: 28, salary: 55000, dept: 'Engineering' },
  { id: 2, name: 'Bob', age: 34, salary: 62000, dept: 'Marketing' },
  { id: 3, name: 'Charlie', age: 25, salary: 48000, dept: 'Engineering' },
  { id: 4, name: 'Diana', age: 31, salary: 71000, dept: 'Sales' },
  { id: 5, name: 'Eve', age: 29, salary: 58000, dept: 'Engineering' },
]

function App() {
  const [pyodideReady, setPyodideReady] = useState(false)
  const [rReady, setRReady] = useState(false)

  // Panel states
  const [showArchitecture, setShowArchitecture] = useState(false)
  const [showDatabase, setShowDatabase] = useState(false)
  const [showPandas, setShowPandas] = useState(false)
  const [showR, setShowR] = useState(false)

  // Dataset state
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [datasetSearch, setDatasetSearch] = useState('')
  const [datasetFilter, setDatasetFilter] = useState<string>('all')
  const [loadingDatasets, setLoadingDatasets] = useState(false)

  // DataFrame state
  const [dataFrame] = useState(sampleDataFrame)

  // Output for side panel R execution
  const [rOutput, setROutput] = useState('')

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

  useEffect(() => {
    loadDatasets()

    // Load Python
    if (window.pyodideReady) {
      window.pyodideReady.then(() => setPyodideReady(true)).catch(console.error)
    }

    // Auto-load R
    if ((window as any).loadWebR) {
      (window as any).loadWebR().then(() => setRReady(true)).catch(console.error)
    }

    window.terminal.init()
  }, [])

  // Run code from notebook
  const runCellCode = useCallback(async (code: string, language: 'python' | 'r'): Promise<string> => {
    return new Promise((resolve) => {
      let output = ''

      const cleanup = window.terminal.onData((data: string) => {
        output += data.replace(/\r\n/g, '\n').replace(/\r/g, '\n')
      })

      const execute = async () => {
        if (language === 'python') {
          window.terminal.send(`${code}\n`)
          await new Promise(r => setTimeout(r, 500))
        } else {
          if ((window as any).runR) {
            try {
              const result = await (window as any).runR(code)
              output = result || ''
            } catch (e: any) {
              output = `R Error: ${e.message}`
            }
          }
        }
        cleanup()
        resolve(output.trim())
      }

      execute()
    })
  }, [])

  const runRCode = async (code: string) => {
    if (!(window as any).runR) return
    setROutput('')
    try {
      const result = await (window as any).runR(code)
      setROutput(result || '')
    } catch (e: any) {
      setROutput(`Error: ${e.message}`)
    }
  }

  const addPyTorchLayer = (code: string) => {
    // TODO: Add to current cell
    console.log('Add layer:', code)
  }

  const filteredDatasets = datasets.filter(d => {
    const matchSearch = d.name.toLowerCase().includes(datasetSearch.toLowerCase())
    const matchFilter = datasetFilter === 'all' || d.source === datasetFilter
    return matchSearch && matchFilter
  })

  return (
    <div className="ide-container">
      {/* Header */}
      <header className="ide-header">
        <div className="header-left">
          <Brain size={20} className="logo-icon" />
          <span className="app-title">Deep Learning IDE</span>
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
          <div className="status-indicators">
            <span className={`status-dot ${pyodideReady ? 'ready' : 'loading'}`}>Py</span>
            <span className={`status-dot ${rReady ? 'ready' : 'loading'}`}>R</span>
          </div>
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
                        {Object.values(row).map((val, i) => (
                          <td key={i}>{typeof val === 'number' ? val.toLocaleString() : val}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Left - R */}
        {showR && (
          <div className="side-panel left-panel r-panel">
            <div className="panel-header"><BarChart2 size={14} /><span>R Console</span>{rReady && <span className="ready-dot">●</span>}</div>
            <div className="panel-content">
              <div className="r-section">
                <h4>Quick Stats</h4>
                <button className="r-btn" onClick={() => runRCode('x <- c(1,2,3,4,5); mean(x)')}>mean()</button>
                <button className="r-btn" onClick={() => runRCode('x <- c(1,2,3,4,5); sd(x)')}>sd()</button>
                <button className="r-btn" onClick={() => runRCode('x <- c(1,2,3,4,5); summary(x)')}>summary()</button>
              </div>
              {rOutput && (
                <div className="r-output">
                  <pre>{rOutput}</pre>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Center - Notebook */}
        <div className="center-panel">
          {/* Architecture Panel */}
          {showArchitecture && (
            <div className="architecture-panel">
              <div className="panel-header"><Boxes size={14} /><span>PyTorch Layers</span></div>
              <div className="architecture-content">
                <div className="layer-palette">
                  <button className="layer-btn linear" onClick={() => addPyTorchLayer('nn.Linear(in, out)')}>Linear</button>
                  <button className="layer-btn conv" onClick={() => addPyTorchLayer('nn.Conv2d(in, out, 3)')}>Conv2d</button>
                  <button className="layer-btn pool" onClick={() => addPyTorchLayer('nn.MaxPool2d(2)')}>MaxPool</button>
                  <button className="layer-btn norm" onClick={() => addPyTorchLayer('nn.BatchNorm2d(ch)')}>BatchNorm</button>
                  <button className="layer-btn rnn" onClick={() => addPyTorchLayer('nn.LSTM(in, hidden)')}>LSTM</button>
                  <button className="layer-btn act" onClick={() => addPyTorchLayer('nn.ReLU()')}>ReLU</button>
                  <button className="layer-btn act" onClick={() => addPyTorchLayer('nn.Dropout(0.5)')}>Dropout</button>
                </div>
              </div>
            </div>
          )}

          {/* Main Notebook */}
          <Notebook onRunCell={runCellCode} isReady={pyodideReady} />
        </div>

        {/* Right - Database */}
        {showDatabase && (
          <div className="side-panel right-panel database-panel">
            <div className="panel-header"><Database size={14} /><span>Datasets</span><button className="icon-btn" onClick={loadDatasets}><RefreshCw size={11} /></button></div>
            <div className="panel-content">
              <div className="search-box"><Search size={12} /><input placeholder="Search..." value={datasetSearch} onChange={e => setDatasetSearch(e.target.value)} /></div>
              <div className="filter-tabs">
                {['all', 'kaggle', 'pytorch', 'tensorflow', 'huggingface'].map(f => (
                  <button key={f} className={`filter-tab ${datasetFilter === f ? 'active' : ''}`} onClick={() => setDatasetFilter(f)}>{f}</button>
                ))}
              </div>
              <div className="dataset-list">
                {loadingDatasets ? <div className="loading"><Loader2 size={14} className="spin" /></div> :
                  filteredDatasets.slice(0, 20).map(d => (
                    <button key={d.id + d.source} className="dataset-btn">
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
        <span>🧠 Deep Learning IDE</span>
        <span className="footer-separator">|</span>
        <span>Python {pyodideReady ? '✓' : '...'}</span>
        <span className="footer-separator">|</span>
        <span>R {rReady ? '✓' : '...'}</span>
        <span className="footer-right">PyTorch + WebR</span>
      </footer>
    </div>
  )
}

export default App
