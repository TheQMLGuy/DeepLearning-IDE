import { useState, useEffect } from 'react'
import { Play, RotateCcw, Activity, Loader2, Copy, Check, Database, BarChart2, Code2, Boxes } from 'lucide-react'
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

  // Panel visibility states
  const [showArchitecture, setShowArchitecture] = useState(false)
  const [showDatabase, setShowDatabase] = useState(false)
  const [showAnalysis, setShowAnalysis] = useState(false) // Pandas/R combined

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
      // Clean up and ensure proper newlines
      const cleanData = data
        .replace(/\r\n/g, '\n')
        .replace(/\r/g, '\n')
        .replace(/\$ $/, '')
        .replace(/^🐍 Running.*\n\n/, '')

      if (cleanData.trim() || cleanData.includes('\n')) {
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

    await window.fileSystem.saveFile('script.py', editorCode)
    window.terminal.send(`python "script.py"\n`)

    setTimeout(() => setIsRunning(false), 500)
  }

  const handleClear = () => setOutput('')

  const handleCopyOutput = () => {
    navigator.clipboard.writeText(output)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="ide-container">
      {/* Header */}
      <header className="ide-header">
        <div className="header-left">
          <Activity size={20} color="#3b82f6" />
          <span className="app-title">AI IDE</span>
          <span className="web-badge">Web</span>
        </div>

        <div className="header-center">
          {/* Toggle Buttons */}
          <button
            className={`toggle-btn ${showArchitecture ? 'active' : ''}`}
            onClick={() => setShowArchitecture(!showArchitecture)}
            title="AI Architecture - Visual neural network editor"
          >
            <Boxes size={16} />
            <span>Architecture</span>
          </button>

          <button
            className={`toggle-btn ${showDatabase ? 'active' : ''}`}
            onClick={() => setShowDatabase(!showDatabase)}
            title="Dataset Browser - Kaggle, TensorFlow, PyTorch"
          >
            <Database size={16} />
            <span>Database</span>
          </button>

          <button
            className={`toggle-btn ${showAnalysis ? 'active' : ''}`}
            onClick={() => setShowAnalysis(!showAnalysis)}
            title="Data Analysis - Pandas & R"
          >
            <BarChart2 size={16} />
            <span>Analysis</span>
          </button>
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
            {isRunning ? <Loader2 size={14} className="spin" /> : <Play size={14} fill="white" />}
            <span>Run</span>
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="ide-main">
        {/* Left Panel - Analysis (Pandas/R) */}
        {showAnalysis && (
          <div className="side-panel left-panel">
            <div className="panel-header">
              <BarChart2 size={14} />
              <span>Data Analysis</span>
            </div>
            <div className="panel-content">
              <div className="analysis-section">
                <h4>🐼 Pandas</h4>
                <div className="analysis-tools">
                  <button className="tool-btn" onClick={() => setEditorCode(prev => prev + "\nimport pandas as pd\ndf = pd.DataFrame()")}>Import Pandas</button>
                  <button className="tool-btn" onClick={() => setEditorCode(prev => prev + "\nprint(df.describe())")}>Describe</button>
                  <button className="tool-btn" onClick={() => setEditorCode(prev => prev + "\nprint(df.info())")}>Info</button>
                  <button className="tool-btn" onClick={() => setEditorCode(prev => prev + "\nprint(df.head())")}>Head</button>
                </div>
              </div>
              <div className="analysis-section">
                <h4>📊 Statistics</h4>
                <div className="analysis-tools">
                  <button className="tool-btn" onClick={() => setEditorCode(prev => prev + "\nprint(df.mean())")}>Mean</button>
                  <button className="tool-btn" onClick={() => setEditorCode(prev => prev + "\nprint(df.corr())")}>Correlation</button>
                  <button className="tool-btn" onClick={() => setEditorCode(prev => prev + "\nprint(df.value_counts())")}>Value Counts</button>
                </div>
              </div>
              <div className="analysis-section">
                <h4>📈 R-Style Analysis</h4>
                <div className="analysis-tools">
                  <button className="tool-btn" onClick={() => setEditorCode(prev => prev + "\n# Summary statistics\nfor col in df.columns:\n    print(f'{col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}')")}>Summary</button>
                  <button className="tool-btn" onClick={() => setEditorCode(prev => prev + "\n# Linear model\nfrom scipy import stats\nslope, intercept, r, p, se = stats.linregress(x, y)")}>Linear Model</button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Center - Editor + Output + Architecture */}
        <div className="center-panel">
          <div className="editor-area">
            {/* Editor */}
            <div className="editor-panel">
              <div className="panel-header">
                <span className="file-tab active">
                  <Code2 size={14} />
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

            {/* Architecture Panel (below editor) */}
            {showArchitecture && (
              <div className="architecture-panel">
                <div className="panel-header">
                  <Boxes size={14} />
                  <span>Neural Network Architecture</span>
                  <span className="panel-hint">Click layers to add to code</span>
                </div>
                <div className="architecture-content">
                  <div className="layer-palette">
                    <div className="layer-group">
                      <h5>Input</h5>
                      <button className="layer-btn input" onClick={() => setEditorCode(prev => prev + "\n# Input Layer\nmodel.add(Input(shape=(784,)))")}>Input</button>
                    </div>
                    <div className="layer-group">
                      <h5>Dense</h5>
                      <button className="layer-btn dense" onClick={() => setEditorCode(prev => prev + "\nmodel.add(Dense(128, activation='relu'))")}>Dense 128</button>
                      <button className="layer-btn dense" onClick={() => setEditorCode(prev => prev + "\nmodel.add(Dense(64, activation='relu'))")}>Dense 64</button>
                      <button className="layer-btn dense" onClick={() => setEditorCode(prev => prev + "\nmodel.add(Dense(32, activation='relu'))")}>Dense 32</button>
                    </div>
                    <div className="layer-group">
                      <h5>Conv2D</h5>
                      <button className="layer-btn conv" onClick={() => setEditorCode(prev => prev + "\nmodel.add(Conv2D(32, (3,3), activation='relu'))")}>Conv 32</button>
                      <button className="layer-btn conv" onClick={() => setEditorCode(prev => prev + "\nmodel.add(Conv2D(64, (3,3), activation='relu'))")}>Conv 64</button>
                    </div>
                    <div className="layer-group">
                      <h5>Pooling</h5>
                      <button className="layer-btn pool" onClick={() => setEditorCode(prev => prev + "\nmodel.add(MaxPooling2D((2,2)))")}>MaxPool</button>
                      <button className="layer-btn pool" onClick={() => setEditorCode(prev => prev + "\nmodel.add(Flatten())")}>Flatten</button>
                    </div>
                    <div className="layer-group">
                      <h5>Regularization</h5>
                      <button className="layer-btn reg" onClick={() => setEditorCode(prev => prev + "\nmodel.add(Dropout(0.5))")}>Dropout</button>
                      <button className="layer-btn reg" onClick={() => setEditorCode(prev => prev + "\nmodel.add(BatchNormalization())")}>BatchNorm</button>
                    </div>
                    <div className="layer-group">
                      <h5>Output</h5>
                      <button className="layer-btn output" onClick={() => setEditorCode(prev => prev + "\nmodel.add(Dense(10, activation='softmax'))")}>Softmax 10</button>
                      <button className="layer-btn output" onClick={() => setEditorCode(prev => prev + "\nmodel.add(Dense(1, activation='sigmoid'))")}>Sigmoid</button>
                    </div>
                  </div>
                  <div className="architecture-preview">
                    <div className="arch-hint">
                      <Boxes size={24} />
                      <p>Visual architecture preview coming soon!</p>
                      <small>Click layers to add them to your code</small>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Output Panel */}
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

        {/* Right Panel - Database */}
        {showDatabase && (
          <div className="side-panel right-panel">
            <div className="panel-header">
              <Database size={14} />
              <span>Dataset Browser</span>
            </div>
            <div className="panel-content">
              <div className="db-section">
                <h4>🏆 Kaggle</h4>
                <div className="dataset-list">
                  <button className="dataset-btn" onClick={() => setEditorCode(`# Titanic Dataset\nimport pandas as pd\n\n# Sample Titanic data\ndata = {\n    'PassengerId': [1, 2, 3, 4, 5],\n    'Survived': [0, 1, 1, 1, 0],\n    'Pclass': [3, 1, 3, 1, 3],\n    'Name': ['Braund', 'Cumings', 'Heikkinen', 'Futrelle', 'Allen'],\n    'Age': [22, 38, 26, 35, 35],\n    'Fare': [7.25, 71.28, 7.92, 53.10, 8.05]\n}\ndf = pd.DataFrame(data)\nprint(df)\nprint(f"\\nSurvival rate: {df['Survived'].mean()*100:.1f}%")`)}>Titanic</button>
                  <button className="dataset-btn" onClick={() => setEditorCode(`# Iris Dataset\nfrom sklearn.datasets import load_iris\nimport pandas as pd\n\n# Using sample data\ndata = {\n    'sepal_length': [5.1, 4.9, 4.7, 5.0, 5.4],\n    'sepal_width': [3.5, 3.0, 3.2, 3.6, 3.9],\n    'petal_length': [1.4, 1.4, 1.3, 1.4, 1.7],\n    'species': ['setosa', 'setosa', 'setosa', 'setosa', 'setosa']\n}\ndf = pd.DataFrame(data)\nprint(df)\nprint(f"\\nMean sepal length: {df['sepal_length'].mean():.2f}")`)}>Iris</button>
                  <button className="dataset-btn" onClick={() => setEditorCode(`# MNIST (sample)\nprint("MNIST Dataset")\nprint("=" * 40)\nprint("28x28 grayscale images of handwritten digits")\nprint("60,000 training samples")\nprint("10,000 test samples")\nprint("10 classes (0-9)")`)}>MNIST</button>
                </div>
              </div>
              <div className="db-section">
                <h4>🔥 TensorFlow</h4>
                <div className="dataset-list">
                  <button className="dataset-btn" onClick={() => setEditorCode(`# Fashion MNIST\nprint("Fashion MNIST Dataset")\nprint("=" * 40)\nprint("28x28 grayscale images")\nprint("Classes: T-shirt, Trouser, Pullover, Dress,")\nprint("         Coat, Sandal, Shirt, Sneaker, Bag, Boot")`)}>Fashion MNIST</button>
                  <button className="dataset-btn" onClick={() => setEditorCode(`# CIFAR-10\nprint("CIFAR-10 Dataset")\nprint("=" * 40)\nprint("32x32 color images in 10 classes")\nprint("Classes: airplane, automobile, bird, cat, deer,")\nprint("         dog, frog, horse, ship, truck")`)}>CIFAR-10</button>
                </div>
              </div>
              <div className="db-section">
                <h4>⚡ PyTorch</h4>
                <div className="dataset-list">
                  <button className="dataset-btn" onClick={() => setEditorCode(`# ImageNet (info)\nprint("ImageNet Dataset")\nprint("=" * 40)\nprint("1000 classes")\nprint("1.2M training images")\nprint("50K validation images")`)}>ImageNet</button>
                  <button className="dataset-btn" onClick={() => setEditorCode(`# COCO (info)\nprint("COCO Dataset")\nprint("=" * 40)\nprint("Object detection & segmentation")\nprint("80 object categories")\nprint("330K images")`)}>COCO</button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Status Bar */}
      <footer className="ide-footer">
        <span>Python 3.11 (Pyodide)</span>
        <span className="footer-separator">|</span>
        <span>UTF-8</span>
        <div className="footer-right">
          <span style={{ color: pyodideReady ? '#4ade80' : '#fbbf24' }}>
            {pyodideReady ? '● Ready' : '○ Loading...'}
          </span>
        </div>
      </footer>
    </div>
  )
}

export default App
