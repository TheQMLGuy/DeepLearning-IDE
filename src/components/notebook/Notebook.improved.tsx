// IMPROVED VERSION - Keyboard shortcuts, execution queue, better UX
import { useState, useCallback, useEffect, useRef } from 'react'
import { Plus, Save, FolderOpen, ExternalLink, LogIn, Play, Keyboard } from 'lucide-react'
import NotebookCell, { CellData } from './NotebookCell'
import './Notebook.css'

interface NotebookProps {
    onRunCell: (code: string, language: 'python' | 'r') => Promise<string>
    isReady: boolean
}

const generateId = () => Math.random().toString(36).substr(2, 9)

const defaultCells: CellData[] = [
    {
        id: generateId(),
        type: 'code',
        language: 'python',
        content: `# 🧠 Deep Learning IDE
# This is a Jupyter-style notebook with Python + R support
# Keyboard shortcuts: Shift+Enter to run, Ctrl+S to save

import torch
import torch.nn as nn

print("PyTorch version:", torch.__version__)`,
        output: '',
        isRunning: false,
        isCollapsed: false
    },
    {
        id: generateId(),
        type: 'code',
        language: 'python',
        content: `# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

model = SimpleNN()
print(model)`,
        output: '',
        isRunning: false,
        isCollapsed: false
    },
    {
        id: generateId(),
        type: 'code',
        language: 'r',
        content: `# R Statistical Analysis
data <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
cat("Mean:", mean(data), "\\n")
cat("SD:", sd(data), "\\n")
cat("Summary:\\n")
summary(data)`,
        output: '',
        isRunning: false,
        isCollapsed: false
    }
]

export function Notebook({ onRunCell, isReady }: NotebookProps) {
    const [cells, setCells] = useState<CellData[]>(defaultCells)
    const [activeCell, setActiveCell] = useState<string | null>(defaultCells[0]?.id || null)
    const [isSignedIn, setIsSignedIn] = useState(false)
    const [userName, setUserName] = useState('')
    const [showShortcuts, setShowShortcuts] = useState(false)
    const [executionQueue, setExecutionQueue] = useState<string[]>([])
    const isExecutingRef = useRef(false)

    // Load notebook from localStorage
    useEffect(() => {
        try {
            const saved = localStorage.getItem('notebook-cells')
            if (saved) {
                const parsed = JSON.parse(saved)
                if (Array.isArray(parsed) && parsed.length > 0) {
                    setCells(parsed)
                    setActiveCell(parsed[0]?.id || null)
                }
            }
        } catch (e) {
            console.error('Failed to load notebook:', e)
        }
    }, [])

    // Auto-save notebook
    useEffect(() => {
        const timer = setTimeout(() => {
            try {
                localStorage.setItem('notebook-cells', JSON.stringify(cells))
            } catch (e) {
                console.error('Failed to save notebook:', e)
            }
        }, 1000)
        return () => clearTimeout(timer)
    }, [cells])

    // Keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Shift+Enter: Run current cell
            if (e.shiftKey && e.key === 'Enter' && activeCell) {
                e.preventDefault()
                runCell(activeCell)
            }

            // Ctrl+Enter: Run current cell and stay
            if (e.ctrlKey && e.key === 'Enter' && activeCell) {
                e.preventDefault()
                runCell(activeCell)
            }

            // Alt+Enter: Run current cell and insert below
            if (e.altKey && e.key === 'Enter' && activeCell) {
                e.preventDefault()
                runCell(activeCell).then(() => addCellBelow(activeCell))
            }

            // Ctrl+S: Save notebook
            if (e.ctrlKey && e.key === 's') {
                e.preventDefault()
                saveNotebook()
            }

            // Esc: Deselect cell
            if (e.key === 'Escape') {
                setActiveCell(null)
            }

            // Arrow Up: Focus previous cell (when not in editor)
            if (e.key === 'ArrowUp' && activeCell && (e.target as HTMLElement).tagName !== 'TEXTAREA') {
                const idx = cells.findIndex(c => c.id === activeCell)
                if (idx > 0) {
                    e.preventDefault()
                    setActiveCell(cells[idx - 1].id)
                }
            }

            // Arrow Down: Focus next cell (when not in editor)
            if (e.key === 'ArrowDown' && activeCell && (e.target as HTMLElement).tagName !== 'TEXTAREA') {
                const idx = cells.findIndex(c => c.id === activeCell)
                if (idx < cells.length - 1) {
                    e.preventDefault()
                    setActiveCell(cells[idx + 1].id)
                }
            }

            // Ctrl+?: Show shortcuts
            if (e.ctrlKey && e.key === '?') {
                e.preventDefault()
                setShowShortcuts(true)
            }
        }

        document.addEventListener('keydown', handleKeyDown)
        return () => document.removeEventListener('keydown', handleKeyDown)
    }, [activeCell, cells])

    // Execute cells from queue
    useEffect(() => {
        const executeNext = async () => {
            if (isExecutingRef.current || executionQueue.length === 0) return
            
            isExecutingRef.current = true
            const cellId = executionQueue[0]
            
            try {
                await runCellInternal(cellId)
            } finally {
                setExecutionQueue(prev => prev.slice(1))
                isExecutingRef.current = false
            }
        }

        executeNext()
    }, [executionQueue])

    const updateCell = useCallback((id: string, content: string) => {
        setCells(prev => prev.map(cell =>
            cell.id === id ? { ...cell, content } : cell
        ))
    }, [])

    const runCellInternal = async (id: string) => {
        const cell = cells.find(c => c.id === id)
        if (!cell || !isReady) return

        setCells(prev => prev.map(c =>
            c.id === id ? { ...c, isRunning: true, output: '' } : c
        ))

        try {
            const output = await onRunCell(cell.content, cell.language)
            setCells(prev => prev.map(c =>
                c.id === id ? { ...c, isRunning: false, output } : c
            ))
        } catch (error: any) {
            setCells(prev => prev.map(c =>
                c.id === id ? { ...c, isRunning: false, output: `Error: ${error.message}` } : c
            ))
        }
    }

    const runCell = useCallback((id: string) => {
        setExecutionQueue(prev => [...prev, id])
        return Promise.resolve()
    }, [])

    const deleteCell = useCallback((id: string) => {
        if (cells.length <= 1) return
        setCells(prev => {
            const filtered = prev.filter(c => c.id !== id)
            if (activeCell === id && filtered.length > 0) {
                setActiveCell(filtered[0].id)
            }
            return filtered
        })
    }, [cells.length, activeCell])

    const addCellBelow = useCallback((id: string) => {
        const index = cells.findIndex(c => c.id === id)
        const newCell: CellData = {
            id: generateId(),
            type: 'code',
            language: 'python',
            content: '',
            output: '',
            isRunning: false,
            isCollapsed: false
        }
        const newCells = [...cells]
        newCells.splice(index + 1, 0, newCell)
        setCells(newCells)
        setActiveCell(newCell.id)
    }, [cells])

    const addCell = (language: 'python' | 'r' = 'python') => {
        const newCell: CellData = {
            id: generateId(),
            type: 'code',
            language,
            content: language === 'python' ? '# Python code' : '# R code',
            output: '',
            isRunning: false,
            isCollapsed: false
        }
        setCells(prev => [...prev, newCell])
        setActiveCell(newCell.id)
    }

    const runAllCells = async () => {
        for (const cell of cells) {
            setExecutionQueue(prev => [...prev, cell.id])
        }
    }

    const clearAllOutputs = () => {
        setCells(prev => prev.map(cell => ({ ...cell, output: '' })))
    }

    const saveNotebook = () => {
        try {
            localStorage.setItem('notebook-cells', JSON.stringify(cells))
            // Visual feedback
            const toolbar = document.querySelector('.notebook-toolbar')
            if (toolbar) {
                toolbar.classList.add('saved-feedback')
                setTimeout(() => toolbar.classList.remove('saved-feedback'), 1000)
            }
        } catch (e) {
            console.error('Failed to save:', e)
            alert('Failed to save notebook')
        }
    }

    const exportToIpynb = () => {
        const notebook = {
            nbformat: 4,
            nbformat_minor: 5,
            metadata: {
                kernelspec: { name: 'python3', display_name: 'Python 3' }
            },
            cells: cells.map(cell => ({
                cell_type: 'code',
                source: cell.content.split('\n'),
                metadata: { language: cell.language },
                outputs: cell.output ? [{ output_type: 'stream', text: cell.output.split('\n') }] : []
            }))
        }

        const blob = new Blob([JSON.stringify(notebook, null, 2)], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `notebook-${Date.now()}.ipynb`
        a.click()
        URL.revokeObjectURL(url)
    }

    const openInColab = () => {
        alert('To open in Colab:\n1. Save notebook with the button\n2. Upload to Google Drive\n3. Open with Colaboratory')
    }

    const handleGoogleSignIn = () => {
        setIsSignedIn(true)
        setUserName('User')
    }

    return (
        <div className="notebook">
            {/* Keyboard Shortcuts Modal */}
            {showShortcuts && (
                <div 
                    className="modal-overlay" 
                    onClick={() => setShowShortcuts(false)}
                    style={{
                        position: 'fixed',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        background: 'rgba(0,0,0,0.7)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        zIndex: 1000
                    }}
                >
                    <div 
                        className="shortcuts-modal"
                        onClick={e => e.stopPropagation()}
                        style={{
                            background: '#1a1d2e',
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: '12px',
                            padding: '24px',
                            maxWidth: '500px',
                            color: '#e0e0e0'
                        }}
                    >
                        <h3 style={{ marginTop: 0 }}>Keyboard Shortcuts</h3>
                        <div style={{ display: 'grid', gap: '8px' }}>
                            <div><kbd>Shift+Enter</kbd> Run cell and select below</div>
                            <div><kbd>Ctrl+Enter</kbd> Run cell</div>
                            <div><kbd>Alt+Enter</kbd> Run cell and insert below</div>
                            <div><kbd>Ctrl+S</kbd> Save notebook</div>
                            <div><kbd>Esc</kbd> Deselect cell</div>
                            <div><kbd>↑/↓</kbd> Navigate cells</div>
                        </div>
                        <button 
                            onClick={() => setShowShortcuts(false)}
                            style={{
                                marginTop: '16px',
                                padding: '8px 16px',
                                background: '#3b82f6',
                                border: 'none',
                                borderRadius: '6px',
                                color: 'white',
                                cursor: 'pointer'
                            }}
                        >
                            Close
                        </button>
                    </div>
                </div>
            )}

            {/* Notebook Toolbar */}
            <div className="notebook-toolbar">
                <div className="toolbar-left">
                    <button className="toolbar-btn" onClick={() => addCell('python')}>
                        <Plus size={14} /> Python
                    </button>
                    <button className="toolbar-btn" onClick={() => addCell('r')}>
                        <Plus size={14} /> R
                    </button>
                    <span className="toolbar-separator" />
                    <button 
                        className="toolbar-btn primary" 
                        onClick={runAllCells} 
                        disabled={!isReady || executionQueue.length > 0}
                        title="Run all cells"
                    >
                        <Play size={14} /> Run All
                    </button>
                    <button 
                        className="toolbar-btn" 
                        onClick={clearAllOutputs}
                        title="Clear all outputs"
                    >
                        Clear
                    </button>
                </div>

                <div className="toolbar-right">
                    <button 
                        className="toolbar-btn" 
                        onClick={() => setShowShortcuts(true)}
                        title="Show keyboard shortcuts (Ctrl+?)"
                    >
                        <Keyboard size={14} />
                    </button>
                    <button className="toolbar-btn" onClick={saveNotebook} title="Save notebook (Ctrl+S)">
                        <Save size={14} /> Save
                    </button>
                    <button className="toolbar-btn" onClick={exportToIpynb}>
                        <FolderOpen size={14} /> Export
                    </button>
                    <button className="toolbar-btn" onClick={openInColab}>
                        <ExternalLink size={14} /> Colab
                    </button>
                    <span className="toolbar-separator" />
                    {isSignedIn ? (
                        <span className="user-badge">👤 {userName}</span>
                    ) : (
                        <button className="toolbar-btn google" onClick={handleGoogleSignIn}>
                            <LogIn size={14} /> Sign In
                        </button>
                    )}
                </div>
            </div>

            {/* Execution Queue Indicator */}
            {executionQueue.length > 0 && (
                <div style={{
                    padding: '8px 16px',
                    background: 'rgba(59, 130, 246, 0.1)',
                    borderBottom: '1px solid rgba(59, 130, 246, 0.3)',
                    color: '#3b82f6',
                    fontSize: '13px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                }}>
                    <div className="spinner" style={{ 
                        width: '12px', 
                        height: '12px',
                        border: '2px solid rgba(59, 130, 246, 0.3)',
                        borderTopColor: '#3b82f6',
                        borderRadius: '50%',
                        animation: 'spin 1s linear infinite'
                    }} />
                    {executionQueue.length} cell(s) in queue
                </div>
            )}

            {/* Cells */}
            <div className="notebook-cells">
                {cells.map((cell, index) => (
                    <NotebookCell
                        key={cell.id}
                        cell={cell}
                        index={index}
                        onUpdate={updateCell}
                        onRun={runCell}
                        onDelete={deleteCell}
                        onAddBelow={addCellBelow}
                        isActive={activeCell === cell.id}
                        onFocus={setActiveCell}
                    />
                ))}

                {/* Add cell at bottom */}
                <div className="add-cell-bottom">
                    <button onClick={() => addCell('python')}>+ Python</button>
                    <button onClick={() => addCell('r')}>+ R</button>
                </div>
            </div>
        </div>
    )
}

export default Notebook
