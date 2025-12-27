import { useState, useCallback } from 'react'
import { Plus, Save, FolderOpen, ExternalLink, LogIn } from 'lucide-react'
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

    const updateCell = useCallback((id: string, content: string) => {
        setCells(prev => prev.map(cell =>
            cell.id === id ? { ...cell, content } : cell
        ))
    }, [])

    const runCell = useCallback(async (id: string) => {
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
    }, [cells, onRunCell, isReady])

    const deleteCell = useCallback((id: string) => {
        if (cells.length <= 1) return
        setCells(prev => prev.filter(c => c.id !== id))
    }, [cells.length])

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
            await runCell(cell.id)
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
        a.download = 'notebook.ipynb'
        a.click()
        URL.revokeObjectURL(url)
    }

    const openInColab = () => {
        // Create a data URL with notebook content (for demo)
        alert('To open in Colab:\n1. Save notebook with the button\n2. Upload to Google Drive\n3. Open with Colaboratory')
    }

    const handleGoogleSignIn = () => {
        // Placeholder for Google OAuth
        setIsSignedIn(true)
        setUserName('User')
    }

    return (
        <div className="notebook">
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
                    <button className="toolbar-btn primary" onClick={runAllCells} disabled={!isReady}>
                        ▶ Run All
                    </button>
                </div>

                <div className="toolbar-right">
                    <button className="toolbar-btn" onClick={exportToIpynb}>
                        <Save size={14} /> Save .ipynb
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
