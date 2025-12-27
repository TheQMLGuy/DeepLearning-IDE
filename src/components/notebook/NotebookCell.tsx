import { useState, useRef } from 'react'
import { Play, Trash2, Plus, Loader2 } from 'lucide-react'
import Editor from '@monaco-editor/react'
import './NotebookCell.css'

export interface CellData {
    id: string
    type: 'code' | 'markdown'
    language: 'python' | 'r'
    content: string
    output: string
    isRunning: boolean
    isCollapsed: boolean
}

interface NotebookCellProps {
    cell: CellData
    index: number
    onUpdate: (id: string, content: string) => void
    onRun: (id: string) => void
    onDelete: (id: string) => void
    onAddBelow: (id: string) => void
    isActive: boolean
    onFocus: (id: string) => void
}

export function NotebookCell({
    cell,
    index,
    onUpdate,
    onRun,
    onDelete,
    onAddBelow,
    isActive,
    onFocus
}: NotebookCellProps) {
    const [isHovered, setIsHovered] = useState(false)
    const editorRef = useRef<any>(null)

    const handleEditorMount = (editor: any) => {
        editorRef.current = editor
        // Auto-resize editor based on content
        editor.onDidContentSizeChange(() => {
            const contentHeight = Math.min(400, Math.max(60, editor.getContentHeight()))
            editor.layout({ height: contentHeight, width: editor.getLayoutInfo().width })
        })
    }

    return (
        <div
            className={`notebook-cell ${isActive ? 'active' : ''} ${cell.isRunning ? 'running' : ''}`}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            onClick={() => onFocus(cell.id)}
        >
            {/* Cell Toolbar */}
            <div className="cell-toolbar">
                <div className="cell-info">
                    <span className="cell-index">[{index + 1}]</span>
                    <span className={`cell-lang ${cell.language}`}>{cell.language}</span>
                </div>

                <div className="cell-actions">
                    <button
                        className="cell-btn run"
                        onClick={(e) => { e.stopPropagation(); onRun(cell.id) }}
                        disabled={cell.isRunning}
                        title="Run cell (Shift+Enter)"
                    >
                        {cell.isRunning ? <Loader2 size={14} className="spin" /> : <Play size={14} fill="currentColor" />}
                    </button>

                    <button
                        className="cell-btn"
                        onClick={(e) => { e.stopPropagation(); onAddBelow(cell.id) }}
                        title="Add cell below"
                    >
                        <Plus size={14} />
                    </button>

                    <button
                        className="cell-btn delete"
                        onClick={(e) => { e.stopPropagation(); onDelete(cell.id) }}
                        title="Delete cell"
                    >
                        <Trash2 size={14} />
                    </button>
                </div>
            </div>

            {/* Code Editor */}
            <div className="cell-editor">
                <Editor
                    height="auto"
                    defaultLanguage={cell.language === 'r' ? 'r' : 'python'}
                    value={cell.content}
                    onChange={(value) => onUpdate(cell.id, value || '')}
                    onMount={handleEditorMount}
                    theme="vs-dark"
                    options={{
                        minimap: { enabled: false },
                        scrollBeyondLastLine: false,
                        lineNumbers: 'on',
                        lineNumbersMinChars: 3,
                        folding: false,
                        fontSize: 13,
                        fontFamily: "'JetBrains Mono', 'SF Mono', Consolas, monospace",
                        padding: { top: 8, bottom: 8 },
                        scrollbar: { vertical: 'hidden', horizontal: 'auto' },
                        overviewRulerBorder: false,
                        renderLineHighlight: 'none',
                        automaticLayout: true,
                    }}
                />
            </div>

            {/* Output */}
            {cell.output && (
                <div className="cell-output">
                    <pre>{cell.output}</pre>
                </div>
            )}

            {/* Add cell hint on hover */}
            {isHovered && (
                <div className="add-cell-hint">
                    <button onClick={() => onAddBelow(cell.id)}>
                        <Plus size={12} /> Add cell
                    </button>
                </div>
            )}
        </div>
    )
}

export default NotebookCell
