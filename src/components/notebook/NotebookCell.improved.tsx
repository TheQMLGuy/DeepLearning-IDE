// IMPROVED VERSION - React.memo for performance, better error handling
import React, { useState, useRef, useEffect } from 'react'
import { Play, Trash2, Plus, Loader2, ChevronDown, ChevronUp } from 'lucide-react'
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

// Memoized cell component for better performance
export const NotebookCell = React.memo(({
    cell,
    index,
    onUpdate,
    onRun,
    onDelete,
    onAddBelow,
    isActive,
    onFocus
}: NotebookCellProps) => {
    const [isHovered, setIsHovered] = useState(false)
    const [isOutputCollapsed, setIsOutputCollapsed] = useState(false)
    const editorRef = useRef<any>(null)
    const cellRef = useRef<HTMLDivElement>(null)

    const handleEditorMount = (editor: any) => {
        editorRef.current = editor

        // Auto-resize editor based on content
        const updateHeight = () => {
            const contentHeight = Math.min(600, Math.max(60, editor.getContentHeight()))
            editor.layout({ height: contentHeight, width: editor.getLayoutInfo().width })
        }

        editor.onDidContentSizeChange(updateHeight)
        updateHeight()

        // Focus on mount if active
        if (isActive) {
            setTimeout(() => editor.focus(), 100)
        }
    }

    // Auto-scroll into view when active
    useEffect(() => {
        if (isActive && cellRef.current) {
            cellRef.current.scrollIntoView({
                behavior: 'smooth',
                block: 'nearest'
            })
        }
    }, [isActive])

    // Handle keyboard shortcuts within cell
    const handleKeyDown = (e: React.KeyboardEvent) => {
        // Prevent parent handlers from interfering
        if (e.shiftKey && e.key === 'Enter') {
            e.stopPropagation()
        }
        if (e.ctrlKey && e.key === 'Enter') {
            e.stopPropagation()
        }
    }

    const handleRunClick = (e: React.MouseEvent) => {
        e.stopPropagation()
        if (!cell.isRunning) {
            onRun(cell.id)
        }
    }

    return (
        <div
            ref={cellRef}
            className={`notebook-cell ${isActive ? 'active' : ''} ${cell.isRunning ? 'running' : ''}`}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            onClick={() => onFocus(cell.id)}
            onKeyDown={handleKeyDown}
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
                        onClick={handleRunClick}
                        disabled={cell.isRunning}
                        title="Run cell (Shift+Enter)"
                        aria-label="Run cell"
                    >
                        {cell.isRunning ? (
                            <Loader2 size={14} className="spin" />
                        ) : (
                            <Play size={14} fill="currentColor" />
                        )}
                    </button>

                    <button
                        className="cell-btn"
                        onClick={(e) => {
                            e.stopPropagation()
                            onAddBelow(cell.id)
                        }}
                        title="Add cell below (Alt+Enter)"
                        aria-label="Add cell below"
                    >
                        <Plus size={14} />
                    </button>

                    <button
                        className="cell-btn delete"
                        onClick={(e) => {
                            e.stopPropagation()
                            if (confirm('Delete this cell?')) {
                                onDelete(cell.id)
                            }
                        }}
                        title="Delete cell"
                        aria-label="Delete cell"
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
                        scrollbar: {
                            vertical: 'auto',
                            horizontal: 'auto',
                            verticalScrollbarSize: 8,
                            horizontalScrollbarSize: 8
                        },
                        overviewRulerBorder: false,
                        renderLineHighlight: isActive ? 'all' : 'none',
                        automaticLayout: true,
                        quickSuggestions: true,
                        suggestOnTriggerCharacters: true,
                        acceptSuggestionOnEnter: 'on',
                        tabCompletion: 'on',
                        wordBasedSuggestions: 'currentDocument',
                    }}
                />
            </div>

            {/* Output */}
            {cell.output && (
                <div className="cell-output-container">
                    <div className="output-header">
                        <span className="output-label">Output</span>
                        <button
                            className="output-collapse-btn"
                            onClick={(e) => {
                                e.stopPropagation()
                                setIsOutputCollapsed(!isOutputCollapsed)
                            }}
                            title={isOutputCollapsed ? 'Expand output' : 'Collapse output'}
                        >
                            {isOutputCollapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
                        </button>
                    </div>
                    {!isOutputCollapsed && (
                        <div className="cell-output">
                            <pre>{cell.output}</pre>
                        </div>
                    )}
                </div>
            )}

            {/* Add cell hint on hover */}
            {isHovered && !cell.isRunning && (
                <div className="add-cell-hint">
                    <button
                        onClick={(e) => {
                            e.stopPropagation()
                            onAddBelow(cell.id)
                        }}
                    >
                        <Plus size={12} /> Add cell
                    </button>
                </div>
            )}

            {/* Execution indicator */}
            {cell.isRunning && (
                <div className="execution-indicator">
                    <div className="pulse-dot" />
                    Running...
                </div>
            )}
        </div>
    )
}, (prevProps, nextProps) => {
    // Custom comparison for better performance
    return (
        prevProps.cell.content === nextProps.cell.content &&
        prevProps.cell.output === nextProps.cell.output &&
        prevProps.cell.isRunning === nextProps.cell.isRunning &&
        prevProps.isActive === nextProps.isActive &&
        prevProps.index === nextProps.index
    )
})

NotebookCell.displayName = 'NotebookCell'

export default NotebookCell
