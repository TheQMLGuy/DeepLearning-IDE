import React, { useState, useEffect } from 'react'
import { Folder, File, FolderPlus, RefreshCw } from 'lucide-react'

interface FileEntry {
    name: string
    isDirectory: boolean
    path: string
}

interface FileExplorerProps {
    onFileSelect: (path: string, content: string) => void
}

const FileExplorer: React.FC<FileExplorerProps> = ({ onFileSelect }) => {
    const [rootPath, setRootPath] = useState<string | null>(null)
    const [files, setFiles] = useState<FileEntry[]>([])
    const [selectedFile, setSelectedFile] = useState<string | null>(null)

    // Auto-load project on mount
    useEffect(() => {
        handleOpenFolder()
    }, [])

    const handleOpenFolder = async () => {
        const path = await window.fileSystem.selectDirectory()
        if (path) {
            setRootPath(path)
            loadDir(path)
        }
    }

    const loadDir = async (path: string) => {
        try {
            const entries = await window.fileSystem.readDir(path)
            setFiles(entries.sort((a, b) => {
                if (a.isDirectory === b.isDirectory) return a.name.localeCompare(b.name)
                return a.isDirectory ? -1 : 1
            }))
        } catch (e) {
            console.error("Error reading dir", e)
        }
    }

    const handleFileClick = async (entry: FileEntry) => {
        if (entry.isDirectory) {
            // Toggle directory - could expand/collapse in future
        } else {
            setSelectedFile(entry.path)
            const content = await window.fileSystem.readFile(entry.path)
            onFileSelect(entry.path, content)
        }
    }

    const getFileIcon = (name: string) => {
        if (name.endsWith('.py')) return '🐍'
        if (name.endsWith('.md')) return '📄'
        if (name.endsWith('.json')) return '📋'
        if (name.endsWith('.txt')) return '📝'
        return null
    }

    return (
        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: 'var(--bg-secondary)', borderRight: '1px solid var(--border-color)' }}>
            <div style={{ padding: '10px', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>EXPLORER</span>
                <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                        onClick={() => rootPath && loadDir(rootPath)}
                        title="Refresh"
                        style={{ color: 'var(--text-secondary)' }}
                    >
                        <RefreshCw size={14} />
                    </button>
                    <button
                        onClick={handleOpenFolder}
                        title="Open Folder"
                        style={{ color: 'var(--text-secondary)' }}
                    >
                        <FolderPlus size={16} />
                    </button>
                </div>
            </div>

            <div style={{ flex: 1, overflowY: 'auto', padding: '10px 0' }}>
                {!rootPath && (
                    <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                        <p style={{ marginBottom: '10px' }}>Loading project...</p>
                    </div>
                )}

                {rootPath && (
                    <div style={{
                        padding: '4px 12px',
                        fontSize: '0.75rem',
                        color: 'var(--accent-primary)',
                        fontWeight: 500,
                        marginBottom: '4px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px'
                    }}>
                        <Folder size={12} />
                        {rootPath.split('/').pop()}
                    </div>
                )}

                {files.map(file => (
                    <div
                        key={file.path}
                        onClick={() => handleFileClick(file)}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px',
                            padding: '4px 16px',
                            cursor: 'pointer',
                            fontSize: '0.85rem',
                            color: selectedFile === file.path ? 'var(--text-primary)' : 'var(--text-secondary)',
                            background: selectedFile === file.path ? 'var(--bg-tertiary)' : 'transparent',
                            whiteSpace: 'nowrap',
                            transition: 'all 0.1s',
                            paddingLeft: file.isDirectory ? '16px' : '24px'
                        }}
                        onMouseEnter={(e) => {
                            if (selectedFile !== file.path) {
                                e.currentTarget.style.background = 'var(--bg-tertiary)'
                                e.currentTarget.style.color = 'var(--text-primary)'
                            }
                        }}
                        onMouseLeave={(e) => {
                            if (selectedFile !== file.path) {
                                e.currentTarget.style.background = 'transparent'
                                e.currentTarget.style.color = 'var(--text-secondary)'
                            }
                        }}
                    >
                        {file.isDirectory ? (
                            <Folder size={14} color="var(--accent-secondary)" />
                        ) : (
                            getFileIcon(file.name) || <File size={14} color="var(--text-secondary)" />
                        )}
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{file.name}</span>
                    </div>
                ))}

                {rootPath && files.length === 0 && (
                    <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.75rem' }}>
                        No files in this folder
                    </div>
                )}
            </div>

            {/* Quick tips */}
            <div style={{
                padding: '10px',
                borderTop: '1px solid var(--border-color)',
                fontSize: '0.7rem',
                color: 'var(--text-secondary)',
                background: 'var(--bg-tertiary)'
            }}>
                💡 Click a file to edit, then Run to execute
            </div>
        </div>
    )
}

export default FileExplorer
