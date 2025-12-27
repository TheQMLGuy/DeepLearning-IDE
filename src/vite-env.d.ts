/// <reference types="vite/client" />

interface FileSystemAPI {
    selectDirectory: () => Promise<string | null>
    readDir: (path: string) => Promise<Array<{ name: string; isDirectory: boolean; path: string }>>
    readFile: (path: string) => Promise<string>
    saveFile: (path: string, content: string) => Promise<{ success?: boolean; error?: string }>
}

interface TerminalAPI {
    init: () => void
    send: (data: string) => void
    onData: (callback: (data: string) => void) => () => void
}

interface AppControlAPI {
    setMode: (mode: string) => void
}

interface KaggleAPI {
    search: (query: string) => Promise<{ success?: boolean; data?: Array<{ id: string; name: string; size: string }>; error?: string }>
}

interface AnalysisAPI {
    recommend: (path: string) => Promise<{ success?: boolean; recommendation?: string; error?: string }>
    checkDeps: (path: string) => Promise<{ missing?: string[]; error?: string }>
}

interface Window {
    fileSystem: FileSystemAPI
    terminal: TerminalAPI
    appControl: AppControlAPI
    kaggle: KaggleAPI
    analysis: AnalysisAPI
    pyodide: any
    pyodideReady: Promise<void>
}
