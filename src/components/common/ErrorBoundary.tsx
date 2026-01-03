// Error Boundary Component - Prevents full app crashes
import { Component, ErrorInfo, ReactNode } from 'react'
import { AlertCircle, RefreshCw } from 'lucide-react'

interface Props {
    children: ReactNode
    fallback?: ReactNode
}

interface State {
    hasError: boolean
    error: Error | null
    errorInfo: ErrorInfo | null
}

class ErrorBoundary extends Component<Props, State> {
    constructor(props: Props) {
        super(props)
        this.state = {
            hasError: false,
            error: null,
            errorInfo: null
        }
    }

    static getDerivedStateFromError(error: Error): State {
        return {
            hasError: true,
            error,
            errorInfo: null
        }
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error('Error caught by boundary:', error, errorInfo)
        this.setState({
            error,
            errorInfo
        })

        // Log to error reporting service in production
        if (process.env.NODE_ENV === 'production') {
            // Example: Sentry.captureException(error, { extra: errorInfo })
        }
    }

    handleReset = () => {
        this.setState({
            hasError: false,
            error: null,
            errorInfo: null
        })
    }

    render() {
        if (this.state.hasError) {
            if (this.props.fallback) {
                return this.props.fallback
            }

            return (
                <div style={{
                    padding: '40px',
                    textAlign: 'center',
                    background: 'rgba(239, 68, 68, 0.1)',
                    border: '2px solid #ef4444',
                    borderRadius: '12px',
                    margin: '20px',
                    color: '#e0e0e0'
                }}>
                    <AlertCircle
                        size={48}
                        color="#ef4444"
                        style={{ marginBottom: '16px' }}
                    />

                    <h2 style={{
                        margin: '0 0 8px 0',
                        color: '#ef4444',
                        fontSize: '24px'
                    }}>
                        Something went wrong
                    </h2>

                    <p style={{
                        margin: '0 0 24px 0',
                        color: '#9ca3af',
                        fontSize: '14px'
                    }}>
                        {this.state.error?.message || 'An unexpected error occurred'}
                    </p>

                    <button
                        onClick={this.handleReset}
                        style={{
                            padding: '10px 20px',
                            background: '#ef4444',
                            border: 'none',
                            borderRadius: '8px',
                            color: 'white',
                            fontSize: '14px',
                            fontWeight: 600,
                            cursor: 'pointer',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '8px',
                            transition: 'all 0.2s'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.background = '#dc2626'
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.background = '#ef4444'
                        }}
                    >
                        <RefreshCw size={16} />
                        Try Again
                    </button>

                    {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
                        <details style={{
                            marginTop: '24px',
                            textAlign: 'left',
                            background: 'rgba(0,0,0,0.3)',
                            padding: '16px',
                            borderRadius: '8px',
                            fontSize: '12px',
                            fontFamily: 'monospace'
                        }}>
                            <summary style={{ cursor: 'pointer', marginBottom: '8px', color: '#f59e0b' }}>
                                Error Details (Development Only)
                            </summary>
                            <pre style={{
                                margin: 0,
                                overflow: 'auto',
                                color: '#e0e0e0'
                            }}>
                                {this.state.error?.stack}
                                {'\n\n'}
                                {this.state.errorInfo.componentStack}
                            </pre>
                        </details>
                    )}
                </div>
            )
        }

        return this.props.children
    }
}

export default ErrorBoundary

// Cell-specific error boundary
export class CellErrorBoundary extends Component<Props, State> {
    constructor(props: Props) {
        super(props)
        this.state = {
            hasError: false,
            error: null,
            errorInfo: null
        }
    }

    static getDerivedStateFromError(error: Error): State {
        return {
            hasError: true,
            error,
            errorInfo: null
        }
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error('Cell error:', error, errorInfo)
        this.setState({ error, errorInfo })
    }

    render() {
        if (this.state.hasError) {
            return (
                <div style={{
                    padding: '16px',
                    background: 'rgba(239, 68, 68, 0.1)',
                    border: '1px solid #ef4444',
                    borderRadius: '8px',
                    margin: '8px 0'
                }}>
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        color: '#ef4444',
                        marginBottom: '8px'
                    }}>
                        <AlertCircle size={16} />
                        <strong>Cell Error</strong>
                    </div>
                    <p style={{
                        margin: '0 0 12px 0',
                        fontSize: '13px',
                        color: '#9ca3af'
                    }}>
                        {this.state.error?.message || 'An error occurred in this cell'}
                    </p>
                    <button
                        onClick={() => this.setState({ hasError: false, error: null, errorInfo: null })}
                        style={{
                            padding: '6px 12px',
                            background: '#ef4444',
                            border: 'none',
                            borderRadius: '6px',
                            color: 'white',
                            fontSize: '12px',
                            cursor: 'pointer'
                        }}
                    >
                        Reset Cell
                    </button>
                </div>
            )
        }

        return this.props.children
    }
}
