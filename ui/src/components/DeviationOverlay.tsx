/**
 * STAC Build — Deviation Analysis Overlay
 * BIM vs Scan heatmap panel with histogram, stats, and controls.
 * Hernán Barreto — Ingerop IN3
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import type { ViewportHandle } from './Viewport'
import './DeviationOverlay.css'

// ── Types ──

export interface DeviationMatch {
    segment_label: string
    element_key: string
    ifc_type?: string
    ifc_name?: string
}

export interface DeviationElementResult {
    element_key: string
    label: string
    status?: string
    ifc_type?: string
    distances_mm?: number[]
    point_indices?: number[]
    total_points?: number
    sabana_positions?: number[]
    sabana_colors?: number[]
    sabana_n_points?: number
    n_faces?: number
    stats?: {
        min_mm: number
        max_mm: number
        mean_mm: number
        std_mm: number
        median_mm: number
        p95_mm: number
        within_tolerance: number
        total_points: number
        pass_rate: number
        tolerance_mm: number
    }
    histogram?: {
        counts: number[]
        bin_edges_mm: number[]
    }
    error?: string
}

export interface DeviationResult {
    ok: boolean
    date?: string
    tolerance_mm: number
    transform?: number[][]
    results: DeviationElementResult[]
}

interface Props {
    sessionId: string
    viewportRef?: React.RefObject<ViewportHandle | null>
    onClose: () => void
    onHeatmapData?: (data: DeviationResult | null) => void
}

// ── Component ──

export default function DeviationOverlay({ sessionId, viewportRef, onClose, onHeatmapData }: Props) {
    const [matches, setMatches] = useState<DeviationMatch[]>([])
    const [result, setResult] = useState<DeviationResult | null>(null)
    const [loading, setLoading] = useState(false)
    const [tolerance, setTolerance] = useState(15) // mm
    const [error, setError] = useState<string | null>(null)
    const histCanvasRef = useRef<HTMLCanvasElement>(null)

    // Auto-match on mount
    useEffect(() => {
        fetch(`/api/bim/auto_match/${sessionId}`)
            .then(r => r.json())
            .then(data => {
                if (data.matches) setMatches(data.matches)
                if (data.error) setError(data.error)
            })
            .catch(() => setError('Failed to auto-match'))
    }, [sessionId])

    // Run comparison
    const runComparison = useCallback(async () => {
        if (!matches.length) return
        setLoading(true)
        setError(null)
        try {
            const res = await fetch('/api/bim/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId,
                    matches: matches.map(m => ({
                        segment_label: m.segment_label,
                        element_key: m.element_key,
                        ifc_type: m.ifc_type,
                    })),
                    tolerance_mm: tolerance,
                }),
            })
            if (!res.ok) throw new Error(await res.text())
            const data: DeviationResult = await res.json()
            setResult(data)
            onHeatmapData?.(data)
            // Apply sábana + move cloud
            if (viewportRef?.current && data.results) {
                // Apply registration transform to point cloud
                if (data.transform) {
                    viewportRef.current.applyRegistrationTransform(data.transform)
                }
                // Collect sábana point clouds per evaluated element
                const sabanaData: Record<string, { positions: number[], colors: number[] }> = {}
                const unmatchedKeys: string[] = []
                for (const r of data.results) {
                    if (r.sabana_positions && r.sabana_positions.length > 0) {
                        sabanaData[r.element_key] = {
                            positions: r.sabana_positions,
                            colors: r.sabana_colors || [],
                        }
                    }
                    if (r.status === 'unmatched') {
                        unmatchedKeys.push(r.element_key)
                    }
                }
                viewportRef.current.applyDeviationSurface(sabanaData, unmatchedKeys)
            }
        } catch (e: any) {
            setError(e.message || 'Comparison failed')
        } finally {
            setLoading(false)
        }
    }, [sessionId, matches, tolerance, onHeatmapData])

    // Draw histogram when result changes
    useEffect(() => {
        if (!result || !histCanvasRef.current) return
        const canvas = histCanvasRef.current
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        // Aggregate histogram from all results
        const allResults = result.results.filter(r => r.histogram)
        if (!allResults.length) return

        const hist = allResults[0].histogram!
        const maxCount = Math.max(...hist.counts, 1)

        canvas.width = canvas.offsetWidth * 2
        canvas.height = canvas.offsetHeight * 2
        ctx.scale(2, 2)
        const w = canvas.offsetWidth
        const h = canvas.offsetHeight

        ctx.clearRect(0, 0, w, h)
        const barW = w / hist.counts.length

        for (let i = 0; i < hist.counts.length; i++) {
            const barH = (hist.counts[i] / maxCount) * (h - 4)
            const pct = i / hist.counts.length
            // Green → Yellow → Red gradient
            const r = pct < 0.5 ? Math.round(pct * 2 * 255) : 255
            const g = pct < 0.5 ? 255 : Math.round((1 - (pct - 0.5) * 2) * 255)
            ctx.fillStyle = `rgb(${r}, ${g}, 0)`
            ctx.fillRect(i * barW, h - barH, barW - 1, barH)
        }

        // Tolerance line
        if (allResults[0].stats) {
            const tolMm = tolerance
            const maxMm = hist.bin_edges_mm[hist.bin_edges_mm.length - 1]
            const tolX = (tolMm / maxMm) * w
            if (tolX > 0 && tolX < w) {
                ctx.strokeStyle = '#00d4ff'
                ctx.lineWidth = 1.5
                ctx.setLineDash([4, 3])
                ctx.beginPath()
                ctx.moveTo(tolX, 0)
                ctx.lineTo(tolX, h)
                ctx.stroke()
                ctx.setLineDash([])
            }
        }
    }, [result, tolerance])

    // Aggregate stats
    const aggStats = result?.results.reduce(
        (acc, r) => {
            if (!r.stats) return acc
            return {
                totalPoints: acc.totalPoints + r.stats.total_points,
                withinTol: acc.withinTol + r.stats.within_tolerance,
                maxDev: Math.max(acc.maxDev, r.stats.max_mm),
                meanDevSum: acc.meanDevSum + r.stats.mean_mm * r.stats.total_points,
            }
        },
        { totalPoints: 0, withinTol: 0, maxDev: 0, meanDevSum: 0 }
    )

    const passRate = aggStats && aggStats.totalPoints > 0
        ? (aggStats.withinTol / aggStats.totalPoints * 100)
        : 0
    const meanDev = aggStats && aggStats.totalPoints > 0
        ? aggStats.meanDevSum / aggStats.totalPoints
        : 0

    const passColor = passRate >= 90 ? '#00ff88' : passRate >= 70 ? '#ffcc00' : '#ff4466'

    return (
        <div className="deviation-panel">
            {/* Header */}
            <div className="deviation-panel-header">
                <h3>⊿ Deviation Analysis</h3>
                <button className="deviation-close-btn" onClick={onClose}>✕</button>
            </div>

            {/* Matches info */}
            {!result && (
                <div style={{ padding: '12px 16px', fontSize: 11, color: '#8892a4' }}>
                    {matches.length > 0
                        ? `${matches.length} segment${matches.length > 1 ? 's' : ''} matched to IFC elements`
                        : 'No matches found — segment labels must match IFC element IDs'
                    }
                    {error && <div style={{ color: '#ff4466', marginTop: 6 }}>{error}</div>}
                </div>
            )}

            {/* Stats */}
            {result && aggStats && aggStats.totalPoints > 0 && (
                <>
                    <div className="deviation-stats">
                        <div className="deviation-stat">
                            <div className="deviation-stat-value" style={{ color: '#00ff88' }}>
                                {meanDev.toFixed(1)}
                            </div>
                            <div className="deviation-stat-label">Mean (mm)</div>
                        </div>
                        <div className="deviation-stat">
                            <div className="deviation-stat-value" style={{ color: '#ffcc00' }}>
                                {aggStats.maxDev.toFixed(1)}
                            </div>
                            <div className="deviation-stat-label">Max (mm)</div>
                        </div>
                        <div className="deviation-stat">
                            <div className="deviation-stat-value" style={{ color: passColor }}>
                                {passRate.toFixed(0)}%
                            </div>
                            <div className="deviation-stat-label">Pass Rate</div>
                        </div>
                    </div>

                    {/* Pass Rate Bar */}
                    <div className="deviation-pass-rate">
                        <div className="deviation-pass-bar">
                            <div
                                className="deviation-pass-fill"
                                style={{
                                    width: `${passRate}%`,
                                    background: `linear-gradient(90deg, #00ff88, ${passColor})`,
                                }}
                            />
                        </div>
                        <div className="deviation-pass-label">
                            <span>{aggStats.withinTol.toLocaleString()} / {aggStats.totalPoints.toLocaleString()} pts</span>
                            <span className="deviation-pass-pct" style={{ color: passColor }}>
                                ±{tolerance}mm tolerance
                            </span>
                        </div>
                    </div>

                    {/* Histogram */}
                    <div className="deviation-histogram">
                        <div className="deviation-histogram-title">Distance Distribution</div>
                        <canvas ref={histCanvasRef} className="deviation-histogram-canvas" />
                    </div>

                    {/* Color Legend */}
                    <div className="deviation-legend">
                        <span className="deviation-legend-label">0mm</span>
                        <div className="deviation-legend-gradient" />
                        <span className="deviation-legend-label">
                            {result.results[0]?.histogram
                                ? `${result.results[0].histogram.bin_edges_mm.slice(-1)[0].toFixed(0)}mm`
                                : `${tolerance * 3}mm`}
                        </span>
                    </div>

                    {/* Element Results */}
                    <div className="deviation-elements">
                        {result.results.map((r, i) => (
                            <div key={i} className="deviation-element">
                                <div
                                    className="deviation-element-indicator"
                                    style={{
                                        background: r.error ? '#666'
                                            : r.stats && r.stats.pass_rate >= 90 ? '#00ff88'
                                                : r.stats && r.stats.pass_rate >= 70 ? '#ffcc00'
                                                    : '#ff4466',
                                    }}
                                />
                                <span className="deviation-element-name" title={r.label}>
                                    {r.label}
                                </span>
                                <span
                                    className="deviation-element-value"
                                    style={{
                                        color: r.error ? '#666'
                                            : r.stats && r.stats.pass_rate >= 90 ? '#00ff88'
                                                : r.stats && r.stats.pass_rate >= 70 ? '#ffcc00'
                                                    : '#ff4466',
                                    }}
                                >
                                    {r.error ? 'ERR' : r.stats ? `${r.stats.pass_rate}%` : '—'}
                                </span>
                            </div>
                        ))}
                    </div>
                </>
            )}

            {/* Tolerance Slider */}
            <div className="deviation-tolerance">
                <div className="deviation-tolerance-header">
                    <span className="deviation-tolerance-label">Tolerance</span>
                    <span className="deviation-tolerance-value">±{tolerance}mm</span>
                </div>
                <input
                    type="range"
                    min={1}
                    max={50}
                    value={tolerance}
                    onChange={e => setTolerance(Number(e.target.value))}
                />
            </div>

            {/* Run Button */}
            <button
                className="deviation-run-btn"
                onClick={runComparison}
                disabled={loading || matches.length === 0}
            >
                {loading ? '⟳ Computing...' : '▶ Run Comparison'}
            </button>
        </div>
    )
}
