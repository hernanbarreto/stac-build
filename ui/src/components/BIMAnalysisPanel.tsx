/**
 * BIM Analysis Panel — Construction progress report
 * Shows per-element quality, advance, and global progress
 * when the sábana (deviation comparison) is active.
 */
import './BIMAnalysisPanel.css'

interface ElementMeta {
    element_key: string
    label: string
    ifc_type: string
    status: string
    quality?: string        // good | regular | bad | not_built
    advance_pct?: number
    coverage_pct?: number
    coverage_cumulative?: number
    occluded_pct?: number
    element_state?: string   // NOT_STARTED | IN_PROGRESS | COMPLETED | VERIFIED | OCCLUDED_FROZEN
    correctness_pct?: number
    mean_mm?: number
    bim_surface_m2?: number
    total_points?: number
}

interface SabanaMeta {
    date: string
    tolerance_mm: number
    total_points: number
    global_advance_pct: number
    quality_thresholds?: {
        good_pct: number
        regular_pct: number
    }
    summary: {
        total_elements: number
        evaluated: number
        unmatched: number
        errors: number
    }
    elements: ElementMeta[]
}

interface BIMAnalysisPanelProps {
    meta: SabanaMeta
    sessionId: string
}

const QUALITY_COLORS: Record<string, string> = {
    good: '#4ade80',
    regular: '#fbbf24',
    bad: '#ef4444',
    not_built: '#64748b',
}

const QUALITY_LABELS: Record<string, string> = {
    good: 'Good',
    regular: 'Regular',
    bad: 'Bad',
    not_built: 'Not Built',
}

const QUALITY_ICONS: Record<string, string> = {
    good: '✅',
    regular: '⚠️',
    bad: '❌',
    not_built: '⬜',
}

const STATE_COLORS: Record<string, string> = {
    NOT_STARTED: '#64748b',
    IN_PROGRESS: '#3b82f6',
    COMPLETED: '#4ade80',
    VERIFIED: '#a78bfa',
    OCCLUDED_FROZEN: '#f97316',
}

const STATE_LABELS: Record<string, string> = {
    NOT_STARTED: 'Not Started',
    IN_PROGRESS: 'In Progress',
    COMPLETED: 'Completed',
    VERIFIED: 'Verified',
    OCCLUDED_FROZEN: 'Occluded',
}

const STATE_ICONS: Record<string, string> = {
    NOT_STARTED: '⬜',
    IN_PROGRESS: '🔨',
    COMPLETED: '✅',
    VERIFIED: '🏆',
    OCCLUDED_FROZEN: '🔒',
}

function formatDate(dateStr: string): string {
    try {
        const d = new Date(dateStr)
        return d.toLocaleDateString('en-US', {
            year: 'numeric', month: 'short', day: 'numeric',
            hour: '2-digit', minute: '2-digit',
        })
    } catch {
        return dateStr
    }
}

function getProgressColor(pct: number): string {
    if (pct >= 80) return '#4ade80'
    if (pct >= 50) return '#fbbf24'
    if (pct > 0) return '#fb923c'
    return '#64748b'
}

export function BIMAnalysisPanel({ meta, sessionId }: BIMAnalysisPanelProps) {
    const evaluated = meta.elements.filter(e => e.status === 'evaluated')
    const unmatched = meta.elements.filter(e => e.status !== 'evaluated')

    // Count by quality
    const counts = { good: 0, regular: 0, bad: 0, not_built: unmatched.length }
    evaluated.forEach(e => {
        const q = e.quality || 'bad'
        if (q in counts) counts[q as keyof typeof counts]++
    })

    return (
        <div className="bap-container">
            {/* ── Header / Metadata ── */}
            <div className="bap-header">
                <div className="bap-title">BIM Analysis Report</div>
                <div className="bap-meta-grid">
                    <div className="bap-meta-item">
                        <span className="bap-meta-label">Session</span>
                        <span className="bap-meta-value">{sessionId}</span>
                    </div>
                    <div className="bap-meta-item">
                        <span className="bap-meta-label">Date</span>
                        <span className="bap-meta-value">{formatDate(meta.date)}</span>
                    </div>
                    <div className="bap-meta-item">
                        <span className="bap-meta-label">Tolerance</span>
                        <span className="bap-meta-value">{meta.tolerance_mm} mm</span>
                    </div>
                    <div className="bap-meta-item">
                        <span className="bap-meta-label">Scan Points</span>
                        <span className="bap-meta-value">{meta.total_points.toLocaleString()}</span>
                    </div>
                </div>
            </div>

            {/* ── Global Progress ── */}
            <div className="bap-global">
                <div className="bap-global-label">Global Progress</div>
                <div className="bap-global-bar-container">
                    <div
                        className="bap-global-bar"
                        style={{
                            width: `${Math.min(meta.global_advance_pct, 100)}%`,
                            backgroundColor: getProgressColor(meta.global_advance_pct),
                        }}
                    />
                </div>
                <div className="bap-global-value" style={{ color: getProgressColor(meta.global_advance_pct) }}>
                    {meta.global_advance_pct}%
                </div>
            </div>

            {/* ── Summary Chips ── */}
            <div className="bap-summary">
                <div className="bap-chip" style={{ borderColor: QUALITY_COLORS.good }}>
                    <span className="bap-chip-count" style={{ color: QUALITY_COLORS.good }}>{counts.good}</span>
                    <span className="bap-chip-label">Good</span>
                </div>
                <div className="bap-chip" style={{ borderColor: QUALITY_COLORS.regular }}>
                    <span className="bap-chip-count" style={{ color: QUALITY_COLORS.regular }}>{counts.regular}</span>
                    <span className="bap-chip-label">Regular</span>
                </div>
                <div className="bap-chip" style={{ borderColor: QUALITY_COLORS.bad }}>
                    <span className="bap-chip-count" style={{ color: QUALITY_COLORS.bad }}>{counts.bad}</span>
                    <span className="bap-chip-label">Bad</span>
                </div>
                <div className="bap-chip" style={{ borderColor: QUALITY_COLORS.not_built }}>
                    <span className="bap-chip-count" style={{ color: QUALITY_COLORS.not_built }}>{counts.not_built}</span>
                    <span className="bap-chip-label">Not Built</span>
                </div>
            </div>

            {/* ── Element List ── */}
            <div className="bap-elements-header">Elements ({meta.elements.length})</div>
            <div className="bap-elements-list">
                {evaluated.map(el => (
                    <div key={el.element_key} className="bap-element">
                        <div className="bap-el-top">
                            <span className="bap-el-icon">{QUALITY_ICONS[el.quality || 'bad']}</span>
                            <div className="bap-el-info">
                                <div className="bap-el-label">{el.label.split(':').slice(0, -1).join(':') || el.label}</div>
                                <div className="bap-el-type">{el.ifc_type.replace('Ifc', '')}</div>
                            </div>
                            <span
                                className="bap-el-quality"
                                style={{ color: QUALITY_COLORS[el.quality || 'bad'] }}
                            >
                                {QUALITY_LABELS[el.quality || 'bad']}
                            </span>
                        </div>
                        <div className="bap-el-stats">
                            <div className="bap-el-bar-container">
                                <div
                                    className="bap-el-bar"
                                    style={{
                                        width: `${Math.min(el.advance_pct || 0, 100)}%`,
                                        backgroundColor: QUALITY_COLORS[el.quality || 'bad'],
                                    }}
                                />
                                {el.coverage_cumulative != null && el.coverage_cumulative !== (el.advance_pct || 0) && (
                                    <div
                                        className="bap-el-bar bap-el-bar-cumul"
                                        style={{
                                            width: `${Math.min(el.coverage_cumulative, 100)}%`,
                                            backgroundColor: '#60a5fa33',
                                            position: 'absolute', top: 0, left: 0, height: '100%',
                                        }}
                                    />
                                )}
                            </div>
                            <span className="bap-el-advance">{el.advance_pct || 0}%</span>
                        </div>
                        {el.element_state && (
                            <div className="bap-el-state">
                                <span
                                    className="bap-el-state-badge"
                                    style={{ backgroundColor: STATE_COLORS[el.element_state] + '22', color: STATE_COLORS[el.element_state], borderColor: STATE_COLORS[el.element_state] }}
                                >
                                    {STATE_ICONS[el.element_state] || '❓'} {STATE_LABELS[el.element_state] || el.element_state}
                                </span>
                                {(el.occluded_pct || 0) > 0 && (
                                    <span className="bap-el-occlusion">
                                        🔒 {el.occluded_pct?.toFixed(0)}% occluded
                                    </span>
                                )}
                                {el.coverage_cumulative != null && (
                                    <span className="bap-el-cumul">
                                        📊 {el.coverage_cumulative.toFixed(0)}% cumulative
                                    </span>
                                )}
                            </div>
                        )}
                        <div className="bap-el-details">
                            <span>Correctness: {el.correctness_pct?.toFixed(1)}%</span>
                            <span>Mean: {el.mean_mm?.toFixed(1)} mm</span>
                            {el.bim_surface_m2 ? <span>Surface: {el.bim_surface_m2.toFixed(2)} m²</span> : null}
                        </div>
                    </div>
                ))}
                {unmatched.length > 0 && (
                    <>
                        <div className="bap-section-divider">Not Built ({unmatched.length})</div>
                        {unmatched.map(el => (
                            <div key={el.element_key} className="bap-element bap-element-unbuilt">
                                <div className="bap-el-top">
                                    <span className="bap-el-icon">⬜</span>
                                    <div className="bap-el-info">
                                        <div className="bap-el-label">{el.label.split(':').slice(0, -1).join(':') || el.label}</div>
                                        <div className="bap-el-type">{el.ifc_type.replace('Ifc', '')}</div>
                                    </div>
                                    <span className="bap-el-quality" style={{ color: '#64748b' }}>Not Built</span>
                                </div>
                            </div>
                        ))}
                    </>
                )}
            </div>
        </div>
    )
}
