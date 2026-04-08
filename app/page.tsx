'use client'

import { useState, useRef, useCallback } from 'react'

type Analysis = {
  plain: string
  risk: 'HIGH' | 'MEDIUM' | 'LOW'
  risk_reason: string
  counter: string
  hindi?: { plain: string; counter: string | null } | null
  source?: string
}

type PDFResult = {
  clause: string
  plain: string
  risk: 'HIGH' | 'MEDIUM' | 'LOW'
  risk_reason: string
  counter: string
}

type PDFReport = {
  clauses: PDFResult[]
  missing: string[]
  score: number
  riskCounts: { HIGH: number; MEDIUM: number; LOW: number }
  totalClauses: number
}

const RISK_CONFIG = {
  HIGH: { emoji: '🔴', label: 'High Risk', bg: 'risk-high', color: '#c0392b' },
  MEDIUM: { emoji: '🟡', label: 'Medium Risk', bg: 'risk-medium', color: '#e67e22' },
  LOW: { emoji: '🟢', label: 'Low Risk', bg: 'risk-low', color: '#27ae60' },
}

const SAMPLE_CLAUSES = [
  'The security deposit shall stand forfeited in its entirety in the event the Lessee vacates the premises prior to the expiry of the lock-in period, irrespective of the reason for such vacation.',
  'The Lessor reserves the right to enter and inspect the demised premises at any time without prior notice for the purposes of inspection, repair, or any other reason deemed appropriate by the Lessor.',
  'This agreement shall automatically renew for a further period of eleven months upon expiry of the current term unless either party provides written notice of termination at least two months prior to the expiry date.',
  'All repairs and maintenance of the premises, whether structural or cosmetic, howsoever arising, shall be carried out at the cost and expense of the Lessee.',
]

export default function Home() {
  const [tab, setTab] = useState<'text' | 'pdf'>('text')
  const [clause, setClause] = useState('')
  const [language, setLanguage] = useState<'english' | 'hindi' | 'both'>('english')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<Analysis | null>(null)
  const [pdfFile, setPdfFile] = useState<File | null>(null)
  const [pdfLoading, setPdfLoading] = useState(false)
  const [pdfReport, setPdfReport] = useState<PDFReport | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [expandedClause, setExpandedClause] = useState<number | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const analyseText = async () => {
    if (!clause.trim()) return
    setLoading(true)
    setResult(null)
    try {
      const res = await fetch('/api/analyse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clause, language }),
      })
      const data = await res.json()
      setResult(data)
    } catch {
      alert('Analysis failed. Please try again.')
    }
    setLoading(false)
  }

  const analysePDF = async () => {
    if (!pdfFile) return
    setPdfLoading(true)
    setPdfReport(null)
    const form = new FormData()
    form.append('pdf', pdfFile)
    try {
      const res = await fetch('/api/analyse-pdf', { method: 'POST', body: form })
      const data = await res.json()
      if (data.error) { alert(data.error); return }
      setPdfReport(data)
    } catch {
      alert('PDF analysis failed. Please try again.')
    }
    setPdfLoading(false)
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f?.type === 'application/pdf') setPdfFile(f)
  }, [])

  const scoreColor = (s: number) => s >= 70 ? '#27ae60' : s >= 40 ? '#e67e22' : '#c0392b'

  return (
    <div className="min-h-screen pattern-bg" style={{ fontFamily: 'var(--font-body)' }}>

      {/* ── Header ── */}
      <header style={{ background: '#0d1117', borderBottom: '3px solid #e8a020' }}>
        <div style={{ maxWidth: 1100, margin: '0 auto', padding: '0 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: 68 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 28 }}>⚖️</span>
            <div>
              <div style={{ fontFamily: 'var(--font-display)', fontSize: 22, fontWeight: 700, color: '#f5f0e8', letterSpacing: '-0.5px' }}>ClauseClear</div>
              <div style={{ fontSize: 11, color: '#e8a020', letterSpacing: 1, textTransform: 'uppercase' }}>Kalpathon 2025 · Team Absolute Cinema</div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            {(['english', 'hindi', 'both'] as const).map(l => (
              <button key={l} onClick={() => setLanguage(l)}
                style={{
                  padding: '5px 14px', borderRadius: 20, fontSize: 12, fontWeight: 600,
                  cursor: 'pointer', border: '1px solid',
                  background: language === l ? '#e8a020' : 'transparent',
                  borderColor: language === l ? '#e8a020' : '#3a3f47',
                  color: language === l ? '#0d1117' : '#8a9bb0',
                  transition: 'all 0.2s',
                }}>
                {l === 'english' ? 'EN' : l === 'hindi' ? 'हि' : 'EN+हि'}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* ── Hero ── */}
      <section style={{ background: 'linear-gradient(135deg, #0d1117 0%, #16213e 60%, #1a2a1a 100%)', padding: '60px 24px 52px', textAlign: 'center', position: 'relative', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', inset: 0, backgroundImage: 'radial-gradient(circle at 20% 50%, rgba(232,160,32,0.08) 0%, transparent 60%), radial-gradient(circle at 80% 50%, rgba(26,122,74,0.06) 0%, transparent 60%)' }} />
        <div style={{ position: 'relative', maxWidth: 700, margin: '0 auto' }}>
          <div style={{ display: 'inline-block', background: 'rgba(232,160,32,0.15)', border: '1px solid rgba(232,160,32,0.3)', borderRadius: 20, padding: '4px 16px', fontSize: 12, color: '#e8a020', marginBottom: 20, letterSpacing: 0.5 }}>
            Fine-tuned Gemma 3 270M · QLoRA · Indian Rental Law
          </div>
          <h1 style={{ fontFamily: 'var(--font-display)', fontSize: 'clamp(32px, 5vw, 52px)', color: '#f5f0e8', fontWeight: 900, lineHeight: 1.15, margin: '0 0 16px' }}>
            Understand your rental<br />
            <span style={{ color: '#e8a020' }}>agreement in plain English</span>
          </h1>
          <p style={{ color: '#8a9bb0', fontSize: 17, lineHeight: 1.7, margin: '0 0 28px' }}>
            50M+ Indian renters sign agreements they can't understand. ClauseClear decodes dense legalese,
            flags risky clauses, and suggests fairer terms — in English and Hindi.
          </p>
          <div style={{ display: 'flex', justifyContent: 'center', gap: 20, flexWrap: 'wrap' }}>
            {[['🔴', 'Flags high-risk clauses'], ['💬', 'Suggests counter-clauses'], ['📂', 'Full PDF analysis'], ['🇮🇳', 'Hindi output']].map(([e, t]) => (
              <div key={t} style={{ color: '#8a9bb0', fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
                <span>{e}</span> {t}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Main Content ── */}
      <main style={{ maxWidth: 1000, margin: '0 auto', padding: '40px 20px' }}>

        {/* Tab switcher */}
        <div style={{ display: 'flex', background: 'white', borderRadius: 12, padding: 4, border: '1px solid #e8e0d0', marginBottom: 28, width: 'fit-content' }}>
          {[{ id: 'text', icon: '✏️', label: 'Analyse Clause' }, { id: 'pdf', icon: '📂', label: 'Full PDF Report' }].map(t => (
            <button key={t.id} onClick={() => setTab(t.id as 'text' | 'pdf')}
              style={{
                padding: '10px 24px', borderRadius: 9, border: 'none', cursor: 'pointer',
                fontFamily: 'var(--font-body)', fontSize: 14, fontWeight: 600,
                background: tab === t.id ? '#0d1117' : 'transparent',
                color: tab === t.id ? '#f5f0e8' : '#8a9bb0',
                transition: 'all 0.2s',
              }}>
              {t.icon} {t.label}
            </button>
          ))}
        </div>

        {/* ── TEXT TAB ── */}
        {tab === 'text' && (
          <div style={{ display: 'grid', gap: 28 }}>
            {/* Input card */}
            <div className="clause-card" style={{ border: '1px solid #e8e0d0' }}>
              <label style={{ display: 'block', fontWeight: 600, marginBottom: 10, fontSize: 14, color: '#2c3e50' }}>
                Paste your rental clause here
              </label>
              <textarea
                value={clause}
                onChange={e => setClause(e.target.value)}
                placeholder="e.g. The security deposit shall stand forfeited in its entirety in the event the Lessee vacates the premises prior to the expiry of the lock-in period..."
                rows={5}
                style={{
                  width: '100%', padding: '14px 16px', borderRadius: 8, border: '1.5px solid #e8e0d0',
                  fontFamily: 'var(--font-body)', fontSize: 14, lineHeight: 1.6,
                  background: '#faf7f2', resize: 'vertical', outline: 'none',
                  transition: 'border-color 0.2s',
                }}
                onFocus={e => e.target.style.borderColor = '#e8a020'}
                onBlur={e => e.target.style.borderColor = '#e8e0d0'}
              />

              {/* Sample clauses */}
              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: 12, color: '#8a9bb0', marginBottom: 8 }}>Try a sample:</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  {SAMPLE_CLAUSES.map((s, i) => (
                    <button key={i} onClick={() => setClause(s)}
                      style={{
                        fontSize: 11, padding: '4px 12px', borderRadius: 14, border: '1px solid #e8e0d0',
                        background: '#faf7f2', color: '#5a6a7a', cursor: 'pointer',
                        transition: 'all 0.15s',
                      }}
                      onMouseEnter={e => { (e.target as HTMLElement).style.borderColor = '#e8a020'; (e.target as HTMLElement).style.color = '#c4841a' }}
                      onMouseLeave={e => { (e.target as HTMLElement).style.borderColor = '#e8e0d0'; (e.target as HTMLElement).style.color = '#5a6a7a' }}>
                      Sample {i + 1}
                    </button>
                  ))}
                </div>
              </div>

              <button onClick={analyseText} disabled={loading || !clause.trim()}
                style={{
                  marginTop: 16, padding: '13px 32px', borderRadius: 8, border: 'none',
                  background: loading || !clause.trim() ? '#c8bfa8' : '#0d1117',
                  color: '#f5f0e8', fontWeight: 600, fontSize: 15, cursor: loading || !clause.trim() ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s', fontFamily: 'var(--font-body)',
                  animation: !loading && clause.trim() ? 'pulse-ring 2s infinite' : 'none',
                }}>
                {loading ? '⏳ Analysing...' : '⚖️ Analyse Clause'}
              </button>
            </div>

            {/* Loading skeleton */}
            {loading && (
              <div style={{ display: 'grid', gap: 12 }}>
                {[80, 100, 60].map((w, i) => (
                  <div key={i} className="shimmer" style={{ height: 18, borderRadius: 9, width: `${w}%` }} />
                ))}
              </div>
            )}

            {/* Result */}
            {result && !loading && (
              <div className="animate-slide-up" style={{ display: 'grid', gap: 16 }}>
                {/* Plain English */}
                <div className="clause-card" style={{ borderLeft: `4px solid ${RISK_CONFIG[result.risk].color}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap' }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: '#8a9bb0', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>Plain English</div>
                      <p style={{ fontSize: 17, lineHeight: 1.65, margin: 0, color: '#1a1a2e', fontWeight: 500 }}>{result.plain}</p>
                      {result.hindi && language !== 'english' && (
                        <p style={{ marginTop: 10, fontSize: 15, lineHeight: 1.65, color: '#2c3e50', fontFamily: 'serif' }}>🇮🇳 {result.hindi.plain}</p>
                      )}
                    </div>
                    <div className={result.risk === 'HIGH' ? 'risk-high' : result.risk === 'MEDIUM' ? 'risk-medium' : 'risk-low'}
                      style={{ padding: '6px 16px', borderRadius: 20, fontSize: 13, fontWeight: 700, whiteSpace: 'nowrap', flexShrink: 0 }}>
                      {RISK_CONFIG[result.risk].emoji} {RISK_CONFIG[result.risk].label}
                    </div>
                  </div>
                  <p style={{ margin: '12px 0 0', fontSize: 13, color: '#8a9bb0' }}>{result.risk_reason}</p>
                </div>

                {/* Counter-clause */}
                <div className="clause-card" style={{ background: '#eafaf1', borderColor: '#a9dfbf' }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#1a7a4a', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>💬 Suggested Counter-Clause</div>
                  <p style={{ margin: 0, fontSize: 14, lineHeight: 1.7, color: '#1a4a2e' }}>{result.counter}</p>
                  {result.hindi?.counter && language !== 'english' && (
                    <p style={{ marginTop: 10, fontSize: 13, color: '#2c6040', fontFamily: 'serif' }}>🇮🇳 {result.hindi.counter}</p>
                  )}
                  <button onClick={() => navigator.clipboard.writeText(result.counter)}
                    style={{ marginTop: 12, fontSize: 12, padding: '5px 14px', borderRadius: 6, border: '1px solid #a9dfbf', background: 'white', color: '#1a7a4a', cursor: 'pointer', fontWeight: 600 }}>
                    📋 Copy Counter-Clause
                  </button>
                </div>

                {result.source === 'engine' && (
                  <p style={{ fontSize: 12, color: '#b0a898', textAlign: 'center' }}>
                    ℹ️ Using built-in rule engine · Set HF_MODEL_ID env var to use your fine-tuned Gemma 3 model
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── PDF TAB ── */}
        {tab === 'pdf' && (
          <div style={{ display: 'grid', gap: 28 }}>
            {/* Drop zone */}
            <div
              onDragOver={e => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileRef.current?.click()}
              style={{
                border: `2px dashed ${dragOver ? '#e8a020' : '#c8bfa8'}`,
                borderRadius: 12, padding: '48px 24px', textAlign: 'center',
                background: dragOver ? 'rgba(232,160,32,0.05)' : 'white',
                cursor: 'pointer', transition: 'all 0.2s',
              }}>
              <input ref={fileRef} type="file" accept=".pdf" style={{ display: 'none' }}
                onChange={e => e.target.files?.[0] && setPdfFile(e.target.files[0])} />
              <div style={{ fontSize: 48, marginBottom: 16 }}>📄</div>
              {pdfFile ? (
                <div>
                  <div style={{ fontWeight: 600, color: '#1a7a4a', fontSize: 16 }}>✅ {pdfFile.name}</div>
                  <div style={{ color: '#8a9bb0', fontSize: 13, marginTop: 4 }}>{(pdfFile.size / 1024).toFixed(0)} KB · Click to change</div>
                </div>
              ) : (
                <div>
                  <div style={{ fontWeight: 600, fontSize: 16, color: '#2c3e50' }}>Drop your rental agreement PDF here</div>
                  <div style={{ color: '#8a9bb0', marginTop: 6, fontSize: 14 }}>or click to browse · Max 5MB · Text-based PDF only</div>
                </div>
              )}
            </div>

            {pdfFile && (
              <button onClick={analysePDF} disabled={pdfLoading}
                style={{
                  padding: '14px 32px', borderRadius: 8, border: 'none', width: 'fit-content',
                  background: pdfLoading ? '#c8bfa8' : '#0d1117', color: '#f5f0e8',
                  fontWeight: 600, fontSize: 15, cursor: pdfLoading ? 'not-allowed' : 'pointer',
                  fontFamily: 'var(--font-body)',
                }}>
                {pdfLoading ? '⏳ Analysing agreement...' : '📊 Generate Health Report'}
              </button>
            )}

            {/* PDF Report */}
            {pdfReport && (
              <div className="animate-slide-up" style={{ display: 'grid', gap: 20 }}>
                {/* Score header */}
                <div className="clause-card" style={{ background: '#0d1117', color: 'white' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 32, alignItems: 'center' }}>
                    <div>
                      <div style={{ fontSize: 64, fontWeight: 900, color: scoreColor(pdfReport.score), fontFamily: 'var(--font-display)', lineHeight: 1 }}>{pdfReport.score}</div>
                      <div style={{ color: '#8a9bb0', fontSize: 13, marginTop: 4 }}>/ 100 Health Score</div>
                    </div>
                    <div>
                      <div style={{ fontFamily: 'var(--font-display)', fontSize: 22, color: '#f5f0e8', marginBottom: 16 }}>Agreement Health Report</div>
                      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                        {[
                          { label: '🔴 High Risk', count: pdfReport.riskCounts.HIGH, color: '#e74c3c' },
                          { label: '🟡 Medium', count: pdfReport.riskCounts.MEDIUM, color: '#e67e22' },
                          { label: '🟢 Low Risk', count: pdfReport.riskCounts.LOW, color: '#27ae60' },
                        ].map(s => (
                          <div key={s.label} style={{ textAlign: 'center' }}>
                            <div style={{ fontSize: 28, fontWeight: 700, color: s.color }}>{s.count}</div>
                            <div style={{ fontSize: 11, color: '#8a9bb0' }}>{s.label}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Missing clauses */}
                {pdfReport.missing.length > 0 && (
                  <div style={{ background: '#fef9e7', border: '1px solid #f8d9a0', borderLeft: '4px solid #e67e22', borderRadius: 12, padding: 20 }}>
                    <div style={{ fontWeight: 700, color: '#c4841a', marginBottom: 12, fontSize: 15 }}>⚠️ Missing Protective Clauses ({pdfReport.missing.length})</div>
                    {pdfReport.missing.map((m, i) => (
                      <div key={i} style={{ fontSize: 14, color: '#7a5c20', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span>•</span> This agreement has no <strong>{m}</strong> — this could disadvantage you as a tenant.
                      </div>
                    ))}
                  </div>
                )}

                {/* Per-clause breakdown */}
                <div>
                  <h3 style={{ fontFamily: 'var(--font-display)', fontSize: 20, marginBottom: 16 }}>Clause-by-Clause Breakdown</h3>
                  {pdfReport.clauses.map((c, i) => (
                    <div key={i} className="clause-card" style={{ marginBottom: 12, borderLeft: `4px solid ${RISK_CONFIG[c.risk]?.color || '#8a9bb0'}` }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, marginBottom: 10 }}>
                        <div style={{ fontSize: 11, color: '#8a9bb0', fontWeight: 600, textTransform: 'uppercase', letterSpacing: 1 }}>Clause {i + 1}</div>
                        <div className={c.risk === 'HIGH' ? 'risk-high' : c.risk === 'MEDIUM' ? 'risk-medium' : 'risk-low'}
                          style={{ padding: '3px 12px', borderRadius: 12, fontSize: 11, fontWeight: 700, flexShrink: 0 }}>
                          {RISK_CONFIG[c.risk]?.emoji} {RISK_CONFIG[c.risk]?.label}
                        </div>
                      </div>

                      <p style={{ fontSize: 16, fontWeight: 500, color: '#1a1a2e', lineHeight: 1.6, margin: '0 0 10px' }}>{c.plain}</p>

                      <button onClick={() => setExpandedClause(expandedClause === i ? null : i)}
                        style={{ fontSize: 12, color: '#8a9bb0', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}>
                        {expandedClause === i ? '▲ Hide details' : '▼ See original + counter-clause'}
                      </button>

                      {expandedClause === i && (
                        <div style={{ marginTop: 12, display: 'grid', gap: 10 }}>
                          <div style={{ background: '#faf7f2', borderRadius: 8, padding: 12 }}>
                            <div style={{ fontSize: 11, fontWeight: 600, color: '#8a9bb0', marginBottom: 6 }}>ORIGINAL CLAUSE</div>
                            <p style={{ margin: 0, fontSize: 13, color: '#4a5568', lineHeight: 1.6 }}>{c.clause}</p>
                          </div>
                          <div style={{ background: '#eafaf1', borderRadius: 8, padding: 12 }}>
                            <div style={{ fontSize: 11, fontWeight: 600, color: '#1a7a4a', marginBottom: 6 }}>💬 COUNTER-CLAUSE</div>
                            <p style={{ margin: 0, fontSize: 13, color: '#1a4a2e', lineHeight: 1.6 }}>{c.counter}</p>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

      </main>

      {/* ── Footer ── */}
      <footer style={{ background: '#0d1117', borderTop: '1px solid #1e2530', padding: '32px 24px', marginTop: 60, textAlign: 'center' }}>
        <div style={{ color: '#8a9bb0', fontSize: 13, lineHeight: 1.8 }}>
          <div style={{ fontFamily: 'var(--font-display)', color: '#f5f0e8', fontSize: 16, marginBottom: 8 }}>⚖️ ClauseClear</div>
          <div>Built by <strong style={{ color: '#e8a020' }}>Team Absolute Cinema</strong> for Kalpathon 2025</div>
          <div style={{ marginTop: 6 }}>Fine-tuned Gemma 3 270M · QLoRA · Indian Rental Law · Model Tenancy Act 2021</div>
          <div style={{ marginTop: 12, fontSize: 11, color: '#5a6a7a' }}>
            ⚠️ This tool is for informational purposes only and does not constitute legal advice.
            Always consult a lawyer for important legal decisions.
          </div>
        </div>
      </footer>

    </div>
  )
}
