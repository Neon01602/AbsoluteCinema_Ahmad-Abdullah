import { NextRequest, NextResponse } from 'next/server'

function splitIntoClauses(text: string): string[] {
  // Try numbered clauses first
  let clauses = text.split(/(?=\b\d{1,2}\.\s+[A-Z])/).filter(s => s.trim().length > 60)
  
  // Try "Clause N" pattern
  if (clauses.length < 3) {
    clauses = text.split(/(?=Clause\s+\d+)/i).filter(s => s.trim().length > 60)
  }
  
  // Fallback: split by full stops before capitals
  if (clauses.length < 3) {
    clauses = text.split(/(?<=[.!?])\s+(?=[A-Z][a-z])/).filter(s => s.trim().length > 80)
  }

  return clauses.slice(0, 20).map(c => c.trim())
}

const MISSING_CLAUSE_CHECKS = [
  { id: 'deposit_return', keywords: ['refund', 'return.*deposit', 'deposit.*return', 'within.*days.*deposit'], message: 'Deposit return timeline' },
  { id: 'notice_period', keywords: ['notice period', 'days.*notice', 'notice.*days', 'vacate.*notice'], message: 'Notice period for vacating' },
  { id: 'maintenance', keywords: ['maintenance', 'repair', 'upkeep'], message: 'Maintenance responsibility' },
  { id: 'escalation', keywords: ['escalation', 'rent.*increase', 'hike'], message: 'Rent escalation terms' },
  { id: 'entry_rights', keywords: ['entry', 'inspect', 'enter.*premises'], message: 'Landlord entry rights / privacy protection' },
  { id: 'dispute', keywords: ['dispute', 'arbitration', 'court'], message: 'Dispute resolution mechanism' },
]

function detectMissingClauses(text: string) {
  const lower = text.toLowerCase()
  return MISSING_CLAUSE_CHECKS
    .filter(c => !c.keywords.some(kw => new RegExp(kw, 'i').test(lower)))
    .map(c => c.message)
}

async function analyseClause(clause: string): Promise<{ plain: string; risk: string; risk_reason: string; counter: string }> {
  const baseUrl = process.env.VERCEL_URL
    ? `https://${process.env.VERCEL_URL}`
    : process.env.NEXTAUTH_URL || 'http://localhost:3000'

  const res = await fetch(`${baseUrl}/api/analyse`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ clause })
  })
  return res.json()
}

export async function POST(req: NextRequest) {
  const formData = await req.formData()
  const file = formData.get('pdf') as File | null

  if (!file) {
    return NextResponse.json({ error: 'No PDF file provided.' }, { status: 400 })
  }

  if (file.size > 5 * 1024 * 1024) {
    return NextResponse.json({ error: 'PDF too large. Maximum 5MB.' }, { status: 400 })
  }

  let text = ''
  
  try {
    const arrayBuffer = await file.arrayBuffer()
    const buffer = Buffer.from(arrayBuffer)

    // Try pdf-parse
    const pdfParse = require('pdf-parse')
    const data = await pdfParse(buffer)
    text = data.text
  } catch (e) {
    return NextResponse.json({ error: 'Could not parse PDF. Please try a text-based PDF.' }, { status: 400 })
  }

  if (!text || text.trim().length < 100) {
    return NextResponse.json({ error: 'Could not extract readable text. Try a non-scanned PDF.' }, { status: 400 })
  }

  const clauses = splitIntoClauses(text)

  if (clauses.length === 0) {
    return NextResponse.json({ error: 'No clauses detected in this PDF.' }, { status: 400 })
  }

  // Analyse all clauses in parallel (limit concurrency)
  const BATCH_SIZE = 5
  const results = []
  for (let i = 0; i < clauses.length; i += BATCH_SIZE) {
    const batch = clauses.slice(i, i + BATCH_SIZE)
    const batchResults = await Promise.all(batch.map(async c => {
      const analysis = await analyseClause(c)
      return { clause: c, ...analysis }
    }))
    results.push(...batchResults)
  }

  const missing = detectMissingClauses(text)
  
  const riskCounts = { HIGH: 0, MEDIUM: 0, LOW: 0 }
  results.forEach(r => { riskCounts[r.risk as keyof typeof riskCounts]++ })
  
  const score = Math.max(0, 100 - riskCounts.HIGH * 18 - riskCounts.MEDIUM * 7 - missing.length * 5)

  return NextResponse.json({
    clauses: results,
    missing,
    score,
    riskCounts,
    totalClauses: results.length,
  })
}

export const config = { api: { bodyParser: false } }
