import { NextRequest, NextResponse } from 'next/server'

/* ─────────────────────────────────────────────
   RISK PATTERNS (Improved)
───────────────────────────────────────────── */
const RISK_PATTERNS = {
  HIGH: [
    /forfeit|forfeiture/i,
    /deposit.*forfeit/i,
    /sole.*discretion/i,
    /without.*notice/i,
    /immediate.*vacate/i,
    /no liability.*whatsoever/i,
    /all repairs.*lessee/i,
    /double.*rent/i,
    /mesne profits/i,
    /enter.*without.*notice/i,
  ],
  MEDIUM: [
    /auto.*renew/i,
    /automatic.*renew/i,
    /15%|percent.*increase/i,
    /arbitration.*final/i,
    /stamp duty/i,
    /7 days|10 days/i,
    /approval/i,
  ],
  LOW: [
    /fair wear and tear/i,
    /residential.*purpose/i,
    /utility.*charges/i,
    /society.*rules/i,
  ]
}

/* ─────────────────────────────────────────────
   COUNTER CLAUSES
───────────────────────────────────────────── */
const COUNTER_CLAUSES: Record<string, string> = {
  'deposit forfeiture': 'Only one month\'s rent equivalent shall be deducted for early exit; remaining deposit shall be refunded within 30 days.',
  'no notice termination': 'Either party must provide at least 30 days written notice before termination.',
  'auto renewal': 'This agreement shall expire automatically unless renewed by mutual written consent.',
  'all repairs lessee': 'Structural repairs shall be borne by landlord; minor repairs may be shared reasonably.',
  'sole discretion': 'Any deductions must be justified with written proof and itemized breakdown.',
  'no notice entry': 'Landlord must provide 48 hours notice before entering, except emergencies.',
  'default': 'This clause should be reviewed carefully for fairness and legal balance.'
}

/* ─────────────────────────────────────────────
   RISK SCORING ENGINE (NEW)
───────────────────────────────────────────── */
function detectRisk(clause: string) {
  const text = clause.toLowerCase()
  let score = 0

  for (const p of RISK_PATTERNS.HIGH) if (p.test(text)) score += 40
  for (const p of RISK_PATTERNS.MEDIUM) if (p.test(text)) score += 20
  for (const p of RISK_PATTERNS.LOW) if (p.test(text)) score += 5

  let risk: 'HIGH' | 'MEDIUM' | 'LOW' = 'LOW'
  let reason = 'This clause is standard and generally fair.'

  if (score >= 70) {
    risk = 'HIGH'
    reason = 'This clause heavily favors the landlord and reduces tenant rights.'
  } else if (score >= 30) {
    risk = 'MEDIUM'
    reason = 'This clause contains negotiable but potentially unfair terms.'
  }

  return {
    risk,
    score: Math.min(100, score),
    reason
  }
}

/* ─────────────────────────────────────────────
   COUNTER CLAUSE GENERATOR
───────────────────────────────────────────── */
function getCounterClause(clause: string): string {
  const lower = clause.toLowerCase()

  if (/forfeit|forfeiture/.test(lower)) return COUNTER_CLAUSES['deposit forfeiture']
  if (/without.*notice|no notice/.test(lower)) return COUNTER_CLAUSES['no notice termination']
  if (/auto.*renew/.test(lower)) return COUNTER_CLAUSES['auto renewal']
  if (/all repair|maintenance.*lessee/.test(lower)) return COUNTER_CLAUSES['all repairs lessee']
  if (/sole discretion/.test(lower)) return COUNTER_CLAUSES['sole discretion']
  if (/enter.*without/.test(lower)) return COUNTER_CLAUSES['no notice entry']

  return COUNTER_CLAUSES['default']
}

/* ─────────────────────────────────────────────
   CLAUSE SIMPLIFIER (Plain English)
───────────────────────────────────────────── */
function simplifyClause(clause: string): string {
  const lower = clause.toLowerCase()

  if (/forfeit.*deposit/.test(lower))
    return 'If you leave early, you lose your entire security deposit.'

  if (/without.*notice/.test(lower))
    return 'Your landlord can take action without informing you beforehand.'

  if (/auto.*renew/.test(lower))
    return 'This agreement will automatically renew unless cancelled in advance.'

  if (/sole discretion/.test(lower))
    return 'The landlord alone has full control over this decision.'

  if (/all repair/.test(lower))
    return 'You must pay for all repairs, including major ones.'

  if (/sublet/.test(lower))
    return 'You cannot rent out the property without permission.'

  if (/utility/.test(lower))
    return 'You must pay all utility bills yourself.'

  return 'This clause contains legal terms that affect your rights.'
}

/* ─────────────────────────────────────────────
   HIGHLIGHT RISK WORDS (NEW FEATURE)
───────────────────────────────────────────── */
function highlightRisks(clause: string) {
  const riskyWords = [
    "forfeit",
    "without notice",
    "immediate",
    "sole discretion",
    "double rent",
    "no liability",
    "all repairs"
  ]

  let highlighted = clause

  for (const word of riskyWords) {
    const regex = new RegExp(word, "gi")
    highlighted = highlighted.replace(regex, `**${word.toUpperCase()}**`)
  }

  return highlighted
}

/* ─────────────────────────────────────────────
   MAIN API ROUTE
───────────────────────────────────────────── */
export async function POST(req: NextRequest) {
  const { clause, language = 'english' } = await req.json()

  if (!clause || clause.trim().length < 10) {
    return NextResponse.json(
      { error: 'Please provide a valid clause.' },
      { status: 400 }
    )
  }

  const HF_TOKEN = process.env.HF_API_TOKEN
  const HF_MODEL = process.env.HF_MODEL_ID

  let result: any = null

  /* ─────────────────────────────
     1. HF MODEL (optional)
  ───────────────────────────── */
  if (HF_TOKEN && HF_MODEL) {
    try {
      const SYSTEM_PROMPT = `You are ClauseClear. Return ONLY JSON:
{"plain":"","risk":"HIGH|MEDIUM|LOW","risk_reason":"","counter":""}`

      const prompt = `<start_of_turn>user\n${SYSTEM_PROMPT}\n\nClause:\n${clause}<end_of_turn>\n<start_of_turn>model\n`

      const hfResponse = await fetch(
        `https://api-inference.huggingface.co/models/${HF_MODEL}`,
        {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${HF_TOKEN}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            inputs: prompt,
            parameters: {
              max_new_tokens: 300,
              do_sample: false,
              return_full_text: false
            }
          }),
          signal: AbortSignal.timeout(15000)
        }
      )

      if (hfResponse.ok) {
        const data = await hfResponse.json()
        const text = Array.isArray(data)
          ? data[0]?.generated_text
          : data?.generated_text

        const jsonMatch = text?.match(/\{[\s\S]*\}/)

        if (jsonMatch) {
          try {
            result = JSON.parse(jsonMatch[0])
          } catch {
            result = null
          }
        }
      }
    } catch (err) {
      console.log("HF fallback triggered")
    }
  }

  /* ─────────────────────────────
     2. RULE ENGINE FALLBACK
  ───────────────────────────── */
  if (!result) {
    const analysis = detectRisk(clause)

    result = {
      plain: simplifyClause(clause),
      risk: analysis.risk,
      risk_score: analysis.score,
      risk_reason: analysis.reason,
      counter: getCounterClause(clause),
      highlighted_clause: highlightRisks(clause),
    }
  }

  /* ─────────────────────────────
     3. RESPONSE
  ───────────────────────────── */
  return NextResponse.json({
    ...result,
    insights: {
      score_out_of_100: result.risk_score ?? 0,
      summary:
        result.risk === "HIGH"
          ? "Strongly unfavorable to tenant"
          : result.risk === "MEDIUM"
          ? "Somewhat risky but negotiable"
          : "Generally fair clause"
    },
    language,
    source: HF_MODEL ? "model" : "engine"
  })
}
