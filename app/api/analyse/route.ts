import { NextRequest, NextResponse } from 'next/server'

// Rule-based fallback engine (works without trained model)
const RISK_PATTERNS = {
  HIGH: [
    /forfeit(ed|ure).*deposit/i,
    /deposit.*forfeit/i,
    /sole discretion of the lessor/i,
    /without.*notice.*terminat/i,
    /immediately.*vacate/i,
    /no liability.*whatsoever/i,
    /all repairs.*lessee/i,
    /double.*rent.*holdover/i,
    /mesne profits/i,
    /without prior notice.*enter/i,
  ],
  MEDIUM: [
    /auto.*renew/i,
    /automatic.*renewal/i,
    /15%.*escalation/i,
    /arbitration.*final.*binding/i,
    /stamp duty.*lessee/i,
    /7 days.*non.payment/i,
    /address.*prior.*approval/i,
  ],
  LOW: [
    /fair wear and tear/i,
    /residential.*purposes/i,
    /society.*rules/i,
    /utility.*charges/i,
  ]
}

const COUNTER_CLAUSES: Record<string, string> = {
  'deposit forfeiture': 'Only one month\'s rent equivalent shall be deducted for early exit; the remainder shall be refunded within 30 days.',
  'no notice termination': 'Either party shall provide a minimum of 30 days written notice before termination of this agreement.',
  'auto renewal': 'This agreement expires on the due date. Renewal requires explicit written consent from both parties.',
  'all repairs lessee': 'Structural and major repairs (above Rs. 2,000) are the Lessor\'s responsibility. Day-to-day maintenance up to Rs. 500 is the Lessee\'s responsibility.',
  'sole discretion': 'Any deductions or decisions must be communicated in writing with itemised justification and supporting evidence.',
  'no notice entry': 'The Lessor shall provide at least 48 hours advance notice before entering the premises, except in genuine emergencies.',
  'default': 'This clause should be reviewed by both parties. Consider adding specific timelines, monetary caps, and written notice requirements.'
}

function detectRisk(clause: string): { risk: 'HIGH' | 'MEDIUM' | 'LOW', reason: string } {
  for (const pattern of RISK_PATTERNS.HIGH) {
    if (pattern.test(clause)) return { risk: 'HIGH', reason: 'This clause severely restricts tenant rights or removes standard legal protections.' }
  }
  for (const pattern of RISK_PATTERNS.MEDIUM) {
    if (pattern.test(clause)) return { risk: 'MEDIUM', reason: 'This clause contains terms that may disadvantage the tenant but are negotiable.' }
  }
  return { risk: 'LOW', reason: 'This clause is standard and does not significantly disadvantage the tenant.' }
}

function getCounterClause(clause: string): string {
  const lower = clause.toLowerCase()
  if (/forfeit|forfeiture/.test(lower)) return COUNTER_CLAUSES['deposit forfeiture']
  if (/without notice|no notice/.test(lower)) return COUNTER_CLAUSES['no notice termination']
  if (/auto.renew|automatic/.test(lower)) return COUNTER_CLAUSES['auto renewal']
  if (/all repair|all maintenance/.test(lower)) return COUNTER_CLAUSES['all repairs lessee']
  if (/sole discretion/.test(lower)) return COUNTER_CLAUSES['sole discretion']
  if (/enter.*without|without.*notice.*enter/.test(lower)) return COUNTER_CLAUSES['no notice entry']
  return COUNTER_CLAUSES['default']
}

function simplifyClause(clause: string): string {
  const lower = clause.toLowerCase()
  
  if (/forfeit.*deposit|deposit.*forfeit/.test(lower))
    return 'If you leave before the minimum period ends, you lose your entire security deposit — no exceptions.'
  if (/without.*notice.*terminat|terminat.*without.*notice/.test(lower))
    return 'Your landlord can terminate your lease and ask you to leave immediately, without any warning.'
  if (/auto.*renew|automatically.*renew/.test(lower))
    return 'This contract will automatically restart for another term unless you send written notice months in advance.'
  if (/sole discretion of the lessor/.test(lower))
    return 'The landlord alone decides on this matter — you have no say or recourse.'
  if (/all repair.*lessee|lessee.*all repair/.test(lower))
    return 'You must pay for ALL repairs and maintenance — even major structural damage — out of your own pocket.'
  if (/sublet|underlet/.test(lower))
    return 'You cannot rent out any part of the property to someone else without written permission from your landlord.'
  if (/utility|electricity|water|gas/.test(lower))
    return 'You must pay all utility bills (electricity, water, gas) entirely on your own.'
  if (/notice period|vacate.*notice/.test(lower))
    return 'Either party must give advance written notice before ending the rental agreement.'
  if (/stamp duty|registration charge/.test(lower))
    return 'You (the tenant) must pay all the legal registration and stamp duty costs for this agreement.'
  if (/mesne profits|double.*rent/.test(lower))
    return 'If you stay even one day past your agreement end date, your landlord can charge you double rent for that entire month.'
  if (/arbitration/.test(lower))
    return 'Any dispute must go to a private arbitrator instead of court — and their decision is final with no appeal.'
  if (/enter.*inspect|inspect.*enter/.test(lower))
    return 'Your landlord can enter and inspect your home at any time without giving you any notice in advance.'
  
  // Generic fallback
  const words = clause.split(' ')
  const simplified = clause
    .replace(/herein|hereof|thereof|thereto|hereinafter/gi, '')
    .replace(/demised premises/gi, 'rented property')
    .replace(/lessor/gi, 'landlord')
    .replace(/lessee/gi, 'tenant')
    .replace(/forthwith/gi, 'immediately')
    .replace(/howsoever/gi, 'however')
    .replace(/pursuant to/gi, 'according to')
  
  const sentences = simplified.split(/[.;]/).filter(s => s.trim().length > 20)
  return sentences[0]?.trim() || 'This clause contains legal terms that restrict tenant rights. Review carefully before signing.'
}

function detectHindi(clause: string, analysis: { plain: string; risk: string; counter: string }) {
  // Basic Hindi translations for key terms
  const translations: Record<string, string> = {
    HIGH: 'उच्च जोखिम',
    MEDIUM: 'मध्यम जोखिम', 
    LOW: 'कम जोखिम',
  }
  return {
    risk_hindi: translations[analysis.risk] || analysis.risk,
    plain_hint: 'पूर्ण हिंदी अनुवाद के लिए Hindi toggle दबाएं' // "Press Hindi toggle for full translation"
  }
}

export async function POST(req: NextRequest) {
  const { clause, language = 'english' } = await req.json()

  if (!clause || clause.trim().length < 10) {
    return NextResponse.json({ error: 'Please provide a valid clause.' }, { status: 400 })
  }

  // Try HF Inference API first (if model is deployed)
  const HF_TOKEN = process.env.HF_API_TOKEN
  const HF_MODEL = process.env.HF_MODEL_ID // e.g. "yourname/clauseclear-gemma3-270m"

  let result: { plain: string; risk: string; risk_reason: string; counter: string } | null = null

  if (HF_TOKEN && HF_MODEL) {
    try {
      const SYSTEM_PROMPT = `You are ClauseClear, an expert in Indian rental law. Analyse the given legal clause and respond ONLY in this exact JSON format:
{"plain":"<one clear sentence>","risk":"<HIGH|MEDIUM|LOW>","risk_reason":"<brief reason>","counter":"<fairer counter-clause>"}`

      const prompt = `<start_of_turn>user\n${SYSTEM_PROMPT}\n\nAnalyse this clause:\n${clause}<end_of_turn>\n<start_of_turn>model\n`

      const hfResponse = await fetch(
        `https://api-inference.huggingface.co/models/${HF_MODEL}`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${HF_TOKEN}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            inputs: prompt,
            parameters: { max_new_tokens: 300, do_sample: false, return_full_text: false }
          }),
          signal: AbortSignal.timeout(15000),
        }
      )

      if (hfResponse.ok) {
        const data = await hfResponse.json()
        const text = Array.isArray(data) ? data[0]?.generated_text : data?.generated_text
        if (text) {
          const jsonMatch = text.match(/\{[\s\S]*\}/)
          if (jsonMatch) result = JSON.parse(jsonMatch[0])
        }
      }
    } catch (e) {
      console.log('HF API unavailable, using fallback engine')
    }
  }

  // Fallback: rule-based engine
  if (!result) {
    const { risk, reason } = detectRisk(clause)
    result = {
      plain: simplifyClause(clause),
      risk,
      risk_reason: reason,
      counter: getCounterClause(clause),
    }
  }

  // Hindi translation
  let hindi = null
  if (language === 'hindi' || language === 'both') {
    const GOOGLE_KEY = process.env.GOOGLE_TRANSLATE_KEY
    if (GOOGLE_KEY) {
      try {
        const tRes = await fetch(
          `https://translation.googleapis.com/language/translate/v2?key=${GOOGLE_KEY}`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ q: [result.plain, result.counter], source: 'en', target: 'hi', format: 'text' })
          }
        )
        const tData = await tRes.json()
        hindi = {
          plain: tData.data?.translations?.[0]?.translatedText,
          counter: tData.data?.translations?.[1]?.translatedText,
        }
      } catch {}
    }
    if (!hindi) {
      hindi = { plain: '(Hindi अनुवाद: Google Translate API key सेट करें)', counter: null }
    }
  }

  return NextResponse.json({ ...result, hindi, source: HF_MODEL ? 'model' : 'engine' })
}
