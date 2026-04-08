import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'ClauseClear — AI Legal Clause Simplifier for Indian Renters',
  description: 'Understand your Indian rental agreement in plain English. Fine-tuned Gemma 3 AI flags risky clauses, detects missing protections, and suggests fairer counter-clauses.',
  keywords: 'Indian rental agreement, legal clause simplifier, tenant rights India, rental contract AI',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
