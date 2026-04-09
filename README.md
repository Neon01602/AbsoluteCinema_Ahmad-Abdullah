# ⚖️ ClauseClear — AI Legal Clause Simplifier for Indian Renters

**Team Absolute Cinema · Kalpathon 2025**

Fine-tuned Gemma 3 270M (QLoRA) that decodes dense Indian rental legalese into plain English, flags risk levels, detects missing clauses, and suggests fairer counter-clauses.

---

## 🚀 Quick Start

### Step 1 — Train the Model (Google Colab)

1. Open `clauseclear_train.ipynb` in Google Colab (T4 GPU, free tier works)
2. Set your HuggingFace token at cell 2
3. Accept Gemma 3 license at https://huggingface.co/google/gemma-3-270m-it
4. Run all cells (~20-30 min on T4)
5. Uncomment cell 10 to push your fine-tuned model to HF Hub
6. Note your model ID: `your-username/clauseclear-gemma3-270m`

### Step 2 — Deploy to Vercel

```bash
# Clone / copy this project
cd clauseclear-web

# Install dependencies
npm install

# Test locally
cp .env.example .env.local
# Edit .env.local with your tokens
npm run dev
```

**Deploy to Vercel:**
```bash
npm install -g vercel
vercel --prod
```

Or via Vercel Dashboard:
1. Push to GitHub
2. Import repo at vercel.com
3. Add environment variables (see below)

### Step 3 — Set Environment Variables in Vercel

Go to: Vercel Dashboard → Your Project → Settings → Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_API_TOKEN` | ✅ Yes | HuggingFace API token |
| `HF_MODEL_ID` | ✅ Yes | Your fine-tuned model (e.g. `yourname/clauseclear-gemma3-270m`) |
| `GOOGLE_TRANSLATE_KEY` | Optional | For Hindi translation |

---

## 🏗️ Architecture

```
User Input (Text / PDF)
        ↓
Next.js API Routes (Vercel Edge)
        ↓
1. HF Inference API → Fine-tuned Gemma 3 270M
   (fallback: Rule-based engine if model unavailable)
        ↓
2. Missing Clause Detector (rule-based)
        ↓
3. Google Translate (optional Hindi)
        ↓
React UI with risk badges + counter-clauses
```

## 🔧 Five Core Features

| Feature | Implementation |
|---|---|
| 🇮🇳 Hindi Output | Google Translate API (optional) |
| 🔴 Risk Scorer | Fine-tune output + pattern fallback |
| 📂 PDF Health Report | `pdf-parse` → clause splitter → batch analysis |
| 🔍 Missing Clause Detector | 6-pattern rule engine |
| 💬 Counter-Clause Generator | Fine-tune output + template fallback |



## 🌐 Works Without the Trained Model

The app includes a sophisticated rule-based fallback engine that:
- Detects 10+ high-risk clause patterns
- Recognises 7 medium-risk patterns  
- Provides contextual counter-clauses
- Detects 6 types of missing protective clauses

This means the Vercel app is fully functional even before the model is trained.

---

*Built for India's 50M+ renters who sign agreements they can't understand.*
