# Lab-LLM-Daniel — N-Gram Language Models from Scratch

**Course:** Large-Scale Language Models — UFCG
**Professor:** Leandro B. Marinho
**Student:** Daniel Anderson de Souza Silva
**Deliverable:** `atividade_ngramas.ipynb` — commit `034f9ab` *"completion of the activity"* ([view on GitHub](https://github.com/DanielAndersonTI/Lab-LLM-Daniel/commit/034f9ab), already pushed to `origin/main`)

A **from-scratch** implementation — no NLTK, spaCy, KenLM, etc., only the Python standard library (`os`, `re`, `math`, `random`, `urllib.request`, `collections`) plus `matplotlib` for the plots — of **unigram, bigram, and trigram** language models, trained on a **Project Gutenberg** book and compared by **perplexity**: MLE, Laplace/add-k smoothing, tuning `k` on the validation set, **linear interpolation**, and **Shannon-style text generation**. Item 8 (bonus) evaluates the models trained on *Dom Casmurro* on a second book, *Iracema*.

## Table of Contents

- [Deliverable](#deliverable)
- [Repository structure](#repository-structure)
- [How to run](#how-to-run)
- [Notebook pipeline (sections 0–9)](#notebook-pipeline-sections-09)
- [Main results](#main-results)
- [Reproducibility](#reproducibility)
- [Known limitations](#known-limitations)
- [AI usage statement](#ai-usage-statement)
- [Tested environment](#tested-environment)

## Deliverable

`atividade_ngramas.ipynb` — a notebook **executed end to end**: 35 cells (17 of code), `execution_count` from 1 to 17, **0 errors**, and **2 plots** saved.

| Assignment requirement | Status in the notebook |
|---|---|
| TODOs in sections 5.1, 5.2, 5.3, 6, and 7 implemented | ✅ no cell with `NotImplementedError` |
| "Verification — do not change" cells intact and passing | ✅ 4 cells printing `OK` |
| Analysis questions (2.1 and 1–8) in text cells | ✅ sections *2.1 Zipf's Law* and *8*, with the measured numbers |
| AI usage statement | ✅ section 9 (mandatory — 5% of the grade) |
| No NLP libraries | ✅ only the standard library + `matplotlib` |

Reading versions of the **same executed notebook** (open without Jupyter, with outputs and plots):

- `atividade_ngramas.html` — self-contained HTML (plots embedded in base64);
- `atividade_ngramas.pdf` — 22 pages, generated from the HTML.

## Repository structure

| File | Size | Description |
|---|---|---|
| `atividade_ngramas.ipynb` | ≈ 175 KB | Activity notebook: code, outputs, and answers |
| `atividade_ngramas.html` | ≈ 550 KB | Reading version (self-contained HTML) |
| `atividade_ngramas.pdf` | ≈ 1.0 MB | Reading version in PDF (22 pages) |
| `livro.txt` | ≈ 408 KB | *Dom Casmurro* — local cache of the **already-cleaned** text (381,375 characters) |
| `livro_bonus.txt` | ≈ 208 KB | *Iracema* — local cache used in item 8 (bonus) |
| `Tasks-LLM` | ≈ 4.6 KB | Development logbook, step by step |
| `.Rhistory` | 0 KB | Environment leftover, empty |

> **Maintenance note:** a duplicated copy of the notebook (`atividade_ngramas(1).ipynb`) is no longer versioned as of commit `034f9ab`. The Git history still contains it, should it be needed for recovery (`git show 0f8688f:"atividade_ngramas(1).ipynb"`).

## How to run

**Prerequisites**

- Python **3.12** (tested with 3.12.0) and **matplotlib** 3.10.9 — the only third-party dependency, used only in the plots, as the assignment allows.
- To run without opening the notebook: **nbconvert** 7.17.1 + **nbclient** 0.10.4, with the `python3` kernelspec registered.
- Internet is **optional**: the notebook reads `livro.txt` from disk and only downloads from Gutenberg if the file does not exist (same logic for `livro_bonus.txt`).

**Option 1 — VS Code (to read or edit)**

1. Open `atividade_ngramas.ipynb` in VS Code (the **Jupyter** extension) and pick the **Python 3** kernel.
2. Use **Run All**. The outputs are already saved in the file, so the notebook opens with the results visible even without executing.

**Option 2 — command line (headless execution, equivalent to Run All)**

```bash
python -m nbconvert --to notebook --execute --inplace atividade_ngramas.ipynb
```

Runs all cells in order and **rewrites the outputs into the `.ipynb` itself** (takes about 1 minute).

**Regenerate the reading versions**

```bash
python -m nbconvert --to html --output atividade_ngramas atividade_ngramas.ipynb
& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless=new --disable-gpu `
    --no-pdf-header-footer --print-to-pdf="atividade_ngramas.pdf" `
    "file:///C:/Users/Daniel/OneDrive/I.T/UFCG/2_Semestre/LLM/Lab-LLM-Daniel/atividade_ngramas.html"
```

(any Chromium browser works for the second command; adjust the `file:///…` if the repository is at a different path)

> **Environment details:** on this machine the `jupyter` executable **is not on the `PATH`** — always use `python -m nbconvert …`. Outside Jupyter, configure `matplotlib` with the non-interactive backend (`matplotlib.use("Agg")`) when running notebook snippets as a script.

## Notebook pipeline (sections 0–9)

| Section | What it does | Recorded result |
|---|---|---|
| 0 | Imports and `SEED = 42` | environment loaded |
| 1 | Downloads and cleans *Dom Casmurro* (Gutenberg **55752**), removing the license header and footer | 400,944 → **381,375** characters |
| 2 | Sentence segmentation by regex and tokenization (lowercase, accents preserved, internal hyphen kept) | **4,564** sentences, **65,775** tokens, **9,414** types |
| 2.1 | Zipf's Law: log(frequency) × log(rank) with linear fit | slope **b ≈ −1.000**; 5,296 *hapax*; most frequent word: `que` (2,663) |
| 3 | Random train/validation/test split and OOV handling with `<UNK>` (`MIN_FREQ = 2`) | 3,651 / 456 / 457 sentences; vocabulary of 3,500 words + `<UNK>` + `</s>`; OOV in test **14.35%** |
| 4 | `NGramLM` class: `<s>` / `</s>` padding, counts C(ctx, w) and C(ctx), MLE, add-k smoothing, log-probability per sentence, and perplexity | verification with the slides' mini-corpus (7 probability assertions; sums equal to 1 for k = 0, 1, and 0.1) |
| 5 | Perplexity: 5.1 MLE (k = 0), 5.2 Laplace (k = 1), and 5.3 search for `k` **on the validation set** | see table in *Main results* |
| 6 | Linear interpolation with grid search (λ₁ > 0, step 0.1) | best λ = **(0.5; 0.4; 0.1)**; PP(validation) = 141.99 |
| 6.1 | Summary table and bar chart of test perplexity | plot saved in the cell |
| 7 | Shannon-style text generation (3 sentences per order) | 1-gram = isolated words; 2-gram = short sentences; 3-gram = coherent passages (with literal copies from the book) |
| 8 | Analysis questions 1 to 7 (2.1 is in section 2) | answers in text cells, with the measured numbers |
| 8 (bonus) | Swaps the corpus for *Iracema* (Gutenberg **67740**) and re-evaluates the *Dom Casmurro* models | see bonus table below |
| 9 | AI usage statement | required by the assignment |

## Main results

**Perplexity (train / test), corpus *Dom Casmurro***

| Model | PP(train) | PP(test) |
|---|---|---|
| unigram MLE (k = 0) | 348.84 | 252.34 |
| unigram Laplace (k = 1) | 350.87 | 256.67 |
| unigram add-k (k = 0.001) | 348.84 | 252.34 |
| bigram MLE (k = 0) | 32.20 | **∞** |
| bigram Laplace (k = 1) | 610.97 | 643.21 |
| bigram add-k (k = 0.01) | 60.92 | 241.66 |
| trigram MLE (k = 0) | 4.54 | **∞** |
| trigram Laplace (k = 1) | 1,281.37 | 1,981.14 |
| trigram add-k (k = 0.005) | 24.49 | 886.84 |
| **linear interpolation** (λ = 0.5 / 0.4 / 0.1) | — | **138.95** |

Readings:

- **MLE zeroes out n-grams never seen** in training, so bigram and trigram perplexity goes to **infinity** on the test set. The unigram does not suffer from this because its vocabulary is closed (`|V|` = 3,500 words): every test token is either in it or becomes `<UNK>`.
- With **Laplace (k = 1)** the "trigram < bigram < unigram" order from the slides **does not hold**: the synthetic mass `k · |V|` is enormous compared to the real counts and the effect worsens with order. It is a portrait of **sparsity**: with a vocabulary of ~3.5 thousand words there are ~4 × 10¹⁰ possible trigrams, against ~66 thousand observed in training.
- The best `k` on validation landed far below 1 (**0.001** for the 1-gram, **0.01** for the 2-gram, and **0.005** for the 3-gram): the sparser the model, the more expensive it is to distribute synthetic mass.
- **Linear interpolation is the best model on the test set (138.95)** and concentrates the largest weight on the **unigram (λ₁ = 0.5)** — a sign that the higher-order models are unreliable on this corpus.
- On question 6 ("is the trigram copying the book?"): verifying copying requires comparing the generated n-grams against the **set of training n-grams**, not just counting repeated words — what can be shown is that several of the 3-gram passages appear literally in the book.

**Bonus (item 8) — models trained on *Dom Casmurro* evaluated on *Iracema***

| Model | PP(Dom Casmurro) | PP(Iracema) | PP(DC, known words only) | PP(Iracema, known only) |
|---|---|---|---|---|
| 1-gram MLE | 252.34 | 113.96 | 405.86 | 314.76 |
| 2-gram MLE | ∞ | ∞ | ∞ | ∞ |
| interpolation | 138.95 | 93.72 | 203.76 | **236.56** |

- Without handling the vocabulary, PP *appears* to drop on Iracema because **33.15%** of tokens become a single `<UNK>` symbol (against 14.35% on the *Dom Casmurro* test set) — it is an artifact of `<UNK>`, not an improvement of the model.
- **Restricting to known words**, the interpolation trained on *Dom Casmurro* goes from 203.76 to **236.56** on *Iracema* (**+16%**): the model loses performance on another author/era, exactly the point of the slide *"The Wall Street Journal is not Shakespeare"*.

## Reproducibility

- `SEED = 42` fixes the train/validation/test split, and **all metrics** (perplexities, best `k`, and λ) are reproducible across runs.
- **Exception:** the **texts in section 7** are not reproducible byte for byte, for two reasons — `gerar()` creates the `random.Random(SEED)` **only once, at function definition** (the state is consumed on each call, so the result depends on how many calls came before), and it iterates over `modelo.vocab`, which is a `set` (non-deterministic order across processes). To make generation deterministic: move `Random(SEED)` inside the function and iterate over `sorted(modelo.vocab)`. The choice was to **keep the code as is** and preserve the outputs from the reference run.
- The notebook was executed in headless mode, so the `execution_count` values from 1 to 17 record the actual execution order and the outputs are **written into the `.ipynb`** — anyone opening it sees the results without running anything.
- Book downloads only happen if `livro.txt` / `livro_bonus.txt` are not in the folder; since both are versioned, the notebook runs **offline**.

## Known limitations

- **Simple regex segmentation:** abbreviations (`Sr.`, `D.`) and dialogue dashes have no special handling (discussed in answer 7).
- **Tokenization drops everything that is not a letter** (or an internal hyphen): the 55 groups of digits in the text (134 digits) disappear, and hyphenated words become a single token.
- **A single `<UNK>`** flattens the diversity of rare words — this is what distorts the corpus comparison in the bonus (hence the "known words only" columns).
- Generation stops when the context does not appear in training and limits sentences to 30 words, which makes the trigram texts short.
- The corpus is small (~381 thousand characters) → sparse counts and high perplexity; the conclusions hold for this corpus.

## AI usage statement

**Section 9 of the notebook** contains the declaration required by the assignment (5% of the grade): tool used (AI assistant in VS Code, with a Claude-family model via API), parts where AI was employed (implementation of the TODOs in sections 5.1, 5.2, 5.3, 6, and 7; headless execution of the notebook; extra measurements used in the answers; drafts of the analysis answers), examples of prompts, what had to be corrected, and three cases where the AI got it wrong. As per the activity rules, **all verification cells pass** and the code can be explained passage by passage.

## Integrity of the delivered files

SHA-256 of the files in commit `034f9ab`, to check that the file opened is the one delivered:

| File | SHA-256 |
|---|---|
| `atividade_ngramas.ipynb` | `A604D89ED6108C8509A0DC21E6FE594C0DCFB52355206A1D65E7B55D17184DE6` |
| `atividade_ngramas.html` | `D23AEBBB4714A2CFB0EB5702B77D9B8E7B4DAF99ACB98CD98654C2088D9177C1` |
| `atividade_ngramas.pdf` | `D908B3419495AD7765E48B419D12744A60F69E59EEED5E12E670DBD4087A13C3` |

```powershell
Get-FileHash atividade_ngramas.ipynb -Algorithm SHA256
```

## Tested environment

| Item | Version |
|---|---|
| System | Windows 10 (10.0.19045), PowerShell 5.1 |
| Python | 3.12.0 (kernelspec `python3`) |
| matplotlib | 3.10.9 |
| nbconvert / nbclient | 7.17.1 / 0.10.4 |
| nbformat / jupyter_client / jupyter_core | 5.10.4 / 8.8.0 / 5.9.1 |
| NLP libraries | none (forbidden by the assignment) |
| NumPy / pandas | not used directly |

## References

- *Dom Casmurro*, Machado de Assis — Project Gutenberg, ID **55752**: `https://www.gutenberg.org/cache/epub/55752/pg55752.txt`
- *Iracema*, José de Alencar — Project Gutenberg, ID **67740** (bonus): `https://www.gutenberg.org/cache/epub/67740/pg67740.txt`
- Course slides *Modelos de Linguagem N-Grama*: mini-corpus `I am Sam` (section 4.1), perplexity table 962 / 170 / 109 (question 2), and the slide *"The Wall Street Journal is not Shakespeare"* (bonus).
- `python -m nbconvert --help` — headless execution and HTML export.

---

**Submission:** the repository is the submission channel (commit + push to `origin/main`); the assignment requires the notebook with the TODOs implemented, the verification cells passing, the analysis answers, and the AI usage statement — all in `atividade_ngramas.ipynb`.