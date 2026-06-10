# AGENTS.md

This directory is an Obsidian vault for fast preparation for the Deep Learning
oral exam. It contains compact and complete answer versions for the same exam
questions. The full study vault at `/home/mikhail/Projects/studies/dl` remains
the canonical source for complete explanations.

## Canonical Sources

- Use `DL_exam.pdf` as the canonical source for question wording.
- Use the matching detailed note in `/home/mikhail/Projects/studies/dl` as the
  source for complete explanations, formulas, diagrams, and pitfalls.
- Do not edit `/home/mikhail/Projects/studies/dl` from this vault.
- Preserve the meaning of every original question. Light cleanup of PDF line
  wrapping is allowed.
- Keep `complete-answers/` synchronized from `/home/mikhail/Projects/studies/dl`
  when complete explanations need to be refreshed.

## Vault Structure

Keep exactly the same topic folders and note filenames in both answer roots:

```text
light-answers/
  Topic 01 - DL Foundations and Training/
  Topic 02 - CNN Fundamentals/
  Topic 03 - CNN Architectures and Transfer Learning/
  Topic 04 - Segmentation and Pose/
  Topic 05 - Object Detection/
  Topic 06 - Knowledge Distillation/
  Topic 07 - Word Representations and Tokenization/
  Topic 08 - Recurrent and Seq2Seq Models/
  Topic 09 - Transformer Architecture/
  Topic 10 - NLP Pretraining and Model Families/
  Topic 11 - LLM Adaptation and Alignment/
  Topic 12 - LLM Inference and Augmentation/
  Topic 13 - Audio and Speech/
  Topic 14 - Vision Transformers/
  Topic 15 - Generative Models/
  Topic 16 - Graph Neural Networks/
  Topic 17 - Diffusion Models/
  Topic 18 - Multimodal Models/
complete-answers/
  Topic 01 - DL Foundations and Training/
  Topic 02 - CNN Fundamentals/
  Topic 03 - CNN Architectures and Transfer Learning/
  Topic 04 - Segmentation and Pose/
  Topic 05 - Object Detection/
  Topic 06 - Knowledge Distillation/
  Topic 07 - Word Representations and Tokenization/
  Topic 08 - Recurrent and Seq2Seq Models/
  Topic 09 - Transformer Architecture/
  Topic 10 - NLP Pretraining and Model Families/
  Topic 11 - LLM Adaptation and Alignment/
  Topic 12 - LLM Inference and Augmentation/
  Topic 13 - Audio and Speech/
  Topic 14 - Vision Transformers/
  Topic 15 - Generative Models/
  Topic 16 - Graph Neural Networks/
  Topic 17 - Diffusion Models/
  Topic 18 - Multimodal Models/
assets/
```

Every exam question gets exactly one Markdown file in `light-answers/` and one
matching Markdown file in `complete-answers/`. Use the global two-digit question
number as the filename prefix.

## Compact Note Template

Every note must use this structure:

```md
# Title

Source: `DL_exam.pdf`, Question NN

Original question:

> Russian question text from the PDF.

## Главная идея

One short intuitive paragraph.

## Минимум для ответа

Dense cheat-sheet bullets: definitions, contrasts, assumptions, practical use,
and limitations.

## Формулы / схема

Only the formulas, objectives, update rules, or pipeline steps that are likely
to be asked orally.

## Диаграмма

One compact Mermaid diagram.

## Уточнения экзаменатора

Three to five likely follow-up questions with short answers.

## Частые ошибки

Common traps and confusions.
```

## Compact Style

- Write in Russian.
- Keep standard English terms where they are expected: `attention`,
  `fine-tuning`, `dropout`, `IoU`, `KV-cache`, and similar.
- Target 250-400 words per note. Formula-heavy notes may reach 450 words.
- Prefer bullets over prose.
- Keep only exam-critical formulas and distinctions.
- Remove long derivations, historical detail, extended examples, and redundant
  caveats unless they are essential for oral correctness.
- Use Obsidian math delimiters:
  - Inline formulas: `$...$`
  - Display formulas: `$$...$$`
  - Do not use `\(...\)` or `\[...\]`.
- Each note must contain exactly one Mermaid block.
- Do not fetch external images in the compact pass.

## Agent Workflow

- Use subagents only for assigned, disjoint note-generation work.
- Current worker mode: `gpt-5.5`, `reasoning_effort: high`,
  `service_tier: priority`.
- Run at most 4 active subagents at a time.
- Each worker owns exactly one target note in this vault.
- Each worker must read this file and the matching detailed note in
  `/home/mikhail/Projects/studies/dl`.
- Workers must report:
  - question number,
  - changed path,
  - final approximate word count,
  - whether Mermaid was used,
  - any uncertainty.
- Completed agents should be closed promptly.

## Local Trainer App

- The local self-check trainer lives under `trainer/`.
- Use a Python virtual environment for trainer work:
  - create it with `python3 -m venv .venv`;
  - activate it with `source .venv/bin/activate`;
  - install dependencies with `pip install -r requirements.txt`.
- Run the trainer with:

```bash
uvicorn trainer.backend.main:app --host 127.0.0.1 --port 8000
```

- The trainer reads paired Markdown notes from `light-answers/` and
  `complete-answers/`. Do not change note content merely to support trainer UI
  behavior.
- Frontend formula and diagram rendering uses local vendor assets under
  `trainer/frontend/vendor/`. If `package.json` changes, refresh them with
  `npm install` and `npm run vendor`.
- Persistent self-rating data lives in local-only
  `trainer/data/progress.json`; do not commit personal progress. The committed
  empty template is `trainer/data/progress.example.json`.
- For tests that write progress, prefer a temporary file via
  `TRAINER_PROGRESS_PATH=/tmp/dl-trainer-test-progress.json`.
- Run `python -m trainer.backend.checks` after backend/parser changes.

## Quality Checklist

Before considering the compact vault complete, verify:

- Exactly 62 compact notes exist under `light-answers/`.
- Exactly 62 complete notes exist under `complete-answers/`.
- Topic folders and filenames match exactly between the two answer roots.
- Every compact note includes `Original question`.
- Every compact note includes the compact required headings.
- Every compact note has exactly one Mermaid block.
- Compact-note word counts are normally 250-400, with a 450-word ceiling for
  formula-heavy topics.
- No forbidden math delimiters appear in compact notes.
