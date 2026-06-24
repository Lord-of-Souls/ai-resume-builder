"""
project_selector.py

Pick the K projects most relevant to a specific job description, so each
generated resume shows a tailored subset instead of the full portfolio.

Integration (in app.py):

    import copy
    from project_selector import select_relevant_projects

    # `resume` is the parsed master_resume.json dict
    # `job_description` is the text you already extract from the LinkedIn post
    tailored = copy.deepcopy(resume)
    tailored["projects"] = select_relevant_projects(
        job_description,
        resume["projects"],
        k=4,
        generate_fn=generate_with_retry,   # <-- your EXISTING retry-wrapped call
    )

    # Feed `tailored` (not `resume`) into your resume-generation prompt and PDF step.

`generate_fn` should be a callable that takes a single prompt string and
returns the model's text response. If you don't pass one, the module falls
back to its own google-genai call with the same flash -> flash-lite + backoff
pattern you're already using.
"""

import json
import time

DEFAULT_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]


def _fallback_generate(prompt: str, models=None, max_retries: int = 4) -> str:
    """Minimal self-contained generate with model fallback + 503 backoff.

    Only used if you don't pass your own generate_fn. Prefer passing your
    existing retry helper so behavior stays consistent across the app.
    """
    from google import genai

    models = models or DEFAULT_MODELS
    client = genai.Client()
    last_err = None
    for model in models:
        delay = 1.0
        for _ in range(max_retries):
            try:
                resp = client.models.generate_content(model=model, contents=prompt)
                return resp.text or ""
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e).lower()
                # Retry only on overload/unavailable; otherwise move to next model.
                if "503" in msg or "overloaded" in msg or "unavailable" in msg:
                    time.sleep(delay)
                    delay *= 2
                    continue
                break
    raise RuntimeError(f"All models failed in project selection: {last_err}")


def select_relevant_projects(job_description, projects, k=4, generate_fn=None):
    """Return the k projects most relevant to job_description.

    - Returns a subset of the ORIGINAL project dicts, unchanged, ordered most
      to least relevant.
    - If there are <= k projects, returns them all (no model call needed).
    - On any parsing/model failure, falls back to the first k projects so the
      resume generation never breaks.
    """
    if not projects:
        return []
    if len(projects) <= k:
        return list(projects)

    generate = generate_fn or _fallback_generate

    # Compact, index-addressable catalog so the model returns indices, not prose.
    catalog = [
        {
            "index": i,
            "name": p.get("name", ""),
            "stack": p.get("stack", []),
            "description": p.get("description", ""),
        }
        for i, p in enumerate(projects)
    ]

    prompt = (
        "You are screening a candidate's project portfolio against one job.\n\n"
        "JOB DESCRIPTION:\n"
        f"{job_description}\n\n"
        "CANDIDATE PROJECTS (JSON):\n"
        f"{json.dumps(catalog, ensure_ascii=False, indent=2)}\n\n"
        f"Select exactly the {k} projects most relevant to this job, judged by "
        "overlap of skills, tools, methods, and domain with the job description. "
        "Prefer projects that demonstrate what the posting actually asks for.\n"
        f"Return ONLY a JSON array of the chosen \"index\" integers, ordered most "
        "to least relevant. No prose, no markdown, no code fences.\n"
        "Example: [3, 0, 7, 5]"
    )

    try:
        raw = generate(prompt) or ""
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        idxs = json.loads(cleaned)

        seen = set()
        chosen = []
        for i in idxs:
            if isinstance(i, int) and 0 <= i < len(projects) and i not in seen:
                seen.add(i)
                chosen.append(i)

        if not chosen:
            raise ValueError("model returned no valid indices")

        # Top up from the front if the model returned fewer than k.
        for i in range(len(projects)):
            if len(chosen) >= k:
                break
            if i not in seen:
                seen.add(i)
                chosen.append(i)

        return [projects[i] for i in chosen[:k]]

    except Exception:
        # Never let selection break the pipeline.
        return list(projects[:k])


if __name__ == "__main__":
    # Tiny smoke test with the fallback path stubbed out.
    sample = [
        {"name": "SQL Integration", "stack": ["SQL"], "description": "databases"},
        {"name": "Data Storytelling", "stack": ["Plotly"], "description": "viz"},
        {"name": "A/B Testing", "stack": ["Python"], "description": "experiments"},
        {"name": "Web App Deploy", "stack": ["Linux"], "description": "deployment"},
        {"name": "Automation", "stack": ["Python"], "description": "scripts"},
    ]
    picked = select_relevant_projects(
        "Looking for SQL and dashboarding skills",
        sample,
        k=2,
        generate_fn=lambda _p: "[0, 1]",
    )
    print([p["name"] for p in picked])  # -> ['SQL Integration', 'Data Storytelling']
