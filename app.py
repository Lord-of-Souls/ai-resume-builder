import re
import json
import time

import requests
import streamlit as st
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from markdown_pdf import MarkdownPdf, Section

from project_selector import select_relevant_projects

# =====================================================================
#  STEP 1 — GET THE JOB DESCRIPTION FROM A LINKEDIN URL  (free, no Apify)
# =====================================================================
#
#  Every LinkedIn job has a numeric ID in its URL, e.g.
#     https://www.linkedin.com/jobs/view/4012345678/
#     https://www.linkedin.com/jobs/search/?currentJobId=4012345678
#  LinkedIn exposes a login-free "guest" page for each job here:
#     https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/<ID>
#  We fetch that page and read the title / company / description out of it.
# ---------------------------------------------------------------------

def extract_job_id(url: str) -> str | None:
    """Pull the numeric job ID out of any LinkedIn job URL."""
    # Case 1: ...?currentJobId=4012345678
    m = re.search(r"currentJobId=(\d+)", url)
    if m:
        return m.group(1)
    # Case 2: .../jobs/view/4012345678  or  .../jobs/view/some-title-4012345678
    m = re.search(r"/jobs/view/(?:[^/?]*-)?(\d+)", url)
    if m:
        return m.group(1)
    # Case 3: last resort — any long number in the URL (job IDs are ~10 digits)
    m = re.search(r"(\d{8,})", url)
    return m.group(1) if m else None


def scrape_linkedin_job(job_url: str):
    """Return (title, company, description) using LinkedIn's free guest endpoint."""
    job_id = extract_job_id(job_url)
    if not job_id:
        raise ValueError(
            "Could not find a job ID in that URL. Make sure you copied the full "
            "LinkedIn job link (it should contain a long number)."
        )

    guest_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
    headers = {
        # A normal browser User-Agent makes LinkedIn far more likely to answer.
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    resp = requests.get(guest_url, headers=headers, timeout=20)
    if resp.status_code == 429:
        raise RuntimeError(
            "LinkedIn temporarily rate-limited this IP (HTTP 429). Wait a few "
            "minutes, or paste the description manually below."
        )
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    def first_text(*selectors):
        for sel in selectors:
            el = soup.select_one(sel)
            if el and el.get_text(strip=True):
                return el.get_text(" ", strip=True)
        return ""

    title = first_text("h2.top-card-layout__title", ".topcard__title") or "Target Role"
    company = first_text(".topcard__org-name-link", ".topcard__flavor") or "Target Company"

    desc_el = soup.select_one(".show-more-less-html__markup, .description__text")
    description = desc_el.get_text("\n", strip=True) if desc_el else ""

    if not description:
        raise RuntimeError(
            "The page loaded but no description was found (LinkedIn may have "
            "changed the layout or shown a login wall). Paste it manually below."
        )

    return title, company, description


# =====================================================================
#  STEP 2 — TAILOR THE RESUME WITH GEMINI  (new google-genai SDK)
# =====================================================================

def _call_gemini(gemini_key: str, prompt: str, prefer_pro: bool = False) -> str:
    """Send a prompt to Gemini with retry/backoff and an automatic model fallback.

    When prefer_pro=True, the call starts on gemini-2.5-pro with extended thinking
    enabled, then automatically falls back to flash -> flash-lite if Pro is
    overloaded (503), rate-limited/unavailable on your tier (429/permission), or
    otherwise fails. The cheaper models are the failsafe, so a Pro outage never
    blocks the app.
    """
    client = genai.Client(api_key=gemini_key)

    if prefer_pro:
        models_to_try = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
    else:
        models_to_try = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

    last_error = None

    for model_name in models_to_try:
        # Enable extended thinking only on Pro; flash thinks dynamically by default
        # and flash-lite stays fast.
        if model_name == "gemini-2.5-pro":
            config = types.GenerateContentConfig(
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_budget=8192),
            )
        else:
            config = types.GenerateContentConfig(temperature=0.2)

        for attempt in range(4):  # waits: 1s, 2s, 4s, then give up on this model
            try:
                response = client.models.generate_content(
                    model=model_name, contents=prompt, config=config
                )
                return response.text
            except Exception as e:
                last_error = e
                transient = any(
                    code in str(e) for code in ("503", "UNAVAILABLE", "429", "overload")
                )
                if transient and attempt < 3:
                    time.sleep(2 ** attempt)  # 1, 2, 4 seconds
                    continue
                break  # non-transient error or out of retries -> next (cheaper) model

    raise RuntimeError(f"Gemini is still unavailable after retries: {last_error}")



def analyze_keyword_match(gemini_key: str, job_desc: str, resume_text: str):
    """Compare the job description to the tailored resume.

    Returns (matched, missing): two lists of the most important ATS keywords from
    the job — the ones already represented in the resume, and the ones still absent.
    Fails soft (returns empty lists) so the bonus report can never break the main
    resume flow.
    """
    prompt = f"""You are an ATS (applicant tracking system) keyword analyzer.

From the JOB DESCRIPTION, extract the 10-15 most important hard skills, tools, and
qualifications a recruiter or ATS would screen for. Then decide, for each one,
whether it appears in the TAILORED RESUME (match by meaning, not just exact string).

Return ONLY valid JSON — no prose, no explanation, no code fences — in exactly
this shape:
{{"matched": ["keyword", "..."], "missing": ["keyword", "..."]}}

- "matched": important job keywords that ARE represented in the resume.
- "missing": important job keywords that are NOT in the resume.

JOB DESCRIPTION:
{job_desc}

TAILORED RESUME:
{resume_text}
"""
    try:
        raw = _call_gemini(gemini_key, prompt).strip()
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
        data = json.loads(raw)
        return list(data.get("matched", [])), list(data.get("missing", []))
    except Exception:
        return [], []


def generate_cover_letter_guide(
    gemini_key: str, master_json: str, job_desc: str, company: str, title: str
) -> str:
    """Return a Markdown BRIEF that helps the candidate write their OWN cover letter.

    Not the letter itself — directions, company/role points to know, the candidate's
    strongest angles for this job, and a few writing tips. Fails soft (returns "").
    """
    prompt = f"""You are a career coach. Help the candidate write their OWN cover
letter — do NOT write the letter for them. Produce a concise, skimmable BRIEF in
Markdown.

STRICT RULES
* Use ONLY facts found in the CANDIDATE DATA and the JOB DESCRIPTION.
* For company facts, use ONLY what the JOB DESCRIPTION actually states. If something
  useful is NOT in the description (mission, products, recent news, values), list it
  as a "Research:" item to look up — NEVER invent or guess company facts.
* Pull the candidate's strengths ONLY from the CANDIDATE DATA. Never invent skills.
* Do not write any part of the actual cover letter or any sample paragraphs.

Output exactly these four sections, with these headings:

### What this role is really about
2-3 bullets on the role's core priorities and what the employer most values, based
on the job description.

### What to know before you write
Key points from the job description worth referencing. Add "Research:" bullets for
important things the description does not cover.

### Your strongest angles for this job
The 3-4 most relevant items from the candidate's data for THIS role, each with a
short note on how to frame it. Candidate data only.

### Writing tips
3-4 specific, actionable tips for this particular letter — what to open with, tone,
what to avoid, and target length.

Keep it tight. Do NOT write the cover letter itself.

---
CANDIDATE DATA (JSON):
{master_json}

---
JOB DESCRIPTION:
{job_desc}

---
ROLE: {title} at {company}
"""
    try:
        return _call_gemini(gemini_key, prompt, prefer_pro=True).strip()
    except Exception:
        return ""


def generate_resume_markdown(gemini_key: str, master_json: str, job_desc: str) -> str:
    prompt = f"""
You are an expert resume writer. Produce a one-page resume tailored to the job
description below, using the Candidate Data as the only source of facts.

CRITICAL RULES
* FACTUAL ACCURACY: Use only facts present in the Candidate Data. Never invent or
  exaggerate skills, metrics, or experience.
* PROJECTS ARE PRE-SELECTED: The Candidate Data already contains only the projects
  chosen for this job. Include ALL projects provided — do not drop any.
* COMPLETENESS — DO NOT PRUNE LISTS: In SKILLS & TOOLS, CERTIFICATIONS & AWARDS,
  EDUCATION, and LANGUAGES, include EVERY item from the Candidate Data. You may
  reorder so the most job-relevant items lead, but NEVER omit a skill, tool,
  language, methodology, certification, award, or degree. These sections are scanned
  by ATS — completeness matters more than brevity here.
* TAILORING: Tailor only the INTRODUCTION and the project descriptions to the job,
  and reorder skills so relevant ones come first. Do not shorten the credential lists.
* MIRROR THE JOB'S WORDING: When the candidate genuinely has a skill the JOB
  DESCRIPTION names, use the job's EXACT terminology instead of a synonym (e.g., if
  the job says "data visualization", write "data visualization", not "charts").
  NEVER invent or imply a skill the candidate lacks just to match wording — factual
  accuracy always overrides mirroring.
* NO HYPERLINKS: Write every URL as plain text exactly like "github.com/user/repo".
  NEVER use Markdown link syntax [text](url) and NEVER wrap URLs in <angle brackets>.
* OUTPUT: Return raw Markdown only. Do NOT wrap it in ```markdown fences.

FOLLOW THIS EXACT STRUCTURE. Put a BLANK LINE between EVERY entry — every project,
every education line, every certification, every award, every skill line, every
language — so each renders on its own line. Use the heading levels exactly as shown
(the name is #, the two contact lines are ####, sections are ##).

# FULL NAME IN CAPITALS

#### City, State | email | phone

#### linkedin url | github url

## INTRODUCTION

A tailored 3-4 sentence professional summary.

## PRACTICAL PROJECTS

**1) Project Title**

One or two sentences tailored to the job. Project Link: plain-text url

**2) Project Title**

One or two sentences tailored to the job. Project Link: plain-text url

(repeat for EVERY project in the Candidate Data, blank line between each)

## EDUCATION

**Program / Degree** — Institution | dates

**Program / Degree** — Institution | dates

(one line per education entry, blank line between each — never run them together)

## CERTIFICATIONS & AWARDS

List EVERY certification and EVERY award from the Candidate Data, each on its own
line, blank line between each. Include all academic olympiad medals and honors.

## SKILLS & TOOLS

**Programming Languages:** every item, comma-separated

**Frameworks & Libraries:** every item

**Tools & Environment:** every item

**Methodologies:** every item

**Soft Skills:** every item, separated by " | "

## LANGUAGES

**Portuguese:** ...

**English:** ...

(one line per language, blank line between each)

---
CANDIDATE DATA (JSON):
{master_json}

---
JOB DESCRIPTION:
{job_desc}
"""
    return _call_gemini(gemini_key, prompt, prefer_pro=True)


# =====================================================================
#  STEP 3 — MARKDOWN -> PDF
# =====================================================================

# One CSS block for the whole resume (single PDF section -> no page break).
# The name is an h1 and the two contact lines are h4 headings; both are centered.
# The body never uses h1/h4, so centering those does not affect anything below.
# PyMuPDF (used by markdown-pdf) supports this subset of CSS. The `a` rule strips
# hyperlink styling if a link ever slips through.
RESUME_CSS = """
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt;
       color: #1a1a1a; line-height: 1.35; }
h1 { font-size: 20pt; text-align: center; margin: 0 0 2px 0; }
h4 { font-size: 10pt; font-weight: normal; text-align: center; color: #333333;
     margin: 1px 0; }
h2 { font-size: 12pt; margin: 12px 0 5px 0; padding-bottom: 2px;
     border-bottom: 1.5px solid #333333; }
p  { margin: 3px 0; }
strong { color: #000000; }
a { color: inherit; text-decoration: none; }
"""


def strip_hyperlinks(md: str) -> str:
    """Safety net: if Gemini ignores the rule, turn any links back into plain text."""
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)   # [text](url) -> text
    md = re.sub(r"<((?:https?://)?[^>\s]+)>", r"\1", md)  # <url> -> url
    return md


def convert_markdown_to_pdf(markdown_text: str, output_filename: str) -> str:
    markdown_text = strip_hyperlinks(markdown_text).strip()
    # Single section = single continuous flow = no forced page break.
    pdf = MarkdownPdf(toc_level=0)
    pdf.add_section(Section(markdown_text), user_css=RESUME_CSS)
    pdf.save(output_filename)
    return output_filename


# =====================================================================
#  STREAMLIT UI
# =====================================================================

st.set_page_config(page_title="AI Resume Builder", page_icon="📄", layout="centered")
st.title("🚀 Automated AI Resume Builder")
st.markdown(
    "Paste a LinkedIn job URL. The app pulls the description for free, tailors "
    "your master resume with Gemini, and gives you a PDF."
)

with st.sidebar:
    st.header("🔑 Configuration")
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    st.caption("Get a free key at aistudio.google.com/apikey")
    st.divider()
    st.caption("Keep `master_resume.json` in the same folder as this app.")

job_url = st.text_input(
    "🔗 LinkedIn Job URL",
    placeholder="https://www.linkedin.com/jobs/view/...",
)

# Manual fallback: shown collapsed, used only if scraping fails / is left filled.
with st.expander("✏️ Or paste the job description manually (fallback)"):
    manual_desc = st.text_area("Job description text", height=180)

if st.button("Generate Tailored Resume", type="primary"):
    if not gemini_api_key:
        st.error("Enter your Gemini API key in the sidebar.")
        st.stop()
    if not job_url and not manual_desc.strip():
        st.error("Enter a LinkedIn URL or paste a description in the fallback box.")
        st.stop()

    try:
        # --- Load master resume ---
        with st.status("Working...", expanded=True) as status:
            try:
                with open("master_resume.json", "r", encoding="utf-8") as f:
                    master_resume_text = f.read()
                json.loads(master_resume_text)  # validate it's real JSON
                st.write("✅ Master resume loaded.")
            except FileNotFoundError:
                status.update(label="Error", state="error")
                st.error("Could not find 'master_resume.json' next to this app.")
                st.stop()
            except json.JSONDecodeError as e:
                status.update(label="Error", state="error")
                st.error(f"master_resume.json is not valid JSON: {e}")
                st.stop()

            # --- Get job description: manual paste wins if provided ---
            if manual_desc.strip():
                job_title, company_name = "Target Role", "Target Company"
                job_description = manual_desc.strip()
                st.write("✅ Using manually pasted description.")
            else:
                status.update(label="Fetching job description from LinkedIn...")
                job_title, company_name, job_description = scrape_linkedin_job(job_url)
                st.write(f"✅ Found: **{job_title}** at **{company_name}**")

            # --- Select the most relevant projects (deterministic cap at k=4) ---
            status.update(label="Selecting your most relevant projects...")
            resume_dict = json.loads(master_resume_text)
            resume_dict["projects"] = select_relevant_projects(
                job_description,
                resume_dict.get("projects", []),
                k=4,
                # Selection returns indices only, so Flash is plenty (cheap + fast).
                generate_fn=lambda p: _call_gemini(gemini_api_key, p),
            )
            tailored_resume_text = json.dumps(resume_dict, ensure_ascii=False, indent=2)
            st.write(
                f"✅ Selected {len(resume_dict['projects'])} best-fit projects."
            )

            # --- Tailor with Gemini (Pro + extended thinking, falls back to Flash) ---
            status.update(label="Gemini Pro is tailoring your resume...")
            tailored_markdown = generate_resume_markdown(
                gemini_api_key, tailored_resume_text, job_description
            )
            st.write("✅ Resume tailored.")

            # --- PDF ---
            status.update(label="Building PDF...")
            safe_title = re.sub(r"[^A-Za-z0-9]+", "_", job_title).strip("_")
            safe_company = re.sub(r"[^A-Za-z0-9]+", "_", company_name).strip("_")
            pdf_filename = f"Resume_{safe_company}_{safe_title}.pdf"
            convert_markdown_to_pdf(tailored_markdown, pdf_filename)
            st.write("✅ PDF ready.")

            # --- ATS keyword match report (bonus; fails soft) ---
            status.update(label="Checking keyword match against the job...")
            matched_kw, missing_kw = analyze_keyword_match(
                gemini_api_key, job_description, tailored_markdown
            )
            st.write("✅ Keyword report ready.")

            # --- Cover-letter brief (guidance to write it yourself; fails soft) ---
            status.update(label="Building your cover-letter brief...")
            cover_guide = generate_cover_letter_guide(
                gemini_api_key, master_resume_text, job_description,
                company_name, job_title,
            )
            st.write("✅ Cover-letter brief ready.")
            status.update(label="Done!", state="complete", expanded=False)

        st.success(f"Generated resume tailored for {company_name}.")
        with open(pdf_filename, "rb") as pdf_file:
            st.download_button(
                "📥 Download PDF Resume",
                data=pdf_file.read(),
                file_name=pdf_filename,
                mime="application/pdf",
                type="primary",
            )

        # --- ATS keyword match report ---
        total_kw = len(matched_kw) + len(missing_kw)
        if total_kw:
            score = round(100 * len(matched_kw) / total_kw)
            st.subheader(f"🎯 ATS Keyword Match: {score}%")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Covered**")
                for kw in matched_kw:
                    st.markdown(f"- {kw}")
            with col2:
                st.markdown("**⚠️ Missing**")
                if missing_kw:
                    for kw in missing_kw:
                        st.markdown(f"- {kw}")
                else:
                    st.markdown("_None — strong coverage._")
            if missing_kw:
                st.caption(
                    "Missing keywords are things the job asks for that aren't in your "
                    "resume. If you genuinely have one, add it to master_resume.json so "
                    "future runs can use it. If you don't have it, treat it as a real "
                    "gap — don't fabricate it."
                )

        # --- Cover letter brief ---
        if cover_guide:
            st.subheader("📝 Cover Letter Brief")
            st.caption(
                "Directions for writing your own cover letter — not the letter itself. "
                "A self-written letter reads as genuine; this just does the prep."
            )
            st.markdown(cover_guide)

        with st.expander("Preview"):
            st.markdown(tailored_markdown)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.info(
            "If LinkedIn blocked the fetch, open the fallback box above, paste the "
            "description, and click Generate again."
        )
