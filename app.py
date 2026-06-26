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

def call_gemini_with_fallback(client, prompt: str, config=None) -> str:
    """Call Gemini with model fallback + exponential backoff on transient errors.

    Tries the main model first; if it stays overloaded, drops to a lighter one
    that usually has more capacity. Shared by both project selection and resume
    generation so the retry behavior stays identical across the app.
    """
    models_to_try = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    last_error = None

    for model_name in models_to_try:
        for attempt in range(4):  # waits: 1s, 2s, 4s, then give up on this model
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                return response.text
            except Exception as e:
                last_error = e
                msg = str(e)
                transient = any(
                    code in msg for code in ("503", "UNAVAILABLE", "429", "overload")
                )
                if transient and attempt < 3:
                    time.sleep(2 ** attempt)  # 1, 2, 4 seconds
                    continue
                break  # non-transient error, or out of retries -> next model

    raise RuntimeError(f"Gemini is still unavailable after retries: {last_error}")


def generate_resume_markdown(client, master_json: str, job_desc: str) -> str:
    prompt = f"""
You are an expert resume writer. Produce a one-page resume tailored to the job
description below, using the Candidate Data as the only source of facts.

CRITICAL RULES
* FACTUAL ACCURACY: Use only facts present in the Candidate Data. Never invent or
  exaggerate skills, metrics, or experience.
* TAILORING: Select and order content by relevance to the job. The PRACTICAL
  PROJECTS in the Candidate Data have ALREADY been pre-filtered to the most
  relevant ones for this job — include EVERY project provided, ordered with the
  best fit first. Mirror the job's keywords only where the candidate truly has
  that skill.
* NO HYPERLINKS: Write every URL as plain text with NO scheme — exactly like
  "github.com/user/repo", never "https://github.com/user/repo". NEVER use Markdown
  link syntax [text](url) and NEVER wrap URLs in <angle brackets>.
* OUTPUT: Return raw Markdown only. Do NOT wrap it in ```markdown fences.

FOLLOW THIS EXACT STRUCTURE (keep the section headings and their order, and keep
the blank lines exactly as shown so each block renders on its own line):

# FULL NAME IN CAPITALS

City, State | email | phone

linkedin url | github url

## INTRODUCTION

A tailored 3-4 sentence professional summary.

## PRACTICAL PROJECTS

**1) Project Title**

One or two sentences describing the project, tailored to the job. Project Link: plain-text url

**2) Project Title**

One or two sentences. Project Link: plain-text url

## EDUCATION & CERTIFICATIONS

**Degree / Program** — Institution | dates

**Certifications:** list every certification from the Candidate Data, comma-separated

**Awards & Achievements:** list every award from the Candidate Data, including the
candidate's national STEM / academic olympiad achievements, comma-separated

## SKILLS & TOOLS

**Programming Languages:** ...

**Frameworks & Libraries:** ...

**Tools & Environment:** ...

**Methodologies:** ...

**Soft Skills:** ... | ... | ...

## LANGUAGES

**Portuguese:** Native

**English:** ...

---
CANDIDATE DATA (JSON):
{master_json}

---
JOB DESCRIPTION:
{job_desc}
"""

    config = types.GenerateContentConfig(temperature=0.2)
    return call_gemini_with_fallback(client, prompt, config=config)


# =====================================================================
#  STEP 3 — MARKDOWN -> PDF
# =====================================================================

# One CSS block for one Section. Splitting the resume into two Sections is what
# forced the body onto a second page (markdown-pdf starts every Section on a new
# page). We now render everything as a SINGLE Section and center only the header
# by emitting it as inline-styled HTML, so the body flows right under it with no
# blank gap. The `a` rule strips hyperlink styling if a link slips through.
PAGE_CSS = """
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt;
       color: #1a1a1a; line-height: 1.35; }
h1 { font-family: Helvetica, Arial, sans-serif; }
h2 { font-size: 12pt; margin: 12px 0 5px 0; padding-bottom: 2px;
     border-bottom: 1.5px solid #333333; }
p  { margin: 3px 0; }
strong { color: #000000; }
a { color: inherit; text-decoration: none; }
"""


def clean_links(md: str) -> str:
    """Normalize links: remove Markdown/angle-bracket link syntax AND strip the
    scheme so URLs read like "github.com/user/repo" instead of "https://...".
    Some AI screeners and parsers choke on or ignore full hyperlinks, so plain
    domain-relative text is the safer form."""
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)      # [text](url) -> text
    md = re.sub(r"<((?:https?://)?[^>\s]+)>", r"\1", md)  # <url> -> url
    md = re.sub(r"https?://(?:www\.)?", "", md)            # https://www.x -> x
    return md


def _build_single_page_markdown(markdown_text: str) -> str:
    """Combine a centered HTML header with the left-aligned Markdown body into one
    string, so the whole resume renders as a single Section (one continuous page)."""
    split_at = markdown_text.find("\n## ")
    if split_at == -1:
        header_md, body_md = markdown_text, ""
    else:
        header_md = markdown_text[:split_at].strip()
        body_md = markdown_text[split_at:].strip()

    header_html = []
    for line in (ln.strip() for ln in header_md.splitlines() if ln.strip()):
        if line.startswith("# "):
            header_html.append(
                f'<h1 style="text-align:center; font-size:20pt; margin:0 0 2px 0;">'
                f'{line[2:].strip()}</h1>'
            )
        else:
            header_html.append(
                f'<p style="text-align:center; font-size:10pt; color:#333333; '
                f'margin:1px 0;">{line}</p>'
            )

    return ("\n".join(header_html) + "\n\n" + body_md).strip()


def convert_markdown_to_pdf(markdown_text: str, output_filename: str) -> str:
    markdown_text = clean_links(markdown_text).strip()
    combined = _build_single_page_markdown(markdown_text)

    pdf = MarkdownPdf(toc_level=0)
    pdf.add_section(Section(combined), user_css=PAGE_CSS)
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
                master_dict = json.loads(master_resume_text)  # validate + keep
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

            # --- Create the Gemini client once, shared by selection + tailoring ---
            client = genai.Client(api_key=gemini_api_key)

            # --- Pick the 4 most relevant projects for THIS job ---
            status.update(label="Selecting the most relevant projects...")
            selected_projects = select_relevant_projects(
                job_description,
                master_dict.get("projects", []),
                k=4,
                generate_fn=lambda p: call_gemini_with_fallback(client, p),
            )
            tailored_dict = {**master_dict, "projects": selected_projects}
            tailored_master_json = json.dumps(tailored_dict, ensure_ascii=False)
            picked_names = ", ".join(p.get("name", "") for p in selected_projects)
            st.write(f"✅ Picked {len(selected_projects)} projects: {picked_names}")

            # --- Tailor with Gemini ---
            status.update(label="Gemini is tailoring your resume...")
            tailored_markdown = generate_resume_markdown(
                client, tailored_master_json, job_description
            )
            tailored_markdown = clean_links(tailored_markdown)
            st.write("✅ Resume tailored.")

            # --- PDF ---
            status.update(label="Building PDF...")
            safe_title = re.sub(r"[^A-Za-z0-9]+", "_", job_title).strip("_")
            safe_company = re.sub(r"[^A-Za-z0-9]+", "_", company_name).strip("_")
            pdf_filename = f"Resume_{safe_company}_{safe_title}.pdf"
            convert_markdown_to_pdf(tailored_markdown, pdf_filename)
            st.write("✅ PDF ready.")
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
        with st.expander("Preview"):
            st.markdown(tailored_markdown)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.info(
            "If LinkedIn blocked the fetch, open the fallback box above, paste the "
            "description, and click Generate again."
        )
