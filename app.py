import streamlit as st
import json
import os
import google.generativeai as genai
from apify_client import ApifyClient
from markdown_pdf import MarkdownPdf, Section

# Note: Before running this, you need to install the required libraries:
# pip install streamlit google-generativeai apify-client markdown-pdf

def scrape_linkedin_job(apify_key: str, job_url: str):
    """Uses Apify to scrape the job description from a LinkedIn URL."""
    client = ApifyClient(apify_key)
    
    # We are using a popular public actor for LinkedIn jobs. 
    # You can swap this ID if you prefer a different scraper from the Apify Store.
    actor_id = "rocky_scraper/linkedin-job-scraper" 
    
    run_input = {
        "startUrls": [{"url": job_url}],
        "maxItems": 1
    }
    
    # Run the Actor and wait for it to finish
    run = client.actor(actor_id).call(run_input=run_input)
    
    # Fetch the results from the dataset
    dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
    
    if not dataset_items:
        raise Exception("Apify could not extract data from this URL. Check the URL or Apify run logs.")
        
    job_data = dataset_items[0]
    # Extracting standard fields (these keys depend on the specific Apify actor used)
    title = job_data.get("title", "Target Role")
    company = job_data.get("companyName", "Target Company")
    description = job_data.get("description", "")
    
    return title, company, description

def generate_resume_markdown(gemini_key: str, master_json: str, job_desc: str) -> str:
    """Sends the master data and job description to Gemini to generate the tailored resume."""
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    You are an expert executive resume writer. Tailor the candidate's resume to perfectly align with the provided job description.
    
    CRITICAL RULES:
    * FACTUAL ACCURACY: ONLY use facts from the Candidate Data. DO NOT invent or hallucinate skills or metrics.
    * TAILORING: Rewrite bullet points to highlight skills relevant to the Job Description. Use matching keywords.
    * FORMAT: Output the final tailored resume in clean, professional Markdown format. Do NOT wrap the output in ```markdown codeblocks, just return the raw markdown text.
    
    ---
    CANDIDATE DATA:
    {master_json}
    
    ---
    JOB DESCRIPTION:
    {job_desc}
    """
    
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.2)
    )
    return response.text

def convert_markdown_to_pdf(markdown_text: str, output_filename: str):
    """Converts the raw markdown string into a formatted PDF file."""
    pdf = MarkdownPdf(toc_level=0) # Disable Table of Contents
    pdf.add_section(Section(markdown_text))
    pdf.save(output_filename)
    return output_filename

st.set_page_config(page_title="AI Resume Builder", page_icon="📄", layout="centered")

st.title("🚀 Automated AI Resume Builder")
st.markdown("Paste a LinkedIn Job URL below to automatically scrape the description, tailor your master resume, and generate a PDF.")

# Sidebar for API Keys (Keeps the main UI clean)
with st.sidebar:
    st.header("🔑 API Configurations")
    st.markdown("Enter your keys below. They are not stored permanently.")
    apify_api_key = st.text_input("Apify API Token", type="password")
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    
    st.divider()
    st.markdown("### Master Resume")
    st.markdown("Ensure your `master_resume.json` file is in the same directory as this app.")

# Main UI
job_url = st.text_input("🔗 LinkedIn Job Post URL", placeholder="https://www.linkedin.com/jobs/view/...")

if st.button("Generate Tailored Resume", type="primary"):
    if not apify_api_key or not gemini_api_key:
        st.error("Please provide both Apify and Gemini API keys in the sidebar.")
    elif not job_url:
        st.error("Please enter a valid LinkedIn Job URL.")
    else:
        try:
            # Step 1: Load Master Resume
            with st.status("Loading Master Resume...", expanded=True) as status:
                try:
                    with open('master_resume.json', 'r', encoding='utf-8') as f:
                        master_resume_text = f.read()
                    st.write("✅ Master resume loaded successfully.")
                except FileNotFoundError:
                    status.update(label="Error", state="error")
                    st.error("Could not find 'master_resume.json'. Please ensure it's in the same folder.")
                    st.stop()

                # Step 2: Scrape LinkedIn via Apify
                status.update(label="Scraping Job Description from LinkedIn via Apify...")
                job_title, company_name, job_description = scrape_linkedin_job(apify_api_key, job_url)
                st.write(f"✅ Found: **{job_title}** at **{company_name}**")
                
                # Step 3: Generate Markdown via Gemini
                status.update(label="AI is tailoring your resume...")
                tailored_markdown = generate_resume_markdown(gemini_api_key, master_resume_text, job_description)
                st.write("✅ Tailored resume generated.")
                
                # Step 4: Convert to PDF
                status.update(label="Converting to PDF...")
                safe_title = job_title.replace(" ", "_").replace("/", "_")
                pdf_filename = f"Resume_{company_name}_{safe_title}.pdf"
                convert_markdown_to_pdf(tailored_markdown, pdf_filename)
                st.write("✅ PDF created successfully.")
                
                status.update(label="Process Complete!", state="complete", expanded=False)

            # Display Success and Download Button
            st.success(f"Successfully generated resume for {company_name}!")
            
            with open(pdf_filename, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
                
            st.download_button(
                label="📥 Download Tailored PDF Resume",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                type="primary"
            )
            
            # Show a preview of the markdown
            with st.expander("Preview Tailored Text"):
                st.markdown(tailored_markdown)
                
        except Exception as e:
            st.error(f"An error occurred during the pipeline: {e}")