import io
import os
import json
import argparse
import tempfile
# ...existing code...
from flask import Flask, request, render_template, jsonify, send_from_directory
# ...existing code...
from datetime import datetime
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import spacy
import re
import pandas as pd
from tabulate import tabulate
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import Counter
from werkzeug.utils import secure_filename

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    print("You may need to install it with: python -m spacy download en_core_web_sm")
    exit(1)

def pdf_reader(file):
    """Extract text from PDF files"""
    try:
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
        page_interpreter = PDFPageInterpreter(resource_manager, converter)

        with open(file, 'rb') as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                page_interpreter.process_page(page)
            
            text = fake_file_handle.getvalue()

        # Close open handles
        converter.close()
        fake_file_handle.close()

        return text
    except FileNotFoundError:
        print(f"Error: The file {file} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_education(text):
    """Extract education information"""
    education_keywords = ['education', 'university', 'college', 'bachelor', 'master', 'phd', 'degree', 'diploma']
    education_info = []
    
    # Split text into sections
    lines = text.split('\n')
    in_education_section = False
    edu_section = ""
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Check if this line marks the start of education section
        if any(keyword in line_lower for keyword in education_keywords) and len(line_lower) < 30:
            in_education_section = True
            edu_section = line + "\n"
            
            # Look at next few lines to capture education details
            for j in range(1, 10):  # Look at next 10 lines max
                if i+j < len(lines):
                    edu_section += lines[i+j] + "\n"
            
            # Use regex to extract degree and institution
            degree_pattern = r'(?:bachelor|master|phd|b\.?[a-z]*|m\.?[a-z]*|ph\.?d)[\s\.]+(?:of|in)?\s+[a-z\s]+(?:engineering|science|arts|commerce|business|administration|technology)'
            institution_pattern = r'(?:university|college|institute|school) of [a-z\s]+|[a-z\s]+ (?:university|college|institute|school)'
            
            degrees = re.findall(degree_pattern, edu_section.lower())
            institutions = re.findall(institution_pattern, edu_section.lower())
            
            if degrees or institutions:
                education_info.append({
                    'degree': degrees[0].strip().title() if degrees else "",
                    'institution': institutions[0].strip().title() if institutions else ""
                })
            else:
                education_info.append({'raw': edu_section.strip()})
    
    return education_info

def extract_experience(text):
    """Extract work experience information"""
    experience_keywords = ['experience', 'work history', 'employment', 'career', 'job history']
    experience_info = []
    
    # Split text into sections
    lines = text.split('\n')
    in_exp_section = False
    exp_section = ""
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Check if this line marks the start of experience section
        if any(keyword in line_lower for keyword in experience_keywords) and len(line_lower) < 30:
            in_exp_section = True
            exp_section = line + "\n"
            
            # Look at next several lines to capture experience details
            for j in range(1, 20):  # Look at next 20 lines max
                if i+j < len(lines):
                    exp_section += lines[i+j] + "\n"
            
            # Extract company names, positions and dates
            company_pattern = r'(?:at|with)?\s([A-Z][A-Za-z0-9\'\-\&\.\s]{2,50})'
            position_pattern = r'([A-Z][a-z]+\s+[A-Za-z]+(?:\s+[A-Za-z]+)?(?:\s+[A-Za-z]+)?)\s+(?:at|in)'
            date_pattern = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-]+\d{4}\s+(?:to|--|–)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-]+\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-]+\d{4}\s+(?:to|--|–)\s+(?:Present|Current|Now)'
            
            companies = re.findall(company_pattern, exp_section)
            positions = re.findall(position_pattern, exp_section)
            dates = re.findall(date_pattern, exp_section)
            
            # Add experience details
            if positions or companies or dates:
                experience_info.append({
                    'position': positions[0].strip() if positions else "",
                    'company': companies[0].strip() if companies else "",
                    'duration': dates[0].strip() if dates else ""
                })
            
            # Extract responsibilities using NLP
            doc = nlp(exp_section)
            responsibilities = []
            for sent in doc.sents:
                if len(sent.text.strip()) > 10 and ('develop' in sent.text.lower() or 
                                                   'manage' in sent.text.lower() or 
                                                   'create' in sent.text.lower() or
                                                   'lead' in sent.text.lower() or
                                                   'implement' in sent.text.lower()):
                    responsibilities.append(sent.text.strip())
            
            if responsibilities:
                if experience_info:
                    experience_info[-1]['responsibilities'] = responsibilities[:3]  # Limit to 3 key responsibilities
    
    return experience_info

def extract_info_custom(text):
    """Extract comprehensive information from resume text"""
    doc = nlp(text)
    
    # Initialize data dictionary
    data = {
        "name": "",
        "email": "",
        "phone": "",
        "skills": [],
        "education": [],
        "experience": [],
        "languages": [],
        "certifications": []
    }
    
    # Extract email using regex
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        data["email"] = emails[0]
    
    # Extract phone using regex
    phone_pattern = r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    phones = re.findall(phone_pattern, text)
    if phones:
        data["phone"] = phones[0]
    
    # Extract GitHub/LinkedIn profiles
    github_pattern = r'github\.com\/[A-Za-z0-9_-]+'
    linkedin_pattern = r'linkedin\.com\/in\/[A-Za-z0-9_-]+'
    
    github = re.findall(github_pattern, text)
    linkedin = re.findall(linkedin_pattern, text)
    
    if github:
        data["github"] = github[0]
    if linkedin:
        data["linkedin"] = linkedin[0]
    
    # Expanded skill keywords
    skill_keywords = [
        # Programming Languages
        "python", "java", "javascript", "typescript", "html", "css", "php", "ruby", "c++", "c#", "swift", "kotlin", "go", "rust", "scala", 
        # Frameworks & Libraries
        "react", "angular", "vue", "node", "django", "flask", "spring", "laravel", "express", "fastapi", "tensorflow", "pytorch", "pandas", "numpy",
        # Databases
        "sql", "mysql", "postgresql", "mongodb", "firebase", "oracle", "redis", "cassandra", "dynamodb", "sqlite",  
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform", "ansible", "git", "github", "cicd", "devops",
        # Data Science & Analytics
        "machine learning", "data analysis", "data science", "big data", "data visualization", "statistics", "nlp", "computer vision", "ai", 
        # Business Intelligence
        "excel", "powerbi", "tableau", "looker", "dax", "power query", "r", "spss", "sas", 
        # Project Management
        "agile", "scrum", "kanban", "jira", "trello", "project management", "waterfall", "pmp", "prince2",
        # Soft Skills
        "leadership", "teamwork", "communication", "problem solving", "critical thinking", "time management",
        # Marketing
        "seo", "sem", "google analytics", "social media", "content marketing", "email marketing",
        # Design
        "photoshop", "illustrator", "figma", "sketch", "ui design", "ux design", "adobe creative suite"
    ]
    
    found_skills = []
    text_lower = text.lower()
    for skill in skill_keywords:
        if skill in text_lower:
            found_skills.append(skill.title())  # Capitalize for consistency
    
    data["skills"] = found_skills
    
    # Extract name from first few lines
    lines = text.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if line and len(line) < 40 and not any(c in line for c in "@.,:;()[]{}"):
            data["name"] = line
            break
    
    # Extract education information
    data["education"] = extract_education(text)
    
    # Extract experience information
    data["experience"] = extract_experience(text)
    
    # Extract languages
    language_section = re.search(r'languages?:?.*?(?:\n|$)((?:.*?\n?){1,5})', text.lower())
    if language_section:
        lang_text = language_section.group(1)
        # Common languages
        languages = ["english", "spanish", "french", "german", "chinese", "japanese", "russian", 
                     "arabic", "hindi", "portuguese", "italian", "dutch", "korean", "swedish", "turkish"]
        for lang in languages:
            if lang in lang_text.lower():
                data["languages"].append(lang.title())
    
    # Extract certifications
    cert_patterns = [
        r'certified\s+[a-z\s]+(?:developer|engineer|administrator|architect|specialist|professional)',
        r'[a-z]+\s+certification',
        r'[A-Z]{2,}(?:-[A-Z\d]+)+' # For abbreviations like AWS-SAA, MCSA, etc.
    ]
    
    certs = []
    for pattern in cert_patterns:
        found = re.findall(pattern, text.lower())
        certs.extend([c.strip().title() for c in found])
    
    if certs:
        data["certifications"] = list(set(certs))  # Remove duplicates
    
    return data

def match_skills(resume_skills, job_requirement):
    """Match resume skills against job requirements"""
    resume_skills_lower = [skill.lower() for skill in resume_skills]
    job_requirement_lower = [skill.lower() for skill in job_requirement]
    
    matched_skills = []
    unmatched_skills = []
    
    for skill in job_requirement_lower:
        found = False
        for resume_skill in resume_skills_lower:
            if skill in resume_skill or resume_skill in skill:
                matched_skills.append(skill)
                found = True
                break
        if not found:
            unmatched_skills.append(skill)
    
    return matched_skills, unmatched_skills

def score_resume(resume_data, job_data):
    """Score the resume based on job data"""
    if not resume_data or not job_data:
        return 0, [], []
    
    score = 0
    matched_skills = []
    missing_skills = []
    
    if resume_data.get('skills') and job_data.get('required_skills'):
        matched, unmatched = match_skills(resume_data['skills'], job_data['required_skills'])
        matched_skills = matched
        missing_skills = unmatched
        
        if job_data['required_skills']:  # Avoid division by zero
            score = len(matched) / len(job_data['required_skills']) * 100
    
    return score, matched_skills, missing_skills

def generate_suggestions(resume_data, job_data, score, missing_skills):
    """Generate personalized suggestions based on resume analysis"""
    suggestions = []
    
    # Suggestion based on skills match
    if score < 50:
        suggestions.append(f"Your skills match is below 50%. Consider upskilling in: {', '.join(missing_skills)}")
    elif score < 75:
        suggestions.append(f"To improve your chances, consider gaining more experience in: {', '.join(missing_skills)}")
    else:
        suggestions.append("Your skills match the job requirements well!")
    
    # Check for contact information
    if not resume_data.get('email') or not resume_data.get('phone'):
        suggestions.append("Add complete contact information (email and phone) to your resume")
    
    # Check for LinkedIn profile
    if not resume_data.get('linkedin'):
        suggestions.append("Consider adding your LinkedIn profile URL to increase credibility")
    
    # Check experience descriptions
    if resume_data.get('experience'):
        has_metrics = False
        for exp in resume_data['experience']:
            if 'responsibilities' in exp:
                for resp in exp.get('responsibilities', []):
                    if any(metric in resp.lower() for metric in ['increase', 'reduce', 'improve', '%', 'percent', 'decreased', 'grew']):
                        has_metrics = True
                        break
        
        if not has_metrics:
            suggestions.append("Add measurable achievements with metrics to your experience descriptions")
    
    # Check for relevant certifications
    if job_data.get('preferred_certifications') and resume_data.get('certifications'):
        cert_matches = [c for c in job_data['preferred_certifications'] if any(rc.lower() in c.lower() or c.lower() in rc.lower() for rc in resume_data['certifications'])]
        if not cert_matches and job_data['preferred_certifications']:
            suggestions.append(f"Consider obtaining relevant certifications like: {', '.join(job_data['preferred_certifications'])}")
    
    # Check resume length indirectly
    word_count = len(resume_data.get('skills', [])) + len(resume_data.get('education', [])) + len(resume_data.get('experience', []))
    if word_count < 10:
        suggestions.append("Your resume may be too brief. Consider adding more details about your experience and skills")
    
    return suggestions

def visualize_skills_match(matched, missing, filename):
    """Create a visualization of skills match"""
    labels = ['Matched Skills', 'Missing Skills']
    sizes = [len(matched), len(missing)]
    colors = ['#10b981', '#e2e8f0']  # Professional green and light gray
    explode = (0.08, 0)  # explode 1st slice

    fig, ax = plt.subplots(figsize=(12, 7))
    wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=90)
    
    # Style the text
    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight('bold')
        text.set_color('#1e293b')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    plt.axis('equal')
    plt.title('Skills Match Analysis', fontsize=14, fontweight='bold', color='#1e3a8a', pad=20)

    # Add text boxes with the actual skills
    matched_legend = '\n'.join([f"✓ {skill.title()}" for skill in matched]) if matched else "None"
    missing_legend = '\n'.join([f"✗ {skill.title()}" for skill in missing]) if missing else "None"

    # Left side for matched skills
    plt.text(0.02, 0.5, f"Matched Skills:\n{matched_legend}", transform=ax.transAxes, fontsize=9,
             verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="#d1fae5", edgecolor="#10b981", linewidth=2, alpha=0.9))

    # Right side for missing skills
    plt.text(0.73, 0.5, f"Missing Skills:\n{missing_legend}", transform=ax.transAxes, fontsize=9,
             verticalalignment='center', horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.5", facecolor="#f1f5f9", edgecolor="#64748b", linewidth=2, alpha=0.9))

    # Adjust layout to make space for text
    plt.subplots_adjust(left=0.2, right=0.8)
    plt.tight_layout()

    plt.savefig(filename, bbox_inches='tight', dpi=100, facecolor='white')
    plt.close()

    return filename

def calculate_ats_score(resume_text, resume_data):
    """
    Calculate ATS compatibility score (0-100)
    Returns score and list of issues/warnings
    """
    ats_score = 100  # Start with perfect score
    issues = []
    
    # Check 1: Special characters & symbols
    special_chars = len(re.findall(r'[^\w\s\-@./#]', resume_text))
    if special_chars > 10:
        ats_score -= 10
        issues.append(f"⚠️ Found {special_chars} special characters (ATS may not parse)")
    
    # Check 2: Contact information completeness
    if not resume_data.get('email'):
        ats_score -= 5
        issues.append("❌ Missing email address")
    if not resume_data.get('phone'):
        ats_score -= 5
        issues.append("❌ Missing phone number")
    
    # Check 3: Required sections
    required_sections = ['experience', 'education', 'skills']
    for section in required_sections:
        if not resume_data.get(section) or len(resume_data[section]) == 0:
            ats_score -= 8
            issues.append(f"❌ Missing {section.title()} section")
    
    # Check 4: Skills formatting
    skills = resume_data.get('skills', [])
    if len(skills) < 5:
        ats_score -= 10
        issues.append("⚠️ Too few skills listed (recommended: 10+)")
    elif len(skills) > 50:
        ats_score -= 5
        issues.append("⚠️ Too many skills listed (ATS may not process all)")
    
    # Check 5: Experience descriptions
    experiences = resume_data.get('experience', [])
    if not experiences:
        ats_score -= 15
        issues.append("❌ No work experience found")
    else:
        has_descriptions = False
        for exp in experiences:
            if exp.get('responsibilities') and len(exp['responsibilities']) > 0:
                has_descriptions = True
                break
        if not has_descriptions:
            ats_score -= 5
            issues.append("⚠️ Experience entries lack detailed descriptions")
    
    # Check 6: Education formatting
    education = resume_data.get('education', [])
    if not education:
        ats_score -= 10
        issues.append("⚠️ No education information found")
    
    # Check 7: Dates formatting
    date_pattern = r'\d{4}|\d{1,2}/\d{1,2}'
    dates_found = len(re.findall(date_pattern, resume_text))
    if dates_found < 4:  # Should have multiple dates
        ats_score -= 5
        issues.append("⚠️ Inconsistent or missing date formatting")
    
    # Check 8: Long lines (ATS readability)
    lines = resume_text.split('\n')
    long_lines = sum(1 for line in lines if len(line) > 100)
    if long_lines > len(lines) * 0.3:  # More than 30% long lines
        ats_score -= 5
        issues.append("⚠️ Some lines too long (may break ATS parsing)")
    
    # Check 9: Keywords relevance
    job_keywords = ['experience', 'education', 'skills', 'responsibility', 'achievement']
    keywords_found = sum(1 for kw in job_keywords if kw in resume_text.lower())
    if keywords_found < 3:
        ats_score -= 5
        issues.append("⚠️ Low keyword density (add industry-specific keywords)")
    
    # Ensure score doesn't go below 0
    ats_score = max(0, ats_score)
    
    return {
        'score': ats_score,
        'issues': issues,
        'recommendation': get_ats_recommendation(ats_score)
    }

def get_ats_recommendation(score):
    """Get recommendation based on ATS score"""
    if score >= 85:
        return "✅ Excellent! Your resume is ATS-friendly"
    elif score >= 70:
        return "⚠️ Good, but address flagged issues to improve ATS compatibility"
    elif score >= 50:
        return "❌ Fair - Multiple issues may prevent ATS from parsing correctly"
    else:
        return "❌ Poor - Critical issues need immediate attention"

def visualize_ats_score(ats_score, filename):
    """Create a gauge chart for ATS score"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Determine color based on score
    if ats_score >= 85:
        color = '#10b981'  # Professional green
    elif ats_score >= 70:
        color = '#f97316'  # Professional orange
    else:
        color = '#ef4444'  # Professional red
    
    # Create gauge
    bar = ax.barh(['ATS Score'], [ats_score], color=color, height=0.3, edgecolor='#1e3a8a', linewidth=2)
    
    # Add background bar
    ax.barh(['ATS Score'], [100], height=0.3, color='#e2e8f0', alpha=0.5, zorder=0)
    
    ax.set_xlim(0, 100)
    ax.text(ats_score/2, 0, f'{ats_score}%', ha='center', va='center', 
            fontsize=18, fontweight='bold', color='white')
    
    plt.title('ATS Compatibility Score', fontsize=13, fontweight='bold', color='#1e3a8a', pad=15)
    ax.set_xlabel('Score', fontsize=10, color='#1e3a8a', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(colors='#1e3a8a')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=100, facecolor='white')
    plt.close()

def generate_cover_letter(resume_data, job_data, matched_skills, profession=None):
    """Generate a customized cover letter based on resume and job requirements"""
    
    # Extract key information from resume
    candidate_name = resume_data.get('name', 'Applicant')
    company_name = job_data.get('title', 'the position').split('at')[-1].strip() if 'at' in job_data.get('title', '') else 'the Company'
    position = job_data.get('title', 'the position')
    
    # Get top matched skills with bullet points
    top_skills = matched_skills[:5] if matched_skills else []
    skills_text = '\n'.join([f"• {skill}" for skill in top_skills]) if top_skills else "• Strong technical acumen\n• Problem-solving abilities\n• Team collaboration"
    
    # Get recent experience highlights
    experience_highlights = []
    for exp in resume_data.get('experience', [])[:2]:  # Get first 2 experiences
        if exp.get('position'):
            highlight = f"- {exp.get('position')} at {exp.get('company')}"
            if exp.get('responsibilities'):
                highlight += f": {exp['responsibilities'][0]}"
            experience_highlights.append(highlight)
    
    experience_text = '\n'.join(experience_highlights) if experience_highlights else '• Demonstrated success in delivering technical solutions'
    
    # Get education
    education_info = ""
    if resume_data.get('education'):
        edu = resume_data['education'][0]
        if edu.get('degree') and edu.get('institution'):
            education_info = f"{edu['degree']} from {edu['institution']}"
        elif edu.get('degree'):
            education_info = edu['degree']
    
    # Build cover letter content
    cover_letter = f"""Dear Hiring Manager,

I am writing to express my strong interest in the {position} position at {company_name}. With my background in {profession if profession else 'technology and problem-solving'} and proven track record of delivering impactful results, I am confident in my ability to contribute meaningfully to your team.

Throughout my career, I have developed expertise in key areas that align perfectly with your requirements:

{skills_text}

My professional experience includes:

{experience_text}

I hold {education_info if education_info else 'a strong educational background'}, which has equipped me with the foundational knowledge and skills necessary to excel in this role.

What excites me most about this opportunity is the chance to leverage my skills and experience to drive innovation and create value for {company_name}. I am particularly drawn to your organization's commitment to excellence and look forward to discussing how I can contribute to your team's success.

I am eager to explore this opportunity further and would welcome the chance to discuss how my background, skills, and enthusiasm can benefit {company_name}. Thank you for considering my application.

Sincerely,

{candidate_name}
Email: {resume_data.get('email', '[Your Email]')}
Phone: {resume_data.get('phone', '[Your Phone]')}
"""
    
    return cover_letter.strip()

def analyze_batch_resumes(resume_files, job_data, profession=None):
    """Analyze multiple resumes and return comparative results"""
    batch_results = []
    
    for resume_file in resume_files:
        try:
            # Extract text from PDF
            resume_text = pdf_reader(resume_file)
            if not resume_text:
                continue
            
            # Extract resume information
            resume_data = extract_info_custom(resume_text)
            
            # Score resume
            score, matched_skills, missing_skills = score_resume(resume_data, job_data)
            
            # Calculate ATS score
            ats_result = calculate_ats_score(resume_text, resume_data)
            
            # Count experience years (estimate from duration strings)
            experience_count = len(resume_data.get('experience', []))
            
            batch_results.append({
                'filename': os.path.basename(resume_file),
                'name': resume_data.get('name', 'Unknown'),
                'email': resume_data.get('email', ''),
                'phone': resume_data.get('phone', ''),
                'match_score': round(score, 2),
                'ats_score': round(ats_result['score'], 2),
                'matched_skills': len(matched_skills),
                'missing_skills': len(missing_skills),
                'total_skills': len(resume_data.get('skills', [])),
                'experience_count': experience_count,
                'education_count': len(resume_data.get('education', [])),
                'skills': resume_data.get('skills', []),
                'matched_skills_list': matched_skills,
                'missing_skills_list': missing_skills,
                'ats_issues': len(ats_result['issues'])
            })
        except Exception as e:
            print(f"Error analyzing {resume_file}: {str(e)}")
            continue
    
    # Sort by match score (descending)
    batch_results = sorted(batch_results, key=lambda x: x['match_score'], reverse=True)
    
    # Add ranking
    for idx, result in enumerate(batch_results, 1):
        result['rank'] = idx
    
    return batch_results

def generate_batch_comparison_chart(batch_results, filename):
    """Create a comparison chart for batch results"""
    if not batch_results:
        return None
    
    # Prepare data
    names = [r['name'][:14] for r in batch_results]  # Truncate long names
    scores = [r['match_score'] for r in batch_results]
    ats_scores = [r['ats_score'] for r in batch_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Match Score Chart - Professional colors
    colors1 = ['#10b981' if score >= 70 else '#f97316' if score >= 50 else '#ef4444' for score in scores]
    bars1 = ax1.barh(names, scores, color=colors1, edgecolor='#1e3a8a', linewidth=1.5, height=0.6)
    ax1.set_xlabel('Match Score (%)', fontsize=10, fontweight='bold', color='#1e3a8a')
    ax1.set_title('Resume Match Scores', fontsize=12, fontweight='bold', color='#1e3a8a')
    ax1.set_xlim(0, 100)
    ax1.grid(axis='x', alpha=0.2, linestyle='--')
    for i, v in enumerate(scores):
        ax1.text(v + 1, i, f'{v}%', va='center', fontweight='bold', fontsize=9, color='#1e3a8a')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ATS Score Chart - Professional colors
    colors2 = ['#10b981' if score >= 85 else '#f97316' if score >= 70 else '#ef4444' for score in ats_scores]
    bars2 = ax2.barh(names, ats_scores, color=colors2, edgecolor='#1e3a8a', linewidth=1.5, height=0.6)
    ax2.set_xlabel('ATS Score (%)', fontsize=10, fontweight='bold', color='#1e3a8a')
    ax2.set_title('ATS Compatibility Scores', fontsize=12, fontweight='bold', color='#1e3a8a')
    ax2.set_xlim(0, 100)
    ax2.grid(axis='x', alpha=0.2, linestyle='--')
    for i, v in enumerate(ats_scores):
        ax2.text(v + 1, i, f'{v}%', va='center', fontweight='bold', fontsize=9, color='#1e3a8a')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Style common elements
    for ax in [ax1, ax2]:
        ax.tick_params(colors='#1e3a8a')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return filename

def export_batch_results_csv(batch_results, filename):
    """Export batch results to CSV"""
    try:
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Rank', 'Name', 'Email', 'Phone', 'Match Score', 'ATS Score', 
                         'Matched Skills', 'Missing Skills', 'Total Skills', 
                         'Experience Count', 'Education Count', 'ATS Issues']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in batch_results:
                writer.writerow({
                    'Rank': result['rank'],
                    'Name': result['name'],
                    'Email': result['email'],
                    'Phone': result['phone'],
                    'Match Score': result['match_score'],
                    'ATS Score': result['ats_score'],
                    'Matched Skills': result['matched_skills'],
                    'Missing Skills': result['missing_skills'],
                    'Total Skills': result['total_skills'],
                    'Experience Count': result['experience_count'],
                    'Education Count': result['education_count'],
                    'ATS Issues': result['ats_issues']
                })
        return filename
    except Exception as e:
        print(f"Error exporting CSV: {str(e)}")
        return None

def generate_report(resume_data, job_data, score, matched_skills, missing_skills, suggestions, suitable_jobs=None):
    """Generate a comprehensive report"""
    report = {
        "candidate_name": resume_data.get("name", "Unknown"),
        "contact": {
            "email": resume_data.get("email", ""),
            "phone": resume_data.get("phone", "")
        },
        "match_score": score,
        "skills_analysis": {
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "total_required_skills": len(job_data.get("required_skills", [])),
            "total_matched_skills": len(matched_skills)
        },
        "education": resume_data.get("education", []),
        "experience": resume_data.get("experience", []),
        "suggestions": suggestions,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add suitable jobs if provided
    if suitable_jobs:
        report["suitable_job_recommendations"] = suitable_jobs
    
    return report

def save_report_as_json(report, filename):
    """Save report as JSON file"""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=4)
    return filename

def print_report(report):
    """Print formatted report to console"""
    print("\n====== RESUME ANALYSIS REPORT ======")
    print(f"Candidate: {report['candidate_name']}")
    print(f"Contact: {report['contact']['email']} | {report['contact']['phone']}")
    print(f"\nMatch Score: {report['match_score']:.2f}%")
    
    print("\nSkills Analysis:")
    print(f"- Matched Skills ({len(report['skills_analysis']['matched_skills'])}): {', '.join(report['skills_analysis']['matched_skills'])}")
    print(f"- Missing Skills ({len(report['skills_analysis']['missing_skills'])}): {', '.join(report['skills_analysis']['missing_skills'])}")
    
    print("\nSuggestions:")
    for i, suggestion in enumerate(report['suggestions'], 1):
        print(f"{i}. {suggestion}")
    
    # Print suitable job recommendations if available
    if 'suitable_job_recommendations' in report:
        print("\nRecommended Job Roles:")
        for i, job in enumerate(report['suitable_job_recommendations'], 1):
            print(f"{i}. {job['title']} (Match: {job['match_score']:.2f}%)")
            print(f"   Key skills: {', '.join(job['key_skills'])}")
    
    print(f"\nAnalysis Date: {report['analysis_date']}")
    print("=====================================")

def analyze_resumes(job_data, resume_files, job_database=None):
    """Analyze multiple resumes against job requirements"""
    results = []
    
    for resume_file in resume_files:
        print(f"\nAnalyzing resume: {resume_file}...")
        resume_text = pdf_reader(resume_file)
        
        if resume_text:
            resume_data = extract_info_custom(resume_text)
            
            score, matched_skills, missing_skills = score_resume(resume_data, job_data)
            suggestions = generate_suggestions(resume_data, job_data, score, missing_skills)
            
            # Find suitable jobs if job database is provided
            suitable_jobs = None
            if job_database:
                suitable_jobs = find_suitable_jobs(resume_data, job_database)
            
            # Generate visualization
            chart_filename = f"skills_match_{os.path.basename(resume_file).split('.')[0]}.png"
            visualize_skills_match(matched_skills, missing_skills, chart_filename)
            
            # Generate and save report
            report = generate_report(resume_data, job_data, score, matched_skills, missing_skills, suggestions, suitable_jobs)
            report_filename = f"report_{os.path.basename(resume_file).split('.')[0]}.json"
            save_report_as_json(report, report_filename)
            
            # Print report
            print_report(report)
            
            results.append({
                "filename": resume_file,
                "name": resume_data.get("name", "Unknown"),
                "score": score,
                "matched_skills": len(matched_skills),
                "missing_skills": len(missing_skills),
                "report_file": report_filename,
                "chart_file": chart_filename
            })
        else:
            print(f"Could not extract text from resume: {resume_file}")
    
    return results

def rank_candidates(results):
    """Rank candidates based on their scores"""
    ranked = sorted(results, key=lambda x: x["score"], reverse=True)
    
    print("\n====== CANDIDATE RANKING ======")
    headers = ["Rank", "Name", "Match Score", "Matched Skills", "Missing Skills"]
    table_data = []
    
    for i, result in enumerate(ranked, 1):
        table_data.append([
            i, 
            result["name"], 
            f"{result['score']:.2f}%", 
            result["matched_skills"],
            result["missing_skills"]
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    return ranked

def find_suitable_jobs(resume_data, job_database):
    """Find suitable jobs for a candidate based on their skills and experience"""
    suitable_jobs = []
    
    if not resume_data.get('skills'):
        return suitable_jobs
    
    for job in job_database:
        # Calculate skill match
        matched, _ = match_skills(resume_data['skills'], job['required_skills'])
        match_score = len(matched) / len(job['required_skills']) * 100 if job['required_skills'] else 0
        
        # Consider job suitable if match score is above 50%
        if match_score >= 50:
            suitable_jobs.append({
                'title': job['title'],
                'match_score': match_score,
                'key_skills': matched,
                'description': job.get('description', '')
            })
    
    # Sort by match score (highest first)
    suitable_jobs = sorted(suitable_jobs, key=lambda x: x['match_score'], reverse=True)
    
    return suitable_jobs[:5]  # Return top 5 suitable jobs

def load_job_database(file_path):
    """Load job database from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Job database file {file_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not parse job database file {file_path}.")
        return []

def analyze_job_description(job_text):
    """Extract skills, experience, and certifications from job description text"""
    doc = nlp(job_text)
    
    # Initialize job data
    job_data = {
        "title": "",
        "required_skills": [],
        "experience_required": "",
        "preferred_certifications": [],
        "education_required": "",
        "description": job_text[:500] + "..." if len(job_text) > 500 else job_text  # Store truncated description
    }
    
    # Extract job title
    lines = job_text.split('\n')
    for line in lines[:5]:  # Check first 5 lines for job title
        line = line.strip()
        if line and len(line) < 60 and not line.lower().startswith(('job', 'position')):
            job_data["title"] = line
            break
    
    # Known skill keywords to look for
    skill_keywords = [
        # Programming Languages
        "python", "java", "javascript", "typescript", "html", "css", "php", "ruby", "c++", "c#", "swift", "kotlin", "go", "rust", "scala", 
        # Frameworks & Libraries
        "react", "angular", "vue", "node", "django", "flask", "spring", "laravel", "express", "fastapi", "tensorflow", "pytorch", "pandas", "numpy",
        # Databases
        "sql", "mysql", "postgresql", "mongodb", "firebase", "oracle", "redis", "cassandra", "dynamodb", "sqlite",  
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform", "ansible", "git", "github", "cicd", "devops",
        # Data Science & Analytics
        "machine learning", "data analysis", "data science", "big data", "data visualization", "statistics", "nlp", "computer vision", "ai", 
        # Business Intelligence
        "excel", "powerbi", "tableau", "looker", "dax", "power query", "r", "spss", "sas", 
        # Project Management
        "agile", "scrum", "kanban", "jira", "trello", "project management", "waterfall", "pmp", "prince2",
        # Soft Skills
        "leadership", "teamwork", "communication", "problem solving", "critical thinking", "time management",
        # Marketing
        "seo", "sem", "google analytics", "social media", "content marketing", "email marketing",
        # Design
        "photoshop", "illustrator", "figma", "sketch", "ui design", "ux design", "adobe creative suite"
    ]
    
    # Extract skills
    found_skills = []
    job_text_lower = job_text.lower()
    for skill in skill_keywords:
        if skill in job_text_lower:
            found_skills.append(skill.title())  # Capitalize for consistency
    
    job_data["required_skills"] = found_skills
    
    # Extract experience requirements
    experience_patterns = [
        r'(\d+)[\+\-]?\s+years?\s+(?:of\s+)?experience',
        r'experience\s+(?:of|for)?\s+(\d+)[\+\-]?\s+years?',
        r'minimum\s+(?:of\s+)?(\d+)[\+\-]?\s+years?\s+(?:of\s+)?experience'
    ]
    
    for pattern in experience_patterns:
        matches = re.findall(pattern, job_text_lower)
        if matches:
            job_data["experience_required"] = f"{matches[0]}+ years"
            break
    
    # Extract certification requirements
    cert_patterns = [
        r'(?:certification|certified)\s+(?:in|as)\s+([a-z\s]+(?:developer|engineer|administrator|architect|specialist|professional))',
        r'([a-z]+\s+certification)',
        r'([A-Z]{2,}(?:-[A-Z\d]+)+)' # For abbreviations like AWS-SAA, MCSA, etc.
    ]
    
    certs = []
    for pattern in cert_patterns:
        found = re.findall(pattern, job_text_lower)
        certs.extend([c.strip().title() for c in found])
    
    if certs:
        job_data["preferred_certifications"] = list(set(certs))  # Remove duplicates
    
    # Extract education requirements
    education_patterns = [
        r'(?:bachelor|master|phd|b\.?[a-z]*|m\.?[a-z]*|ph\.?d)[\s\.]+(?:of|in)?\s+[a-z\s]+(?:engineering|science|arts|commerce|business|administration|technology)',
        r'(?:bachelor|master|phd|graduate|undergraduate)\s+degree',
        r'degree\s+in\s+[a-z\s]+'
    ]
    
    for pattern in education_patterns:
        matches = re.findall(pattern, job_text_lower)
        if matches:
            job_data["education_required"] = matches[0].strip().title()
            break
    
    return job_data

def save_job_to_database(job_data, database_file):
    """Save a job to the job database"""
    job_database = []
    
    # Load existing database if it exists
    try:
        with open(database_file, 'r') as f:
            job_database = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        job_database = []
    
    # Add new job if it doesn't already exist
    job_exists = False
    for job in job_database:
        if job.get('title') == job_data.get('title'):
            job_exists = True
            break
    
    if not job_exists:
        job_database.append(job_data)
        
        # Save updated database
        with open(database_file, 'w') as f:
            json.dump(job_database, f, indent=4)
        
        print(f"Job '{job_data.get('title')}' added to database.")
    else:
        print(f"Job '{job_data.get('title')}' already exists in database.")
    
    return job_data

def main():
    """Main function to run the resume analyzer"""
    parser = argparse.ArgumentParser(description='Enhanced Resume Analyzer')
    parser.add_argument('--resumes', nargs='+', help='Paths to resume PDF files')
    parser.add_argument('--skills', nargs='+', help='Required skills for the job')
    parser.add_argument('--certifications', nargs='+', help='Preferred certifications for the job')
    parser.add_argument('--output-dir', default=".", help='Output directory for reports and charts')
    parser.add_argument('--summary', action='store_true', help='Generate a summary report for all candidates')
    parser.add_argument('--job-database', default="job_database.json", help='Path to job database file')
    parser.add_argument('--analyze-job', action='store_true', help='Analyze a job description')
    parser.add_argument('--job-description-file', help='Path to job description file')
    
    args = parser.parse_args()
    
    # Change to output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
    
    # Handle job description analysis
    if args.analyze_job:
        if not args.job_description_file:
            print("Error: Please provide a job description file with --job-description-file")
            return
        
        job_text = pdf_reader(args.job_description_file)
        if job_text:
            job_data = analyze_job_description(job_text)
            save_job_to_database(job_data, args.job_database)
            print(f"Job description analyzed and saved to {args.job_database}")
        else:
            print("Error: Could not read job description file")
            return
    
    # Process resumes
    if args.resumes:
        # Create job data from arguments or load from database
        job_data = {}
        if args.skills:
            job_data["required_skills"] = args.skills
        if args.certifications:
            job_data["preferred_certifications"] = args.certifications
        
        # If job data is empty, use the last job from database
        if not job_data.get("required_skills"):
            job_database = load_job_database(args.job_database)
            if job_database:
                job_data = job_database[-1]  # Use the most recently added job
                print(f"Using job '{job_data.get('title', 'Unknown')}' from database for analysis")
            else:
                print("Error: No job data provided. Please specify skills or add a job to the database")
                return
        
        # Analyze resumes
        results = analyze_resumes(job_data, args.resumes, load_job_database(args.job_database))
        
        # Generate summary if requested
        if args.summary and results:
            rank_candidates(results)
    else:
        if not args.analyze_job:
            print("Error: Please provide resume files with --resumes or use --analyze-job to analyze a job description")
            return

# Profession-specific skills mapping
PROFESSION_SKILLS = {
    'software_developer': ['python', 'javascript', 'html', 'css', 'sql', 'git', 'react', 'node', 'java', 'c++'],
    'data_analyst': ['python', 'sql', 'excel', 'powerbi', 'tableau', 'r', 'pandas', 'numpy', 'statistics'],
    'data_scientist': ['python', 'machine learning', 'tensorflow', 'pytorch', 'sql', 'pandas', 'numpy', 'statistics', 'r'],
    'web_developer': ['html', 'css', 'javascript', 'react', 'angular', 'vue', 'node', 'php', 'sql', 'git'],
    'devops_engineer': ['docker', 'kubernetes', 'aws', 'azure', 'jenkins', 'terraform', 'linux', 'git', 'python'],
    'business_analyst': ['excel', 'sql', 'powerbi', 'tableau', 'requirements gathering', 'agile', 'scrum', 'jira'],
    'project_manager': ['agile', 'scrum', 'kanban', 'jira', 'trello', 'project management', 'leadership', 'communication'],
    'marketing_specialist': ['seo', 'sem', 'google analytics', 'social media', 'content marketing', 'email marketing', 'photoshop'],
    'ui_ux_designer': ['figma', 'sketch', 'photoshop', 'illustrator', 'ui design', 'ux design', 'adobe creative suite'],
    'cybersecurity_analyst': ['network security', 'ethical hacking', 'firewalls', 'encryption', 'linux', 'python', 'sql']
}

# Flask Web Application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    """Handle resume upload and analysis"""
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['resume']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text from PDF
        resume_text = pdf_reader(filepath)
        if not resume_text:
            os.remove(filepath)  # Clean up
            return jsonify({'error': 'Could not extract text from PDF'}), 400
        
        # Extract resume information
        resume_data = extract_info_custom(resume_text)

        # Get profession from form data
        profession = request.form.get('profession', 'software_developer')

        # Use profession-specific skills or fall back to job database
        if profession in PROFESSION_SKILLS:
            job_data = {
                'required_skills': PROFESSION_SKILLS[profession],
                'preferred_certifications': []
            }
        else:
            # Load job database for comparison (use default job if available)
            job_database = load_job_database('job_database.json')
            job_data = job_database[-1] if job_database else {
                'required_skills': ['python', 'javascript', 'sql'],  # Default skills
                'preferred_certifications': []
            }
        
        # Score resume
        score, matched_skills, missing_skills = score_resume(resume_data, job_data)
        suggestions = generate_suggestions(resume_data, job_data, score, missing_skills)
        
        # Calculate ATS score
        ats_result = calculate_ats_score(resume_text, resume_data)
        
        # Generate visualization
        chart_filename = f"skills_match_{os.path.splitext(filename)[0]}.png"
        chart_path = os.path.join('static', 'charts', chart_filename)
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        visualize_skills_match(matched_skills, missing_skills, chart_path)
        
        # Generate ATS score visualization
        ats_chart_filename = f"ats_score_{os.path.splitext(filename)[0]}.png"
        ats_chart_path = os.path.join('static', 'charts', ats_chart_filename)
        os.makedirs(os.path.dirname(ats_chart_path), exist_ok=True)
        visualize_ats_score(ats_result['score'], ats_chart_path)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Prepare response
        response = {
            'name': resume_data.get('name', 'Unknown'),
            'email': resume_data.get('email', ''),
            'phone': resume_data.get('phone', ''),
            'skills': resume_data.get('skills', []),
            'match_score': round(score, 2),
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'suggestions': suggestions,
            'chart_url': f'/static/charts/{chart_filename}',
            'ats_score': round(ats_result['score'], 2),
            'ats_issues': ats_result['issues'],
            'ats_recommendation': ats_result['recommendation'],
            'ats_chart_url': f'/static/charts/{ats_chart_filename}'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch-upload', methods=['POST'])
def batch_upload_resumes():
    """Handle batch resume upload and analysis"""
    try:
        if 'resumes' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('resumes')
        if not files or len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
        
        if len(files) > 20:
            return jsonify({'error': 'Maximum 20 resumes allowed at once'}), 400
        
        # Get profession from form data
        profession = request.form.get('profession', 'software_developer')
        
        # Use profession-specific skills
        if profession in PROFESSION_SKILLS:
            job_data = {
                'required_skills': PROFESSION_SKILLS[profession],
                'title': profession.replace('_', ' ').title(),
                'preferred_certifications': []
            }
        else:
            job_database = load_job_database('job_database.json')
            job_data = job_database[-1] if job_database else {
                'required_skills': ['python', 'javascript', 'sql'],
                'title': 'Software Developer'
            }
        
        # Save files temporarily
        temp_files = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                continue
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            temp_files.append(filepath)
        
        if not temp_files:
            return jsonify({'error': 'No valid PDF files found'}), 400
        
        # Analyze batch resumes
        batch_results = analyze_batch_resumes(temp_files, job_data, profession)
        
        # Generate comparison chart
        chart_filename = f"batch_comparison_{int(datetime.now().timestamp())}.png"
        chart_path = os.path.join('static', 'charts', chart_filename)
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        generate_batch_comparison_chart(batch_results, chart_path)
        
        # Export to CSV
        csv_filename = f"batch_results_{int(datetime.now().timestamp())}.csv"
        csv_path = os.path.join('static', 'exports', csv_filename)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        export_batch_results_csv(batch_results, csv_path)
        
        # Clean up temp files
        for filepath in temp_files:
            try:
                os.remove(filepath)
            except:
                pass
        
        # Prepare response
        response = {
            'success': True,
            'batch_results': batch_results,
            'comparison_chart': f'/static/charts/{chart_filename}',
            'csv_export': f'/static/exports/{csv_filename}',
            'job_title': job_data.get('title', 'Software Developer'),
            'total_candidates': len(batch_results)
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        print(f"Error in batch upload: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/download-export/<filename>')
def download_export(filename):
    """Download exported CSV file"""
    return send_from_directory(os.path.join('static', 'exports'), filename, as_attachment=True)

@app.route('/generate-cover-letter', methods=['POST'])
def generate_cover_letter_route():
    """Generate a customized cover letter"""
    try:
        data = request.get_json()
        
        if not data or 'resume_data' not in data or 'job_data' not in data:
            return jsonify({'error': 'Missing resume or job data'}), 400
        
        resume_data = data.get('resume_data')
        job_data = data.get('job_data')
        matched_skills = data.get('matched_skills', [])
        profession = data.get('profession', 'Software Development')
        
        # Generate cover letter
        cover_letter = generate_cover_letter(resume_data, job_data, matched_skills, profession)
        
        return jsonify({
            'success': True,
            'cover_letter': cover_letter
        })
        
    except Exception as e:
        import traceback
        print(f"Error in cover letter generation: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/static/charts/<filename>')
def serve_chart(filename):
    """Serve generated chart files"""
    return send_from_directory(os.path.join('static', 'charts'), filename)

@app.route('/get_chart/<filename>')
def get_chart(filename):
    """Serve generated chart files"""
    return send_from_directory(os.path.join('static', 'charts'), filename)

if __name__ == "__main__":
    # Check if running as web app or CLI
    if len(os.sys.argv) > 1:
        main()  # Run CLI version
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)  # Run web app
