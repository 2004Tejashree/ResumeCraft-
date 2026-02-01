ResumeCraft AI 📄✨
ResumeCraft AI is an intelligent, full-stack resume analysis tool designed to help job seekers optimize their CVs and clear Applicant Tracking Systems (ATS).
Using Natural Language Processing (NLP) and Machine Learning techniques, it parses resumes, matches them against job descriptions, and provides actionable insights to improve hiring chances.

🚀 Key Features
📄 AI Resume Parsing: Extracts contact info, skills, education, and experience from PDF resumes using pdfminer and spaCy.
🎯 Job Compatibility Score: Calculates a % match score based on semantic skill matching between the resume and job description.
🤖 ATS Verification: Analyzes resume formatting to ensure it is machine-readable and "robot-friendly."
📊 Visual Insights: Generates interactive charts (Matplotlib) to visualize matched vs. missing skills.
✍️ Smart Suggestions: Provides personalized feedback on how to improve the resume (e.g., "Add more metrics," "Include LinkedIn").
✉️ Cover Letter Generator: Auto-generates a tailored cover letter based on the candidate's profile and the specific job role.
📂 Batch Mode: Allows recruiters to upload multiple resumes and rank candidates automatically in a CSV report.
🛠️ Tech Stack
Backend: Python, Flask
NLP & ML: spaCy (en_core_web_sm), scikit-learn (TF-IDF), NLTK
Parsing: PDFMiner.six
Frontend: HTML5, CSS3 (Modern Glassmorphism Design), JavaScript
Visualization: Matplotlib
📦 Installation
1. Clone the repository:
git clone https://github.com/yourusername/ResumeCraft-AI.git

2. cd ResumeCraft-AI

3. Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install dependencies:
pip install -r requirements.txt
python -m spacy download en_core_web_sm

5. Run the application:
python app.py

<img width="1790" height="910" alt="image" src="https://github.com/user-attachments/assets/56288ef0-ba71-4394-b6e4-931c6bcb84bf" />
