# ReviewTrust
## Detecting Suspicious Reviews Using Temporal and Network Analytics

ReviewTrust is a Python Flask web application that detects potentially suspicious or coordinated online review activity.

The system combines:

- Temporal review analysis
- Poisson-based burst detection
- Change-point/activity analysis
- Reviewer-product network analysis
- Jaccard similarity
- Union-Find clustering
- Random Forest classification
- Weighted score fusion
- Interactive dashboard visualization

> **Important:** ReviewTrust does not use an LLM, ChatGPT API, Claude API, or any external AI API for its fraud-scoring engine. The scoring is implemented locally in Python.

---

# 1. Project Structure

```text
reviewtrust/
├── app.py
├── requirements.txt
├── README.md
└── frontend/
    └── index.html
```

### Main files

- `app.py` — Flask backend, data processing, detection algorithms, ML model, scoring API, and server.
- `frontend/index.html` — dashboard interface.
- `requirements.txt` — Python packages required to run the project.
- `README.md` — complete installation, usage, dataset, and troubleshooting guide.

---

# 2. System Requirements

Before installing ReviewTrust, make sure the computer has:

- Windows 10/11, Linux, or macOS
- Python 3.10 or newer
- Internet connection for the first installation of Python packages
- Git (optional, but recommended for GitHub cloning)
- A modern web browser such as Chrome, Edge, or Firefox

## Check Python

Open Command Prompt / PowerShell / Terminal and run:

```bash
python --version
```

or on some Windows systems:

```bash
py --version
```

You should see a Python version such as:

```text
Python 3.10.x
```

or newer.

If Python is not installed, install it from the official Python website:

https://www.python.org/downloads/

### Windows Python installation

During Python installation, make sure to enable:

```text
Add Python.exe to PATH
```

Then finish the installation and reopen Command Prompt.

Verify again:

```bash
python --version
pip --version
```

---

# 3. Get the Project

## Option A — Clone from GitHub

After the project is uploaded to GitHub:

```bash
git clone https://github.com/YOUR_USERNAME/ReviewTrust-Fake-Review-Detection.git
```

Enter the project folder:

```bash
cd ReviewTrust-Fake-Review-Detection
```

If the repository contains the project inside another folder, enter that folder instead:

```bash
cd reviewtrust
```

## Option B — Download ZIP

If you downloaded the repository as a ZIP:

1. Extract the ZIP.
2. Open the extracted `reviewtrust` folder.
3. Open Command Prompt / PowerShell in that folder.

You should see:

```text
app.py
requirements.txt
README.md
frontend/
```

---

# 4. Create a Virtual Environment

A virtual environment keeps this project's Python packages separate from other Python projects.

From the project folder, run:

```bash
python -m venv venv
```

If `python` does not work on Windows, try:

```bash
py -m venv venv
```

---

# 5. Activate the Virtual Environment

## Windows Command Prompt

```bash
venv\Scripts\activate
```

## Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\venv\Scripts\Activate.ps1
```

## Linux / macOS

```bash
source venv/bin/activate
```

When activation succeeds, the terminal normally shows:

```text
(venv)
```

at the beginning of the command line.

---

# 6. Upgrade pip

With the virtual environment activated:

```bash
python -m pip install --upgrade pip
```

---

# 7. Install All Required Packages

Run:

```bash
pip install -r requirements.txt
```

This installs the packages used by ReviewTrust:

- Flask
- python-dateutil
- scikit-learn

`scikit-learn` is required because the application uses `RandomForestClassifier`, model training, train/test splitting, evaluation metrics, and prediction probabilities.

---

# 8. Verify the Installation

Run:

```bash
python -c "import flask, sklearn, dateutil; print('ReviewTrust dependencies installed successfully')"
```

If it prints:

```text
ReviewTrust dependencies installed successfully
```

the main Python dependencies are installed correctly.

---

# 9. Start ReviewTrust

Make sure you are inside the project folder and that the virtual environment is active.

Run:

```bash
python app.py
```

The Flask application starts on:

```text
http://127.0.0.1:5000/
```

Open that address in your browser.

The application serves both the dashboard and API from the same Flask process.

---

# 10. How to Stop the Application

Keep the terminal open while using the website.

To stop the server:

```text
Press CTRL + C
```

Closing the terminal also stops the local Flask server.

---

# 11. How to Run the Project Again Later

You do not need to reinstall everything every time.

Open the project folder and activate the environment.

### Windows:

```bash
venv\Scripts\activate
```

Then:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

---

# 12. How the Project Works

The main workflow is:

```text
Review Data
     ↓
Data Ingestion
     ↓
Preprocessing
     ↓
Temporal Analysis
     ↓
Reviewer Network Analysis
     ↓
Random Forest Classification
     ↓
Score Fusion
     ↓
Manipulation Score
     ↓
Interactive Dashboard
```

---

# 13. Algorithms Used

## 13.1 Poisson Burst Detection

The system checks whether an unusually large number of reviews appear within a short period.

The burst window used by the application is:

```text
15 minutes
```

A high number of reviews in a short period can indicate coordinated activity.

---

## 13.2 Change-Point Analysis

The system compares current review activity with previous hourly activity.

It considers changes in:

- Review volume
- Rating behavior

A sudden change produces a higher change-point signal.

---

## 13.3 Reviewer Network Analysis

Reviewers are connected based on shared review behavior and common products.

The system builds reviewer relationships and calculates network-based features.

---

## 13.4 Jaccard Similarity

Jaccard similarity measures how much two reviewers' product sets overlap.

```text
Jaccard Similarity =
Number of common products
-------------------------
Number of unique products
```

A higher value means the reviewers have more similar product-reviewing behavior.

---

## 13.5 Union-Find

Union-Find / Disjoint Set is used to group connected reviewers into components.

Connected components with multiple reviewers can be examined as potential reviewer groups or rings.

---

## 13.6 Random Forest

A Random Forest classifier provides an additional machine-learning risk signal.

The model uses behavioral and network-related features such as:

- Rating
- Reviewer review count
- Product review count
- Reviews in a burst
- Reviewer-product count
- Shared-product count

If labeled data is available, the application can train/evaluate the model on that data.

---

# 14. Manipulation Score

The final Manipulation Score combines four signals:

```text
Temporal Score       × 30%
Network Score        × 25%
ML Score             × 30%
Change-Point Score   × 15%
```

Conceptually:

```text
Manipulation Score =
    (Temporal × 0.30)
  + (Network × 0.25)
  + (ML × 0.30)
  + (Change Point × 0.15)
```

The final score is constrained to:

```text
0 – 100
```

A higher score indicates stronger suspicious/manipulation signals.

---

# 15. Dataset / CSV Input

ReviewTrust can analyze the built-in synthetic demonstration data or a user-uploaded CSV.

## Required information

The uploaded CSV needs:

- A reviewer column
- A product column
- Either a `minutes_ago` column OR a timestamp/date column

Rating is optional and defaults to 5 if it is not provided.

A label column is optional and is used when labeled data is available for machine-learning evaluation/training.

---

# 16. CSV Example

A simple supported CSV looks like:

```csv
reviewer,product,rating,timestamp,label
alex_42,Wireless Earbuds Pro,5,2026-07-11T14:32:00,1
jamie_88,Wireless Earbuds Pro,5,2026-07-11T14:34:00,1
sam_19,Wireless Earbuds Pro,5,2026-07-11T14:37:00,1
priya_61,Standing Desk Converter,4,2026-07-10T09:12:00,
noah_23,Insulated Water Bottle,5,2026-07-09T18:45:00,
```

The project also provides a CSV template through:

```text
GET /api/template
```

---

# 17. Flexible CSV Column Names

ReviewTrust does not require one exact column name.

For example, reviewer can be represented as:

```text
reviewer
user
username
customer
author
reviewer_id
```

Product can be:

```text
product
item
product_name
product_id
ASIN
```

Rating can be:

```text
rating
stars
score
overall
```

Time can be:

```text
minutes_ago
timestamp
date
datetime
created_at
review_date
```

The application maps these aliases automatically.

---

# 18. CSV Size Limit

The application has:

```text
MAX_ROWS = 6000
```

This is a practical limit because the reviewer-network comparison can become computationally expensive for very large datasets.

If a CSV contains more than 6000 rows, the application processes the first 6000 rows.

---

# 19. API Routes

The Flask backend provides these routes:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Opens the dashboard |
| `/api/health` | GET | Checks server health |
| `/api/sample` | GET | Runs the synthetic demo dataset |
| `/api/upload` | POST | Uploads and analyzes a CSV |
| `/api/template` | GET | Returns a CSV template |

---

# 20. Testing the API

After starting the application, open:

```text
http://127.0.0.1:5000/api/health
```

Expected result:

```json
{
  "ok": true
}
```

For the synthetic demo:

```text
http://127.0.0.1:5000/api/sample
```

---

# 21. Troubleshooting

## Problem: `python is not recognized`

Try:

```bash
py --version
```

If `py` works, use:

```bash
py -m venv venv
py -m pip install -r requirements.txt
py app.py
```

If neither works, reinstall Python and enable:

```text
Add Python.exe to PATH
```

---

## Problem: `pip is not recognized`

Use:

```bash
python -m pip install -r requirements.txt
```

instead of:

```bash
pip install -r requirements.txt
```

---

## Problem: `ModuleNotFoundError: No module named 'flask'`

Activate the virtual environment and reinstall:

```bash
venv\Scripts\activate
python -m pip install -r requirements.txt
```

---

## Problem: `ModuleNotFoundError: No module named 'sklearn'`

Install the missing dependency:

```bash
python -m pip install scikit-learn
```

Then run:

```bash
python app.py
```

---

## Problem: Port 5000 is already in use

The default port is:

```text
5000
```

Close the other program using the port, or change the `PORT` value near the top of `app.py`.

For example:

```python
PORT = 5001
```

Then run:

```bash
python app.py
```

and open:

```text
http://127.0.0.1:5001/
```

---

## Problem: PowerShell does not allow activation

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.\venv\Scripts\Activate.ps1
```

---

## Problem: Browser does not open automatically

Start:

```bash
python app.py
```

Then manually open:

```text
http://127.0.0.1:5000/
```

---

## Problem: CSV upload fails

Make sure the CSV contains:

```text
reviewer
product
```

and either:

```text
minutes_ago
```

or:

```text
timestamp/date/datetime
```

Also make sure those fields are not empty.

---

# 22. Development Server Note

The project uses Flask's built-in development server.

This is suitable for:

- College demonstrations
- Local testing
- Development

For public production deployment, use a production WSGI server and appropriate deployment configuration.

---

# 23. Technology Stack

### Backend

- Python
- Flask

### Machine Learning

- scikit-learn
- Random Forest

### Statistical / analytical methods

- Poisson burst detection
- Change-point/activity analysis

### Network analysis

- Jaccard similarity
- Union-Find clustering
- Reviewer-product relationships

### Frontend

- HTML
- CSS
- JavaScript

---

# 24. Project Output

The dashboard presents information such as:

- Review activity
- Temporal signals
- Reviewer networks
- Suspicious reviewer groups
- Manipulation scores
- Risk breakdown
- Machine-learning risk information
- Explanations for suspicious activity

---

# 25. Results

On the labeled demonstration dataset, the Random Forest component achieved the evaluation values reported in the project presentation:

- Accuracy: 95.1%
- Precision: 87.3%
- Recall: 98.2%
- F1-score: 92.4%

These are demonstration/synthetic-data results and should not be interpreted as performance on the full Amazon Reviews dataset.

---

# 26. Limitations

The current implementation is primarily designed around:

- Synthetic demonstration data
- User-uploaded CSV datasets
- Behavioral and structural review signals

The current system does not perform NLP-based review-text classification.

---

# 27. Future Scope

Possible future improvements include:

- Evaluation on the full Amazon Reviews dataset
- Review-text/NLP analysis
- Larger datasets
- More advanced graph algorithms
- Real-time review monitoring
- Additional machine-learning models
- Improved risk calibration

---

# 28. GitHub Installation — Quick Version

For someone who already has Python installed:

```bash
git clone https://github.com/YOUR_USERNAME/ReviewTrust-Fake-Review-Detection.git
cd ReviewTrust-Fake-Review-Detection
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

---

# 29. GitHub Files That Should NOT Be Uploaded

Do not commit:

```text
venv/
__pycache__/
*.pyc
.env
API keys
passwords
private/personal datasets
```

The virtual environment should be recreated using:

```bash
python -m venv venv
```

and dependencies should be installed using:

```bash
pip install -r requirements.txt
```

---

# 30. Project Team

**Project:** ReviewTrust

**Team Members:**

- S. Vijaya Persis
- K. Tharun Reddy

**Department:** CSE (AI & ML)

---

# 31. License

For an academic project, add an appropriate license only if you are comfortable making the source code reusable. If your college requires a particular license, follow the college/project guidelines.
