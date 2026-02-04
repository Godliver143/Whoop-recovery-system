# Deploying to Streamlit Community Cloud

## Step-by-Step Guide

### 1. Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right → "New repository"
3. Name it: `whoop-recovery-system` (or your preferred name)
4. Choose Public (required for free Streamlit Cloud)
5. **Don't** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### 2. Initialize Git and Push Code

Open terminal in your project directory and run:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: Whoop Recovery Prediction System"

# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/whoop-recovery-system.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Important Notes:**
- Replace `YOUR_USERNAME` with your GitHub username
- Replace `whoop-recovery-system` with your repository name
- You may need to authenticate (use GitHub CLI, personal access token, or SSH)

### 3. Prepare for Streamlit Cloud

#### Option A: Include Models in Repository (Recommended for small models)

If your models are small enough (<100MB total), you can include them:

```bash
# Make sure saved_models/ directory exists with all models
git add saved_models/
git commit -m "Add trained models"
git push
```

#### Option B: Use Cloud Storage (For large models)

If models are too large, you'll need to:
1. Upload models to cloud storage (AWS S3, Google Cloud Storage, etc.)
2. Update `dashboard.py` to download models on startup
3. Add download code to dashboard.py

### 4. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Fill in the form:
   - **Repository**: Select your repository (`whoop-recovery-system`)
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `dashboard.py`
   - **App URL**: (auto-generated, or choose custom)
5. Click "Deploy"

### 5. Configure Environment (If Needed)

If your dashboard needs environment variables:
1. In Streamlit Cloud, go to your app settings
2. Click "Secrets"
3. Add any required secrets (API keys, etc.)

### 6. Update Dashboard for Cloud Deployment

If deploying dashboard separately from API, update the API URL in `dashboard.py`:

```python
# In dashboard.py, change:
API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")

# To your deployed API URL:
API_URL = st.sidebar.text_input("API URL", value="https://your-api-url.com")
```

Or set a default:
```python
API_URL = os.getenv("API_URL", "https://your-api-url.com")
```

## Troubleshooting

### Models Not Found Error

If you get errors about missing models:
1. Ensure `saved_models/` directory is committed to GitHub
2. Check file sizes (GitHub has 100MB file limit)
3. Use Git LFS for large files, or cloud storage

### Import Errors

If you get import errors:
1. Check `requirements.txt` includes all dependencies
2. Ensure all imports are available on PyPI
3. Check Streamlit Cloud logs for specific errors

### API Connection Issues

If dashboard can't connect to API:
1. Ensure API is deployed separately
2. Update API URL in dashboard
3. Check CORS settings on API
4. Verify API is publicly accessible

## Alternative: Deploy Both Dashboard and API

### Option 1: Separate Deployments
- Dashboard: Streamlit Cloud
- API: Heroku, Railway, Render, or AWS

### Option 2: Combined Deployment
- Use Streamlit Cloud for dashboard
- Embed API calls directly in dashboard (no separate API needed)
- Models loaded in dashboard itself

## Quick Checklist

- [ ] GitHub repository created
- [ ] Code pushed to GitHub
- [ ] `requirements.txt` is complete
- [ ] `README.md` exists
- [ ] Models are accessible (in repo or cloud storage)
- [ ] Streamlit Cloud app created
- [ ] App deployed successfully
- [ ] Tested all features

## Need Help?

- Streamlit Cloud Docs: https://docs.streamlit.io/streamlit-community-cloud
- GitHub Docs: https://docs.github.com
- Check Streamlit Cloud logs for deployment errors
