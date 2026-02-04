# Installation Instructions

## Required Dependencies

To run the Whoop Recovery System, you need to install the following Python packages:

### Option 1: Install from requirements.txt (Recommended)

```bash
pip install -r requirements.txt
```

### Option 2: Install packages individually

```bash
pip install streamlit plotly fastapi uvicorn requests pandas numpy scikit-learn tensorflow keras matplotlib seaborn
```

### Option 3: If you encounter SSL certificate errors

Try installing with trusted hosts:

```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org streamlit plotly fastapi uvicorn requests
```

Or use conda (if you have Anaconda/Miniconda):

```bash
conda install -c conda-forge streamlit plotly
pip install fastapi uvicorn requests
```

## Verify Installation

After installation, verify that packages are installed:

```bash
python -c "import streamlit; import plotly; import fastapi; print('All packages installed successfully!')"
```

## Running the System

1. **Start the API server:**
   ```bash
   python api.py
   ```
   Or use the script:
   ```bash
   ./start_api.sh
   ```

2. **Start the Dashboard** (in a new terminal):
   ```bash
   streamlit run dashboard.py
   ```
   Or use the script:
   ```bash
   ./start_dashboard.sh
   ```

## Troubleshooting

### ModuleNotFoundError
If you get `ModuleNotFoundError`, make sure you're using the correct Python environment:
- Check which Python: `which python` or `which python3`
- If using conda: `conda activate your_environment_name`
- If using venv: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)

### SSL Certificate Errors
If you encounter SSL errors:
1. Update pip: `python -m pip install --upgrade pip`
2. Try using conda instead of pip
3. Check your network/firewall settings

### Port Already in Use
If port 8000 or 8501 is already in use:
- Change the port in `api.py` (line with `uvicorn.run`)
- Or kill the process using the port:
  ```bash
  lsof -ti:8000 | xargs kill -9  # For port 8000
  lsof -ti:8501 | xargs kill -9  # For port 8501
  ```
