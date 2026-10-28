# Derma.co — Derma-Scan Frontend & FastAPI Integration Guide

Welcome to **Derma.co**, a complete, multi-page, vanilla HTML5/CSS3/JS frontend for the **Derma-Scan** AI-assisted skin screening web application.

🌐 **Live Backend API:** https://derma-co.onrender.com
📘 **API Docs:** https://derma-co.onrender.com/docs

---

## 1. Running the FastAPI Backend Locally

Inside your `backend/app/` directory (where your `app.py` and trained `skin_classifier_new.pkl` live), run:

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
*(If your main FastAPI file is named `main.py`, replace `app:app` with `main:app`.)*

Verify the backend is alive by visiting `http://127.0.0.1:8000/` or `http://127.0.0.1:8000/docs`.

---

## 2. Serving the Frontend Locally

Open a second terminal inside `derma-scan/frontend/` and start a local static server on port `5500`:

```bash
cd derma-scan/frontend
python -m http.server 5500
```

Then open **`http://127.0.0.1:5500/index.html`** (or `http://localhost:5500/index.html`) in your browser.

---

## 3. FastAPI CORS Configuration (`localhost` vs `127.0.0.1`)

Browsers treat `http://127.0.0.1:5500` and `http://localhost:5500` as **two completely different origins**. Ensure your FastAPI `app.py` includes `CORSMiddleware` listing both local origins **and your deployed frontend URL**:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://prateek8683.github.io",   # production frontend (change if hosted elsewhere)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "ok", "service": "Derma-Scan API"}
```

---

## 4. Production Deployment Notes

- **Production API:** The frontend uses `https://derma-co.onrender.com` when it is not running on localhost. `API_BASE_URL` in `js/data.js` switches automatically:

```javascript
const IS_LOCAL = ["localhost", "127.0.0.1"].includes(window.location.hostname);
const API_BASE_URL = IS_LOCAL
  ? "http://127.0.0.1:8000"
  : "https://derma-co.onrender.com";
```

- **Render settings:**
  - Build Command: `pip install -r backend/requirements.txt`
  - Start Command: `cd backend/app && uvicorn app:app --host 0.0.0.0 --port $PORT`
  - Environment Variable: `PYTHON_VERSION = 3.11.9`
  - Health Check Path: `/`
- **Large model files:** Files over 100 MB cannot be stored in the GitHub repo. The trained model (`skin_classifier_new.pkl`) and `X_embeddings_new.npy` are published under [Releases](https://github.com/Prateek8683/Derma.co/releases), and the backend downloads the model automatically on first start if it is missing.
- **Cold start:** The free Render plan sleeps after 15 minutes of inactivity. The first request can take 30–60 seconds. The frontend pings `API_BASE_URL + "/"` on page load to wake the server.
- **No Frontend Secrets:** Never place private API keys, Hugging Face tokens, or database credentials inside any frontend `.js` or `.html` file.

---

## 5. Troubleshooting Common Issues

| Issue / Symptom | Cause & Fix |
| :--- | :--- |
| **CORS policy blocked request** | Ensure `http://127.0.0.1:5500`, `http://localhost:5500` and your production frontend URL are in FastAPI's `allow_origins`. |
| **`Failed to fetch` / Network error** | Backend is not running on port `8000` (local), or `API_BASE_URL` points to the wrong URL. On production, the Render service may be waking up, so wait about 60 seconds and retry. |
| **`422 Unprocessable Entity`** | FastAPI expects the multipart field name to be **`file`**. `js/scan.js` uses `formData.append("file", selectedFile)` and leaves `Content-Type` unset so the browser generates the boundary automatically. |
| **`413 Payload Too Large`** | The image exceeds the server or frontend size limit (default 10 MB in `js/data.js`). |
| **Render build fails: `backend/requirements.txt` not found** | Wrong path. Check the folder structure on GitHub and set the Render Root Directory / Build Command accordingly. |
| **Render app crashes / Out of memory** | Free plan has 512 MB RAM. Use `tensorflow-cpu` or a lighter model, or upgrade the plan. |
| **Model file not found on server** | Model is not in the repo. Make sure the Release asset URL in `app.py` is correct and public. |
| **GitHub push rejected (file over 100 MB)** | Add the file to `.gitignore`, remove it from history, and upload it to Releases or Hugging Face instead. |
| **Live Camera not starting** | `navigator.mediaDevices.getUserMedia` requires a secure context (`https://`, `http://localhost`, or `http://127.0.0.1`). Opening via `file://` or an insecure LAN IP blocks camera access. |
| **Changes to JS/CSS not appearing** | Hard-refresh your browser (`Ctrl + Shift + R` on Windows) to bypass cached static assets. |
