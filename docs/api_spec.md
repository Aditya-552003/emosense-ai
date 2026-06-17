# API Specification — Emotion Detection API

**Base URL (local):** `http://localhost:8080`

---

## `GET /`

**Health check and capability discovery.**

### Response `200 OK`
```json
{
  "status": "online",
  "api": "Emotion Detection API",
  "version": "1.0.0",
  "device": "cpu",
  "emotion_classes": ["anger", "fear", "joy", "neutral", "sadness", "surprise"],
  "endpoints": {
    "health":  "GET  /",
    "predict": "POST /predict  { text: string }"
  }
}
```

---

## `POST /predict`

**Classify the emotion expressed in a text string.**

### Request
```
Content-Type: application/json
```
```json
{ "text": "I just got promoted and I couldn't be happier!" }
```

| Field  | Type   | Max length | Required |
|--------|--------|------------|----------|
| `text` | string | 2000 chars | ✓        |

### Response `200 OK`
```json
{
  "success": true,
  "predicted_emotion": "joy",
  "confidence": 0.9898,
  "confidence_pct": "98.98%",
  "probabilities": {
    "anger":   0.0004,
    "fear":    0.0002,
    "joy":     0.9898,
    "neutral": 0.0021,
    "sadness": 0.0060,
    "surprise": 0.0015
  },
  "processing_time_ms": 287.4
}
```

### Validation Error `422 Unprocessable Entity`
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "text"],
      "msg": "Value error, Input text must not be empty.",
      "input": ""
    }
  ]
}
```

### Server Error `500 Internal Server Error`
```json
{ "success": false, "detail": "Internal server error." }
```

---

## Emotion Classes

| Class      | Emoji | Hex colour |
|------------|-------|------------|
| `joy`      | 😊    | `hsl(45, 95%, 58%)` |
| `sadness`  | 😢    | `hsl(220, 80%, 55%)` |
| `anger`    | 😡    | `hsl(5, 85%, 52%)` |
| `fear`     | 😨    | `hsl(270, 70%, 55%)` |
| `surprise` | 😲    | `hsl(190, 85%, 50%)` |
| `neutral`  | 😐    | `hsl(215, 15%, 55%)` |

---

## Example cURL calls

```bash
# Health check
curl http://localhost:8080/

# Predict
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I am so scared something terrible is going to happen."}'
```

---

## Deployment

### Render
1. Push `backend/` to GitHub.
2. New **Web Service** → Build: `pip install -r requirements.txt` → Start: `uvicorn app:create_app --factory --host 0.0.0.0 --port $PORT`.
3. Upload `model/` as a Persistent Disk mounted at `/app/model`.

### Railway
```bash
railway init && railway up
```
Set env var `START_COMMAND = uvicorn app:create_app --factory --host 0.0.0.0 --port $PORT`.

### Netlify / Vercel (frontend)
Drop the `frontend/` folder into Netlify drag-and-drop, or run `vercel --prod` from the `frontend/` directory. Update `API_BASE` in `js/config.js` to your production backend URL.
