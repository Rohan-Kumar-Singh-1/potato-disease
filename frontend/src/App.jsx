import { useState, useRef, useCallback } from "react";

const API_BASE = "http://localhost:8000";

const severityBadge = {
  none: { bg: "#14532d", text: "#bbf7d0", label: "Healthy" },
  moderate: { bg: "#713f12", text: "#fef08a", label: "Moderate Risk" },
  severe: { bg: "#7f1d1d", text: "#fecaca", label: "Severe Risk" },
};

export default function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef();

  const handleFile = (file) => {
    if (!file || !file.type.startsWith("image/")) {
      setError("Please upload a valid image file (JPG, PNG, WEBP).");
      return;
    }
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  const onDragOver = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);

  const predict = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", image);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed.");
      }
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div style={styles.root}>
      {/* Background texture */}
      <div style={styles.bgBlob1} />
      <div style={styles.bgBlob2} />

      <div style={styles.container}>
        {/* Header */}
        <header style={styles.header}>
          <div style={styles.logoRow}>
            <span style={styles.logoEmoji}>🥔</span>
            <div>
              <h1 style={styles.title}>PotatoScan</h1>
              <p style={styles.subtitle}>AI-powered potato disease detection</p>
            </div>
          </div>
          <div style={styles.badge}>CNN Model v1.0</div>
        </header>

        {/* Upload area */}
        <div
          style={{
            ...styles.uploadArea,
            borderColor: dragging ? "#84cc16" : preview ? "#4ade80" : "#334155",
            background: dragging ? "rgba(132,204,22,0.07)" : "rgba(15,23,42,0.6)",
          }}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={() => !preview && fileRef.current.click()}
        >
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => handleFile(e.target.files[0])}
          />

          {preview ? (
            <div style={styles.previewWrapper}>
              <img src={preview} alt="Preview" style={styles.previewImg} />
              <button style={styles.clearBtn} onClick={(e) => { e.stopPropagation(); reset(); }}>
                ✕ Remove
              </button>
            </div>
          ) : (
            <div style={styles.uploadPrompt}>
              <div style={styles.uploadIcon}>📷</div>
              <p style={styles.uploadText}>Drop a leaf image here</p>
              <p style={styles.uploadHint}>or click to browse · JPG, PNG, WEBP</p>
            </div>
          )}
        </div>

        {/* Predict button */}
        <button
          style={{
            ...styles.predictBtn,
            opacity: image && !loading ? 1 : 0.45,
            cursor: image && !loading ? "pointer" : "not-allowed",
          }}
          onClick={predict}
          disabled={!image || loading}
        >
          {loading ? (
            <span style={styles.spinnerRow}>
              <span style={styles.spinner} /> Analyzing…
            </span>
          ) : (
            "🔬 Analyze Disease"
          )}
        </button>

        {/* Error */}
        {error && (
          <div style={styles.errorBox}>
            <span>⚠️</span> {error}
          </div>
        )}

        {/* Result card */}
        {result && <ResultCard result={result} />}

        <footer style={styles.footer}>
          Built with FastAPI · TensorFlow Serving · React
        </footer>
      </div>
    </div>
  );
}

function ResultCard({ result }) {
  const badge = severityBadge[result.severity] || severityBadge.none;
  const topTwo = result.all_predictions.slice(0, 3);

  return (
    <div style={styles.resultCard}>
      {/* Main result */}
      <div style={styles.resultTop}>
        <span style={styles.resultEmoji}>{result.emoji}</span>
        <div style={{ flex: 1 }}>
          <div style={styles.resultLabelRow}>
            <h2 style={{ ...styles.resultLabel, color: result.color }}>{result.label}</h2>
            <span
              style={{
                ...styles.severityBadge,
                background: badge.bg,
                color: badge.text,
              }}
            >
              {badge.label}
            </span>
          </div>
          <p style={styles.confidenceText}>
            Confidence:{" "}
            <strong style={{ color: result.color }}>{result.confidence.toFixed(1)}%</strong>
          </p>
        </div>
      </div>

      {/* Confidence bar */}
      <div style={styles.barTrack}>
        <div
          style={{
            ...styles.barFill,
            width: `${result.confidence}%`,
            background: result.color,
          }}
        />
      </div>

      {/* Description */}
      <p style={styles.descText}>{result.description}</p>

      {/* Treatment */}
      <div style={styles.treatmentBox}>
        <p style={styles.treatmentTitle}>💊 Recommended Action</p>
        <p style={styles.treatmentText}>{result.treatment}</p>
      </div>

      {/* All class probabilities */}
      <div style={styles.allPreds}>
        <p style={styles.allPredsTitle}>All Predictions</p>
        {topTwo.map((p) => (
          <div key={p.class_key} style={styles.predRow}>
            <span style={styles.predLabel}>{p.label}</span>
            <div style={styles.predBarTrack}>
              <div
                style={{
                  ...styles.predBarFill,
                  width: `${p.probability}%`,
                  background: p.color,
                }}
              />
            </div>
            <span style={{ ...styles.predPct, color: p.color }}>{p.probability.toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Styles ─────────────────────────────────────────────── */
const styles = {
  root: {
    minHeight: "100vh",
    background: "#0b1120",
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "center",
    padding: "40px 16px 60px",
    fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
    color: "#e2e8f0",
    position: "relative",
    overflow: "hidden",
  },
  bgBlob1: {
    position: "fixed", top: "-200px", left: "-200px",
    width: "600px", height: "600px", borderRadius: "50%",
    background: "radial-gradient(circle, rgba(34,197,94,0.12) 0%, transparent 70%)",
    pointerEvents: "none",
  },
  bgBlob2: {
    position: "fixed", bottom: "-200px", right: "-200px",
    width: "500px", height: "500px", borderRadius: "50%",
    background: "radial-gradient(circle, rgba(132,204,22,0.1) 0%, transparent 70%)",
    pointerEvents: "none",
  },
  container: {
    width: "100%", maxWidth: "560px",
    display: "flex", flexDirection: "column", gap: "20px",
    position: "relative", zIndex: 1,
  },
  header: {
    display: "flex", alignItems: "center",
    justifyContent: "space-between", flexWrap: "wrap", gap: "12px",
  },
  logoRow: { display: "flex", alignItems: "center", gap: "14px" },
  logoEmoji: { fontSize: "48px", lineHeight: 1 },
  title: {
    margin: 0, fontSize: "28px", fontWeight: 800,
    background: "linear-gradient(90deg, #4ade80, #84cc16)",
    WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
    letterSpacing: "-0.5px",
  },
  subtitle: { margin: "2px 0 0", fontSize: "13px", color: "#64748b" },
  badge: {
    fontSize: "11px", fontWeight: 600, letterSpacing: "0.05em",
    padding: "5px 10px", borderRadius: "20px",
    background: "rgba(74,222,128,0.1)", color: "#4ade80",
    border: "1px solid rgba(74,222,128,0.25)",
  },
  uploadArea: {
    border: "2px dashed",
    borderRadius: "16px",
    padding: "32px 24px",
    cursor: "pointer",
    transition: "all 0.2s",
    minHeight: "220px",
    display: "flex", alignItems: "center", justifyContent: "center",
  },
  uploadPrompt: { textAlign: "center" },
  uploadIcon: { fontSize: "48px", marginBottom: "12px" },
  uploadText: { margin: "0 0 6px", fontSize: "16px", fontWeight: 600, color: "#cbd5e1" },
  uploadHint: { margin: 0, fontSize: "13px", color: "#475569" },
  previewWrapper: {
    display: "flex", flexDirection: "column", alignItems: "center", gap: "14px", width: "100%",
  },
  previewImg: {
    maxHeight: "260px", maxWidth: "100%", borderRadius: "10px",
    objectFit: "contain", boxShadow: "0 4px 30px rgba(0,0,0,0.4)",
  },
  clearBtn: {
    background: "rgba(239,68,68,0.15)", border: "1px solid rgba(239,68,68,0.3)",
    color: "#fca5a5", borderRadius: "8px", padding: "6px 14px",
    fontSize: "13px", cursor: "pointer", transition: "all 0.15s",
  },
  predictBtn: {
    width: "100%", padding: "16px",
    background: "linear-gradient(135deg, #16a34a, #65a30d)",
    border: "none", borderRadius: "12px",
    color: "#fff", fontSize: "16px", fontWeight: 700,
    cursor: "pointer", transition: "opacity 0.2s, transform 0.1s",
    letterSpacing: "0.02em",
  },
  spinnerRow: { display: "flex", alignItems: "center", justifyContent: "center", gap: "10px" },
  spinner: {
    display: "inline-block", width: "16px", height: "16px",
    border: "2px solid rgba(255,255,255,0.3)",
    borderTopColor: "#fff", borderRadius: "50%",
    animation: "spin 0.7s linear infinite",
  },
  errorBox: {
    background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)",
    borderRadius: "10px", padding: "12px 16px",
    fontSize: "14px", color: "#fca5a5", display: "flex", gap: "8px",
  },
  resultCard: {
    background: "rgba(15,23,42,0.8)",
    border: "1px solid rgba(74,222,128,0.2)",
    borderRadius: "16px", padding: "24px",
    backdropFilter: "blur(10px)",
    display: "flex", flexDirection: "column", gap: "16px",
  },
  resultTop: { display: "flex", alignItems: "flex-start", gap: "16px" },
  resultEmoji: { fontSize: "48px", lineHeight: 1 },
  resultLabelRow: { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "10px" },
  resultLabel: { margin: 0, fontSize: "22px", fontWeight: 800 },
  severityBadge: {
    fontSize: "11px", fontWeight: 700, letterSpacing: "0.06em",
    padding: "4px 10px", borderRadius: "20px", textTransform: "uppercase",
  },
  confidenceText: { margin: "6px 0 0", fontSize: "14px", color: "#94a3b8" },
  barTrack: {
    height: "8px", background: "#1e293b", borderRadius: "99px", overflow: "hidden",
  },
  barFill: { height: "100%", borderRadius: "99px", transition: "width 0.6s ease" },
  descText: { margin: 0, fontSize: "14px", color: "#94a3b8", lineHeight: 1.6 },
  treatmentBox: {
    background: "rgba(30,41,59,0.7)", borderRadius: "10px", padding: "14px 16px",
    border: "1px solid #1e293b",
  },
  treatmentTitle: { margin: "0 0 6px", fontSize: "13px", fontWeight: 700, color: "#e2e8f0" },
  treatmentText: { margin: 0, fontSize: "13px", color: "#94a3b8", lineHeight: 1.6 },
  allPreds: { display: "flex", flexDirection: "column", gap: "10px" },
  allPredsTitle: { margin: "0 0 4px", fontSize: "13px", fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.07em" },
  predRow: { display: "flex", alignItems: "center", gap: "10px" },
  predLabel: { fontSize: "13px", color: "#cbd5e1", width: "120px", flexShrink: 0 },
  predBarTrack: { flex: 1, height: "6px", background: "#1e293b", borderRadius: "99px", overflow: "hidden" },
  predBarFill: { height: "100%", borderRadius: "99px", transition: "width 0.5s ease" },
  predPct: { fontSize: "12px", fontWeight: 700, width: "42px", textAlign: "right", flexShrink: 0 },
  footer: { textAlign: "center", fontSize: "12px", color: "#334155", marginTop: "8px" },
};