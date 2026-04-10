import { useState, useEffect, useRef, useCallback } from "react";

/* ─── Google Fonts ─────────────────────────────────────────── */
const FontLink = () => (
  <style>{`@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');`}</style>
);

/* ─── API Config ────────────────────────────────────────────── */
const API_BASE = "http://localhost:8000";

const apiFetch = async (endpoint, options = {}) => {
  const res = await fetch(`${API_BASE}${endpoint}`, options);
  if (!res.ok) throw new Error(`API Error ${res.status}: ${await res.text()}`);
  return res.json();
};

/* ─── Particle Canvas ───────────────────────────────────────── */
const ParticleField = () => {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let animId;
    const resize = () => { canvas.width = canvas.offsetWidth; canvas.height = canvas.offsetHeight; };
    resize();
    window.addEventListener("resize", resize);
    const particles = Array.from({ length: 80 }, () => ({
      x: Math.random() * canvas.width, y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3,
      r: Math.random() * 1.5 + 0.3, opacity: Math.random() * 0.5 + 0.1,
    }));
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.forEach(p => {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0) p.x = canvas.width; if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height; if (p.y > canvas.height) p.y = 0;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0,255,140,${p.opacity})`;
        ctx.fill();
      });
      particles.forEach((a, i) => particles.slice(i + 1).forEach(b => {
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d < 100) {
          ctx.beginPath();
          ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = `rgba(0,255,140,${0.06 * (1 - d / 100)})`;
          ctx.lineWidth = 0.5; ctx.stroke();
        }
      }));
      animId = requestAnimationFrame(draw);
    };
    draw();
    return () => { cancelAnimationFrame(animId); window.removeEventListener("resize", resize); };
  }, []);
  return <canvas ref={canvasRef} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }} />;
};

/* ─── Animated Counter ──────────────────────────────────────── */
const Counter = ({ value, suffix = "" }) => {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    let start = 0; const end = parseFloat(value) || 0;
    if (end === 0) return;
    const step = end / 40;
    const timer = setInterval(() => {
      start += step;
      if (start >= end) { setDisplay(end); clearInterval(timer); }
      else setDisplay(parseFloat(start.toFixed(2)));
    }, 30);
    return () => clearInterval(timer);
  }, [value]);
  return <span>{display}{suffix}</span>;
};

/* ─── Scan Line Effect ──────────────────────────────────────── */
const ScanLine = () => (
  <div style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "hidden", borderRadius: "inherit" }}>
    <div style={{
      position: "absolute", left: 0, right: 0, height: "2px",
      background: "linear-gradient(90deg, transparent, rgba(0,255,140,0.4), transparent)",
      animation: "scanline 3s linear infinite",
    }} />
  </div>
);

/* ─── Typewriter ────────────────────────────────────────────── */
const Typewriter = ({ text, speed = 40 }) => {
  const [displayed, setDisplayed] = useState("");
  useEffect(() => {
    setDisplayed("");
    let i = 0;
    const t = setInterval(() => {
      if (i < text.length) { setDisplayed(text.slice(0, ++i)); }
      else clearInterval(t);
    }, speed);
    return () => clearInterval(t);
  }, [text]);
  return <span>{displayed}<span style={{ animation: "blink 1s step-end infinite", color: "#00ff8c" }}>|</span></span>;
};

/* ─── Badge ─────────────────────────────────────────────────── */
const Badge = ({ label, value, color = "#00ff8c" }) => (
  <div style={{
    display: "inline-flex", alignItems: "center", gap: 6,
    background: `${color}14`, border: `1px solid ${color}40`,
    borderRadius: 4, padding: "3px 10px", fontSize: 11,
    fontFamily: "'Space Mono', monospace",
  }}>
    <span style={{ width: 6, height: 6, borderRadius: "50%", background: color, display: "inline-block", animation: "pulse 2s ease-in-out infinite" }} />
    <span style={{ color: `${color}99` }}>{label}:</span>
    <span style={{ color }}>{value}</span>
  </div>
);

/* ─── Progress Ring ─────────────────────────────────────────── */
const ProgressRing = ({ value = 0, size = 80, stroke = 5, color = "#00ff8c" }) => {
  const r = (size - stroke * 2) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (value / 100) * circ;
  return (
    <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(0,255,140,0.08)" strokeWidth={stroke} />
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color}
        strokeWidth={stroke} strokeDasharray={circ} strokeDashoffset={offset}
        strokeLinecap="round" style={{ transition: "stroke-dashoffset 1s ease" }} />
    </svg>
  );
};

/* ─── Entity Chip ───────────────────────────────────────────── */
const ENTITY_COLORS = {
  DRUG: "#00d4ff", DOSAGE: "#00ff8c", DATE: "#ffd700", DOCTOR: "#ff6b9d",
  MONETARY_VALUE: "#a78bfa", JURISDICTION: "#fb923c", PARTY: "#34d399",
  LAB_VALUE: "#60a5fa", INVOICE_NUMBER: "#f59e0b", PATIENT_AGE: "#e879f9",
  OBLIGATION: "#f87171", DURATION: "#4ade80", DEFAULT: "#94a3b8",
};
const EntityChip = ({ label, value }) => {
  const color = ENTITY_COLORS[label] || ENTITY_COLORS.DEFAULT;
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      background: `${color}12`, border: `1px solid ${color}35`,
      borderRadius: 20, padding: "4px 10px", fontSize: 11, margin: "3px 3px",
      fontFamily: "'Space Mono', monospace", cursor: "default",
      transition: "all 0.2s", color: "#e2e8f0",
    }}
      onMouseEnter={e => { e.currentTarget.style.background = `${color}25`; e.currentTarget.style.transform = "translateY(-1px)"; }}
      onMouseLeave={e => { e.currentTarget.style.background = `${color}12`; e.currentTarget.style.transform = "translateY(0)"; }}
    >
      <span style={{ fontSize: 9, color, fontWeight: 700 }}>{label}</span>
      <span style={{ color: "#94a3b8", margin: "0 2px" }}>·</span>
      <span style={{ color: "#e2e8f0", maxWidth: 140, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{value}</span>
    </span>
  );
};

/* ─── Doc Type Icon ─────────────────────────────────────────── */
const DOCTYPE_META = {
  medical_prescription: { label: "Medical Prescription", color: "#00ff8c", icon: "Rx" },
  lab_report: { label: "Lab Report", color: "#00d4ff", icon: "Lab" },
  legal_contract: { label: "Legal Contract", color: "#a78bfa", icon: "§" },
  affidavit: { label: "Affidavit", color: "#ffd700", icon: "A" },
  invoice: { label: "Invoice", color: "#fb923c", icon: "$" },
  unknown: { label: "Unknown", color: "#6b7280", icon: "?" },
};

/* ─── Stat Card ─────────────────────────────────────────────── */
const StatCard = ({ label, value, suffix, color = "#00ff8c", icon }) => (
  <div style={{
    background: "rgba(0,255,140,0.03)", border: "1px solid rgba(0,255,140,0.1)",
    borderRadius: 12, padding: "20px 24px", position: "relative", overflow: "hidden",
    transition: "all 0.3s",
  }}
    onMouseEnter={e => { e.currentTarget.style.borderColor = `${color}50`; e.currentTarget.style.background = `${color}06`; }}
    onMouseLeave={e => { e.currentTarget.style.borderColor = "rgba(0,255,140,0.1)"; e.currentTarget.style.background = "rgba(0,255,140,0.03)"; }}
  >
    <div style={{ fontSize: 11, color: "#4b5563", fontFamily: "'Space Mono', monospace", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 10 }}>{label}</div>
    <div style={{ fontSize: 28, fontFamily: "'Syne', sans-serif", fontWeight: 800, color }}>
      <Counter value={value} suffix={suffix} />
    </div>
    {icon && <div style={{ position: "absolute", right: 16, top: 16, fontSize: 20, opacity: 0.15, color }}>{icon}</div>}
  </div>
);

/* ─── Main App ──────────────────────────────────────────────── */
export default function OCRSystem() {
  const [activeTab, setActiveTab] = useState("analyze");
  const [file, setFile] = useState(null);
  const [textInput, setTextInput] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [healthStatus, setHealthStatus] = useState(null);
  const [trainStatus, setTrainStatus] = useState(null);
  const [trainLoading, setTrainLoading] = useState(false);
  const [preview, setPreview] = useState(null);
  const [mounted, setMounted] = useState(false);
  const [processingStep, setProcessingStep] = useState(0);
  const fileRef = useRef();

  useEffect(() => {
    setMounted(true);
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try { setHealthStatus(await apiFetch("/health")); }
    catch { setHealthStatus({ status: "offline", pipeline_loaded: false }); }
  };

  const PROCESSING_STEPS = ["Preprocessing image", "Running OCR engine", "Cleaning text", "Classifying document", "Extracting entities"];

  const simulateSteps = () => {
    setProcessingStep(0);
    let i = 0;
    const t = setInterval(() => {
      setProcessingStep(++i);
      if (i >= PROCESSING_STEPS.length) clearInterval(t);
    }, 600);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null); setProcessingStep(0);
    simulateSteps();
    try {
      const fd = new FormData(); fd.append("file", file);
      const data = await apiFetch("/analyze", { method: "POST", body: fd });
      setResult(data);
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  };

  const handleClassify = async () => {
    if (!textInput.trim()) return;
    setLoading(true); setError(null); setResult(null);
    simulateSteps();
    try {
      const data = await apiFetch("/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textInput }),
      });
      setResult(data);
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  };

  const handleTrain = async () => {
    setTrainLoading(true); setTrainStatus(null);
    try {
      const data = await apiFetch("/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ base_model: "bert-base-uncased", num_epochs: 5, learning_rate: 0.00002, n_per_class: 200 }),
      });
      setTrainStatus(data);
    } catch (e) { setTrainStatus({ status: "error", message: e.message }); }
    finally { setTrainLoading(false); }
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) { setFile(f); setPreview(f.type.startsWith("image/") ? URL.createObjectURL(f) : null); }
  }, []);

  const handleFileSelect = (e) => {
    const f = e.target.files[0];
    if (f) { setFile(f); setPreview(f.type.startsWith("image/") ? URL.createObjectURL(f) : null); }
  };

  const docMeta = result?.document ? DOCTYPE_META[result.document.type] || DOCTYPE_META.unknown : null;
  const confPct = result?.document ? Math.round(result.document.classification_confidence * 100) : 0;

  const TABS = [
    { id: "analyze", label: "Analyze Doc" },
    { id: "classify", label: "Classify Text" },
    { id: "train", label: "Train Model" },
    { id: "health", label: "System Status" },
  ];

  return (
    <>
      <FontLink />
      <style>{`
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #030712; color: #e2e8f0; font-family: 'DM Sans', sans-serif; }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: #0a0f1a; } ::-webkit-scrollbar-thumb { background: #00ff8c30; border-radius: 2px; }
        @keyframes scanline { 0% { top: -2px; } 100% { top: 100%; } }
        @keyframes blink { 50% { opacity: 0; } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes glitch { 0%,100% { clip-path: inset(0 0 98% 0); transform: translateX(0); } 10% { clip-path: inset(10% 0 60% 0); transform: translateX(-3px); } 20% { clip-path: inset(50% 0 30% 0); transform: translateX(3px); } 30% { clip-path: inset(80% 0 5% 0); transform: translateX(-2px); } }
        @keyframes rotateRing { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes floatUp { 0% { opacity: 0; transform: translateY(0); } 50% { opacity: 1; } 100% { opacity: 0; transform: translateY(-40px); } }
        @keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
        @keyframes borderPulse { 0%,100% { border-color: rgba(0,255,140,0.2); } 50% { border-color: rgba(0,255,140,0.6); } }
        .tab-btn:hover { background: rgba(0,255,140,0.08) !important; }
        .upload-zone:hover { border-color: rgba(0,255,140,0.5) !important; background: rgba(0,255,140,0.04) !important; }
        .action-btn:hover { background: rgba(0,255,140,0.15) !important; transform: translateY(-1px); box-shadow: 0 8px 30px rgba(0,255,140,0.2) !important; }
        .action-btn:active { transform: translateY(0) scale(0.98); }
        .action-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none !important; }
      `}</style>

      <div style={{ minHeight: "100vh", background: "#030712", position: "relative", overflow: "hidden" }}>

        {/* Background layers */}
        <div style={{ position: "fixed", inset: 0, pointerEvents: "none" }}>
          <ParticleField />
          <div style={{ position: "absolute", inset: 0, backgroundImage: "radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,255,140,0.06) 0%, transparent 60%), radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0,100,255,0.04) 0%, transparent 50%)" }} />
          <div style={{ position: "absolute", inset: 0, backgroundImage: "linear-gradient(rgba(0,255,140,0.015) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,140,0.015) 1px, transparent 1px)", backgroundSize: "60px 60px" }} />
          <div style={{ position: "absolute", inset: 0, background: "linear-gradient(to bottom, transparent 80%, #030712 100%)" }} />
        </div>

        <div style={{ position: "relative", zIndex: 1, maxWidth: 1100, margin: "0 auto", padding: "0 24px 80px" }}>

          {/* ── Header ── */}
          <div style={{ padding: "60px 0 40px", animation: "fadeUp 0.8s ease forwards" }}>
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: 20 }}>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
                  <div style={{ position: "relative", width: 44, height: 44 }}>
                    <div style={{ position: "absolute", inset: 0, border: "2px solid rgba(0,255,140,0.6)", borderRadius: 10, animation: "rotateRing 8s linear infinite" }} />
                    <div style={{ position: "absolute", inset: 4, background: "rgba(0,255,140,0.1)", borderRadius: 7, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, fontFamily: "'Space Mono', monospace", color: "#00ff8c", fontWeight: 700 }}>O</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, fontFamily: "'Space Mono', monospace", color: "#00ff8c", letterSpacing: "0.2em", marginBottom: 2, textTransform: "uppercase" }}>v2.0 — Intelligence Engine</div>
                    <h1 style={{ fontSize: "clamp(22px, 4vw, 36px)", fontFamily: "'Syne', sans-serif", fontWeight: 800, color: "#f1f5f9", lineHeight: 1, letterSpacing: "-0.02em" }}>
                      Multi-Modal OCR System
                    </h1>
                  </div>
                </div>
                <p style={{ fontSize: 14, color: "#64748b", maxWidth: 480, lineHeight: 1.6, fontFamily: "'DM Sans', sans-serif" }}>
                  Legal &amp; medical document intelligence — preprocessing, OCR, classification, entity extraction, and BERT fine-tuning in one unified pipeline.
                </p>
              </div>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ width: 8, height: 8, borderRadius: "50%", background: healthStatus?.status === "ok" ? "#00ff8c" : "#ef4444", animation: "pulse 2s ease-in-out infinite" }} />
                  <span style={{ fontSize: 12, fontFamily: "'Space Mono', monospace", color: healthStatus?.status === "ok" ? "#00ff8c" : "#ef4444" }}>
                    {healthStatus?.status === "ok" ? "API ONLINE" : "API OFFLINE"}
                  </span>
                  <button onClick={checkHealth} style={{ background: "none", border: "1px solid rgba(0,255,140,0.2)", borderRadius: 4, padding: "3px 8px", fontSize: 10, color: "#00ff8c", cursor: "pointer", fontFamily: "'Space Mono', monospace" }}>PING</button>
                </div>
                {healthStatus && (
                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "flex-end" }}>
                    <Badge label="PIPELINE" value={healthStatus.pipeline_loaded ? "LOADED" : "STANDBY"} color={healthStatus.pipeline_loaded ? "#00ff8c" : "#ffd700"} />
                    <Badge label="VER" value={healthStatus.version || "2.0.0"} />
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ── Tabs ── */}
          <div style={{ display: "flex", gap: 4, marginBottom: 28, background: "rgba(0,0,0,0.4)", border: "1px solid rgba(0,255,140,0.1)", borderRadius: 10, padding: 4, width: "fit-content", animation: "fadeUp 0.8s 0.1s ease both" }}>
            {TABS.map(t => (
              <button key={t.id} className="tab-btn" onClick={() => setActiveTab(t.id)} style={{
                padding: "9px 20px", borderRadius: 7, border: "none", cursor: "pointer",
                fontFamily: "'DM Sans', sans-serif", fontSize: 13, fontWeight: 500,
                background: activeTab === t.id ? "rgba(0,255,140,0.12)" : "transparent",
                color: activeTab === t.id ? "#00ff8c" : "#475569",
                borderBottom: activeTab === t.id ? "2px solid #00ff8c" : "2px solid transparent",
                transition: "all 0.2s",
              }}>{t.label}</button>
            ))}
          </div>

          {/* ── ANALYZE TAB ── */}
          {activeTab === "analyze" && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, animation: "fadeUp 0.6s ease forwards" }}>

              {/* Upload Zone */}
              <div>
                <div className="upload-zone"
                  onDrop={handleDrop} onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)} onClick={() => fileRef.current.click()}
                  style={{
                    border: `2px dashed ${dragOver ? "rgba(0,255,140,0.6)" : "rgba(0,255,140,0.2)"}`,
                    borderRadius: 16, padding: "40px 24px", textAlign: "center",
                    cursor: "pointer", background: dragOver ? "rgba(0,255,140,0.06)" : "rgba(0,255,140,0.02)",
                    transition: "all 0.25s", position: "relative", overflow: "hidden",
                    minHeight: 220, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                    animation: "borderPulse 3s ease-in-out infinite",
                  }}>
                  <ScanLine />
                  <input ref={fileRef} type="file" hidden accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp" onChange={handleFileSelect} />
                  {preview ? (
                    <img src={preview} alt="preview" style={{ maxHeight: 140, maxWidth: "100%", objectFit: "contain", borderRadius: 8, opacity: 0.9 }} />
                  ) : (
                    <>
                      <div style={{ fontSize: 40, marginBottom: 12, opacity: 0.3 }}>⬆</div>
                      <div style={{ fontSize: 15, fontFamily: "'Syne', sans-serif", fontWeight: 600, color: "#94a3b8", marginBottom: 6 }}>Drop document here</div>
                      <div style={{ fontSize: 12, color: "#4b5563" }}>PDF, PNG, JPG, TIFF — max 50 MB</div>
                    </>
                  )}
                </div>
                {file && (
                  <div style={{ marginTop: 10, padding: "10px 14px", background: "rgba(0,255,140,0.06)", border: "1px solid rgba(0,255,140,0.15)", borderRadius: 8, display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontSize: 18 }}>📄</span>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 13, color: "#e2e8f0", fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{file.name}</div>
                      <div style={{ fontSize: 11, color: "#64748b", fontFamily: "'Space Mono', monospace" }}>{(file.size / 1024).toFixed(1)} KB</div>
                    </div>
                    <button onClick={() => { setFile(null); setPreview(null); setResult(null); }}
                      style={{ background: "none", border: "none", color: "#ef4444", cursor: "pointer", fontSize: 16, opacity: 0.6 }}>✕</button>
                  </div>
                )}
                <button className="action-btn" onClick={handleAnalyze} disabled={!file || loading} style={{
                  width: "100%", marginTop: 14, padding: "14px 0",
                  background: "rgba(0,255,140,0.1)", border: "1px solid rgba(0,255,140,0.35)",
                  borderRadius: 10, color: "#00ff8c", fontFamily: "'Syne', sans-serif",
                  fontSize: 14, fontWeight: 700, cursor: "pointer", letterSpacing: "0.05em",
                  transition: "all 0.2s", position: "relative", overflow: "hidden",
                }}>
                  {loading ? "PROCESSING..." : "RUN FULL PIPELINE →"}
                </button>
              </div>

              {/* Processing / Results panel */}
              <div>
                {loading && (
                  <div style={{ background: "rgba(0,0,0,0.4)", border: "1px solid rgba(0,255,140,0.15)", borderRadius: 16, padding: 28 }}>
                    <div style={{ fontSize: 13, fontFamily: "'Space Mono', monospace", color: "#00ff8c", marginBottom: 20, letterSpacing: "0.1em" }}>PIPELINE RUNNING...</div>
                    {PROCESSING_STEPS.map((step, i) => (
                      <div key={i} style={{
                        display: "flex", alignItems: "center", gap: 12, padding: "10px 0",
                        borderBottom: i < PROCESSING_STEPS.length - 1 ? "1px solid rgba(0,255,140,0.06)" : "none",
                        opacity: i < processingStep ? 1 : 0.3, transition: "opacity 0.4s",
                      }}>
                        <div style={{
                          width: 24, height: 24, borderRadius: "50%", flexShrink: 0,
                          border: i < processingStep ? "2px solid #00ff8c" : "2px solid rgba(0,255,140,0.2)",
                          background: i < processingStep ? "rgba(0,255,140,0.15)" : "transparent",
                          display: "flex", alignItems: "center", justifyContent: "center",
                          fontSize: 11, color: "#00ff8c", fontFamily: "'Space Mono', monospace",
                          transition: "all 0.3s",
                        }}>{i < processingStep ? "✓" : i + 1}</div>
                        <div>
                          <div style={{ fontSize: 13, color: i < processingStep ? "#e2e8f0" : "#64748b", fontWeight: 500 }}>{step}</div>
                          {i < processingStep && <div style={{ fontSize: 11, color: "#00ff8c", fontFamily: "'Space Mono', monospace" }}>complete</div>}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {error && (
                  <div style={{ background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 16, padding: 24 }}>
                    <div style={{ fontSize: 12, fontFamily: "'Space Mono', monospace", color: "#ef4444", marginBottom: 8 }}>ERROR</div>
                    <div style={{ fontSize: 13, color: "#fca5a5" }}>{error}</div>
                    <div style={{ marginTop: 12, fontSize: 12, color: "#6b7280" }}>Make sure the API server is running at {API_BASE}</div>
                  </div>
                )}

                {result && !loading && <ResultPanel result={result} docMeta={docMeta} confPct={confPct} />}

                {!loading && !result && !error && (
                  <div style={{ background: "rgba(0,0,0,0.3)", border: "1px solid rgba(0,255,140,0.06)", borderRadius: 16, padding: 40, textAlign: "center" }}>
                    <div style={{ fontSize: 13, color: "#374151", fontFamily: "'Space Mono', monospace", lineHeight: 2 }}>
                      Upload a document<br />and run the pipeline<br />to see results here
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── CLASSIFY TEXT TAB ── */}
          {activeTab === "classify" && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, animation: "fadeUp 0.6s ease forwards" }}>
              <div>
                <div style={{ fontSize: 12, fontFamily: "'Space Mono', monospace", color: "#00ff8c", letterSpacing: "0.1em", marginBottom: 10 }}>INPUT TEXT</div>
                <textarea value={textInput} onChange={e => setTextInput(e.target.value)}
                  placeholder="Paste document text here — prescription, contract, lab report, invoice, or affidavit..."
                  style={{
                    width: "100%", height: 240, background: "rgba(0,0,0,0.5)", border: "1px solid rgba(0,255,140,0.15)",
                    borderRadius: 12, padding: 16, color: "#e2e8f0", fontFamily: "'DM Sans', sans-serif",
                    fontSize: 13, lineHeight: 1.6, resize: "vertical", outline: "none",
                    transition: "border-color 0.2s",
                  }}
                  onFocus={e => e.target.style.borderColor = "rgba(0,255,140,0.4)"}
                  onBlur={e => e.target.style.borderColor = "rgba(0,255,140,0.15)"}
                />
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 8, marginBottom: 14 }}>
                  <span style={{ fontSize: 11, color: "#374151", fontFamily: "'Space Mono', monospace" }}>{textInput.length} chars</span>
                  <button onClick={() => setTextInput("")} style={{ background: "none", border: "none", color: "#4b5563", fontSize: 11, cursor: "pointer", fontFamily: "'Space Mono', monospace" }}>CLEAR</button>
                </div>
                <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
                  {["SAMPLE: PRESCRIPTION", "SAMPLE: LEGAL", "SAMPLE: LAB"].map((label, i) => (
                    <button key={i} onClick={() => setTextInput([
                      "MEDICAL PRESCRIPTION\nPatient: John Doe, 45M\nDate: 14/03/2024\nRx:\n1. Amoxicillin 500mg — twice daily for 7 days\n2. Ibuprofen 400mg — thrice daily after meals\nPrescribed by: Dr. Sarah Williams\nRefills: 0",
                      "SERVICE AGREEMENT\nThis agreement is entered on 10/03/2024 between:\nXYZ Technology Solutions Pvt. Ltd. (hereinafter Service Provider)\nAND ABC Retail Corporation (hereinafter Client)\n1. PAYMENT: Client shall pay USD 15,000 per month.\n2. TERMINATION: 30 days written notice required.\n3. JURISDICTION: Governed by laws of State of California.",
                      "PATHCARE DIAGNOSTICS — LAB REPORT\nPatient: Jane Smith\nHbA1c: 6.1% — ELEVATED\nFasting Glucose: 108 mg/dL — BORDERLINE\nHemoglobin: 11.2 g/dL — LOW\nWBC: 7.5 K/uL — NORMAL",
                    ][i])} style={{
                      background: "rgba(0,255,140,0.05)", border: "1px solid rgba(0,255,140,0.15)",
                      borderRadius: 4, padding: "4px 10px", fontSize: 10, color: "#00ff8c88",
                      cursor: "pointer", fontFamily: "'Space Mono', monospace", transition: "all 0.2s",
                    }}
                      onMouseEnter={e => e.currentTarget.style.borderColor = "rgba(0,255,140,0.4)"}
                      onMouseLeave={e => e.currentTarget.style.borderColor = "rgba(0,255,140,0.15)"}
                    >{label}</button>
                  ))}
                </div>
                <button className="action-btn" onClick={handleClassify} disabled={!textInput.trim() || loading} style={{
                  width: "100%", padding: "14px 0", background: "rgba(0,255,140,0.1)",
                  border: "1px solid rgba(0,255,140,0.35)", borderRadius: 10, color: "#00ff8c",
                  fontFamily: "'Syne', sans-serif", fontSize: 14, fontWeight: 700, cursor: "pointer",
                  letterSpacing: "0.05em", transition: "all 0.2s",
                }}>{loading ? "CLASSIFYING..." : "CLASSIFY + EXTRACT →"}</button>
              </div>
              <div>
                {error && (
                  <div style={{ background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 16, padding: 24 }}>
                    <div style={{ fontSize: 12, fontFamily: "'Space Mono', monospace", color: "#ef4444", marginBottom: 8 }}>ERROR</div>
                    <div style={{ fontSize: 13, color: "#fca5a5" }}>{error}</div>
                  </div>
                )}
                {result && !loading && <ResultPanel result={result} docMeta={docMeta} confPct={confPct} />}
                {loading && (
                  <div style={{ background: "rgba(0,0,0,0.4)", border: "1px solid rgba(0,255,140,0.15)", borderRadius: 16, padding: 28, textAlign: "center" }}>
                    <div style={{ fontSize: 13, fontFamily: "'Space Mono', monospace", color: "#00ff8c", animation: "pulse 1s ease-in-out infinite" }}>ANALYZING TEXT...</div>
                  </div>
                )}
                {!loading && !result && !error && (
                  <div style={{ background: "rgba(0,0,0,0.3)", border: "1px solid rgba(0,255,140,0.06)", borderRadius: 16, padding: 40, textAlign: "center" }}>
                    <div style={{ fontSize: 13, color: "#374151", fontFamily: "'Space Mono', monospace", lineHeight: 2 }}>Paste or select a sample,<br />then classify</div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── TRAIN TAB ── */}
          {activeTab === "train" && (
            <div style={{ maxWidth: 640, animation: "fadeUp 0.6s ease forwards" }}>
              <div style={{ background: "rgba(0,0,0,0.4)", border: "1px solid rgba(0,255,140,0.12)", borderRadius: 16, padding: 32 }}>
                <div style={{ fontSize: 12, fontFamily: "'Space Mono', monospace", color: "#00ff8c", letterSpacing: "0.1em", marginBottom: 6 }}>BERT FINE-TUNING</div>
                <h2 style={{ fontFamily: "'Syne', sans-serif", fontSize: 22, fontWeight: 700, color: "#f1f5f9", marginBottom: 10 }}>Train Document Classifier</h2>
                <p style={{ fontSize: 13, color: "#64748b", lineHeight: 1.7, marginBottom: 28 }}>
                  Triggers fine-tuning of bert-base-uncased on synthetic medical and legal document data. Training runs in the background — monitor progress via MLflow at <span style={{ color: "#00ff8c", fontFamily: "'Space Mono', monospace" }}>localhost:5000</span>.
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 24 }}>
                  {[["Model", "bert-base-uncased"], ["Epochs", "5"], ["Learning Rate", "2e-5"], ["Samples/class", "200"]].map(([k, v]) => (
                    <div key={k} style={{ background: "rgba(0,255,140,0.04)", border: "1px solid rgba(0,255,140,0.1)", borderRadius: 8, padding: "12px 16px" }}>
                      <div style={{ fontSize: 10, color: "#4b5563", fontFamily: "'Space Mono', monospace", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>{k}</div>
                      <div style={{ fontSize: 14, fontFamily: "'Space Mono', monospace", color: "#00ff8c" }}>{v}</div>
                    </div>
                  ))}
                </div>
                <button className="action-btn" onClick={handleTrain} disabled={trainLoading} style={{
                  width: "100%", padding: "14px 0", background: "rgba(167,139,250,0.1)",
                  border: "1px solid rgba(167,139,250,0.35)", borderRadius: 10, color: "#a78bfa",
                  fontFamily: "'Syne', sans-serif", fontSize: 14, fontWeight: 700,
                  cursor: "pointer", letterSpacing: "0.05em", transition: "all 0.2s",
                }}>{trainLoading ? "STARTING..." : "TRIGGER TRAINING →"}</button>
                {trainStatus && (
                  <div style={{
                    marginTop: 16, padding: "14px 18px",
                    background: trainStatus.status === "error" ? "rgba(239,68,68,0.06)" : "rgba(0,255,140,0.06)",
                    border: `1px solid ${trainStatus.status === "error" ? "rgba(239,68,68,0.3)" : "rgba(0,255,140,0.2)"}`,
                    borderRadius: 10,
                  }}>
                    <div style={{ fontSize: 12, fontFamily: "'Space Mono', monospace", color: trainStatus.status === "error" ? "#ef4444" : "#00ff8c", marginBottom: 4 }}>
                      {trainStatus.status?.toUpperCase()}
                    </div>
                    <div style={{ fontSize: 13, color: "#94a3b8" }}>{trainStatus.message}</div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── HEALTH TAB ── */}
          {activeTab === "health" && (
            <div style={{ animation: "fadeUp 0.6s ease forwards" }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14, marginBottom: 20 }}>
                <StatCard label="API Status" value={healthStatus?.status === "ok" ? 100 : 0} suffix="%" color="#00ff8c" icon="●" />
                <StatCard label="Pipeline" value={healthStatus?.pipeline_loaded ? 100 : 0} suffix="%" color="#00d4ff" icon="◈" />
                <StatCard label="Version" value={2} suffix=".0" color="#a78bfa" icon="v" />
              </div>
              {healthStatus && (
                <div style={{ background: "rgba(0,0,0,0.4)", border: "1px solid rgba(0,255,140,0.1)", borderRadius: 16, padding: 28 }}>
                  <div style={{ fontSize: 12, fontFamily: "'Space Mono', monospace", color: "#00ff8c", letterSpacing: "0.1em", marginBottom: 16 }}>SYSTEM CONFIG</div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 12 }}>
                    {Object.entries(healthStatus.config || {}).map(([k, v]) => (
                      <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "10px 14px", background: "rgba(0,0,0,0.3)", borderRadius: 8, border: "1px solid rgba(0,255,140,0.06)" }}>
                        <span style={{ fontSize: 12, color: "#64748b", fontFamily: "'Space Mono', monospace" }}>{k}</span>
                        <span style={{ fontSize: 12, color: "#e2e8f0", fontFamily: "'Space Mono', monospace" }}>{String(v)}</span>
                      </div>
                    ))}
                  </div>
                  <button onClick={checkHealth} style={{
                    marginTop: 20, padding: "10px 24px", background: "rgba(0,255,140,0.06)",
                    border: "1px solid rgba(0,255,140,0.2)", borderRadius: 8,
                    color: "#00ff8c", fontFamily: "'Space Mono', monospace", fontSize: 12,
                    cursor: "pointer", letterSpacing: "0.08em",
                  }}>REFRESH STATUS</button>
                </div>
              )}
              {!healthStatus && (
                <div style={{ padding: 32, textAlign: "center", color: "#374151", fontFamily: "'Space Mono', monospace", fontSize: 13 }}>
                  API unreachable — start the server with: <span style={{ color: "#00ff8c" }}>python run.py serve</span>
                </div>
              )}
            </div>
          )}

        </div>
      </div>
    </>
  );
}

/* ─── Result Panel Component ─────────────────────────────────── */
function ResultPanel({ result, docMeta, confPct }) {
  if (!result) return null;
  const entities = result.entities?.entities_by_type || {};
  const totalEntities = result.entities?.entity_count || 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, animation: "fadeUp 0.5s ease forwards" }}>

      {/* Doc type hero */}
      <div style={{
        background: `linear-gradient(135deg, ${docMeta?.color}10 0%, rgba(0,0,0,0.5) 100%)`,
        border: `1px solid ${docMeta?.color}35`, borderRadius: 16, padding: "20px 24px",
        display: "flex", alignItems: "center", gap: 18, position: "relative", overflow: "hidden",
      }}>
        <ScanLine />
        <div style={{
          width: 52, height: 52, borderRadius: 12, background: `${docMeta?.color}18`,
          border: `2px solid ${docMeta?.color}40`, display: "flex", alignItems: "center",
          justifyContent: "center", fontSize: 20, fontFamily: "'Space Mono', monospace",
          color: docMeta?.color, fontWeight: 700, flexShrink: 0,
        }}>{docMeta?.icon}</div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 11, fontFamily: "'Space Mono', monospace", color: `${docMeta?.color}88`, letterSpacing: "0.12em", marginBottom: 4 }}>CLASSIFIED AS</div>
          <div style={{ fontSize: 18, fontFamily: "'Syne', sans-serif", fontWeight: 700, color: "#f1f5f9", lineHeight: 1 }}>{docMeta?.label}</div>
          <div style={{ fontSize: 11, color: "#64748b", marginTop: 4, fontFamily: "'Space Mono', monospace" }}>via {result.document?.classification_method}</div>
        </div>
        <div style={{ position: "relative", flexShrink: 0 }}>
          <ProgressRing value={confPct} color={docMeta?.color} size={72} stroke={4} />
          <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
            <span style={{ fontSize: 14, fontWeight: 700, color: docMeta?.color, fontFamily: "'Space Mono', monospace" }}>{confPct}%</span>
            <span style={{ fontSize: 8, color: "#64748b", fontFamily: "'Space Mono', monospace" }}>CONF</span>
          </div>
        </div>
      </div>

      {/* Review flag */}
      {result.requires_human_review && (
        <div style={{ background: "rgba(255,215,0,0.06)", border: "1px solid rgba(255,215,0,0.3)", borderRadius: 10, padding: "10px 14px", display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 16 }}>⚠</span>
          <div>
            <span style={{ fontSize: 12, color: "#ffd700", fontFamily: "'Space Mono', monospace" }}>HUMAN REVIEW REQUIRED</span>
            <div style={{ fontSize: 11, color: "#92400e", marginTop: 2 }}>{result.review_reasons?.join(", ")}</div>
          </div>
        </div>
      )}

      {/* Stats row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
        {[
          { label: "Words", value: result.ocr?.word_count || 0 },
          { label: "Entities", value: totalEntities },
          { label: "Latency", value: `${Math.round(result.performance?.total_latency_ms || 0)}ms` },
        ].map(s => (
          <div key={s.label} style={{ background: "rgba(0,0,0,0.4)", border: "1px solid rgba(0,255,140,0.08)", borderRadius: 8, padding: "10px 12px", textAlign: "center" }}>
            <div style={{ fontSize: 16, fontFamily: "'Syne', sans-serif", fontWeight: 700, color: "#e2e8f0" }}>{s.value}</div>
            <div style={{ fontSize: 10, color: "#4b5563", fontFamily: "'Space Mono', monospace", textTransform: "uppercase", marginTop: 2 }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* OCR method */}
      {result.ocr?.engine_used && result.ocr.engine_used !== "none" && (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <Badge label="ENGINE" value={result.ocr.engine_used.toUpperCase()} color="#00d4ff" />
          <Badge label="OCR CONF" value={`${Math.round((result.ocr.avg_confidence || 0) * 100)}%`} color="#00d4ff" />
          <Badge label="PAGES" value={result.ocr.pages_processed || 0} color="#00d4ff" />
        </div>
      )}

      {/* Entities */}
      {totalEntities > 0 && (
        <div style={{ background: "rgba(0,0,0,0.35)", border: "1px solid rgba(0,255,140,0.08)", borderRadius: 12, padding: "16px 18px" }}>
          <div style={{ fontSize: 11, fontFamily: "'Space Mono', monospace", color: "#4b5563", letterSpacing: "0.1em", marginBottom: 12 }}>EXTRACTED ENTITIES</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {Object.entries(entities).flatMap(([label, values]) =>
              (Array.isArray(values) ? values : [values]).map((v, i) => (
                <EntityChip key={`${label}-${i}`} label={label} value={String(v)} />
              ))
            )}
          </div>
        </div>
      )}

      {/* All type scores */}
      {result.document?.all_type_scores && (
        <div style={{ background: "rgba(0,0,0,0.35)", border: "1px solid rgba(0,255,140,0.06)", borderRadius: 12, padding: "16px 18px" }}>
          <div style={{ fontSize: 11, fontFamily: "'Space Mono', monospace", color: "#4b5563", letterSpacing: "0.1em", marginBottom: 12 }}>ALL CLASS SCORES</div>
          {Object.entries(result.document.all_type_scores).sort(([, a], [, b]) => b - a).map(([cls, score]) => {
            const meta = DOCTYPE_META[cls] || DOCTYPE_META.unknown;
            const pct = Math.round(score * 100);
            return (
              <div key={cls} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                <div style={{ width: 80, fontSize: 10, color: "#4b5563", fontFamily: "'Space Mono', monospace", textAlign: "right", flexShrink: 0 }}>
                  {cls.replace("_", " ").slice(0, 12)}
                </div>
                <div style={{ flex: 1, height: 4, background: "rgba(0,0,0,0.5)", borderRadius: 2, overflow: "hidden" }}>
                  <div style={{ height: "100%", width: `${pct}%`, background: meta.color, borderRadius: 2, transition: "width 1s ease" }} />
                </div>
                <div style={{ width: 32, fontSize: 10, fontFamily: "'Space Mono', monospace", color: meta.color, textAlign: "right" }}>{pct}%</div>
              </div>
            );
          })}
        </div>
      )}

    </div>
  );
}
