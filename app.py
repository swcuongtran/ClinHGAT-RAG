from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent  # app.py is at project root
HGAT_MODEL     = ROOT / "models" / "clinical_hgat.pt"
TEMPORAL_MODEL = ROOT / "models" / "temporal_forecaster.pt"

# ─── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical CDSS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.2rem;
}
.card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid rgba(226, 232, 240, 0.8);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 20px -3px rgba(0, 0, 0, 0.06), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
    border-color: rgba(59, 130, 246, 0.4);
}
.card h4 {
    margin: 0 0 0.4rem;
    color: #1e3a8a;
    font-size: 0.95rem;
    font-weight: 600;
}
.card p {
    margin: 0;
    color: #4b5563;
    font-size: 0.88rem;
}
.sec {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e293b;
    border-left: 5px solid #2563eb;
    padding-left: 0.75rem;
    margin: 1.5rem 0 0.8rem;
    letter-spacing: -0.015em;
}
.badge-hgat {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
    color: #ffffff;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 600;
    box-shadow: 0 2px 4px 0 rgba(79, 70, 229, 0.2);
}
.badge-rag {
    background: linear-gradient(135deg, #0d9488 0%, #10b981 100%);
    color: #ffffff;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 600;
    box-shadow: 0 2px 4px 0 rgba(13, 148, 136, 0.2);
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid rgba(226, 232, 240, 0.8);
    border-radius: 12px;
    padding: 0.8rem 1.1rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.1), 0 4px 6px -2px rgba(59, 130, 246, 0.05);
    border-color: rgba(59, 130, 246, 0.4);
}
[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] h2 {
    color: #1e293b;
    font-weight: 700;
}
[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.02), 0 4px 6px -2px rgba(0, 0, 0, 0.01);
}
div.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: white !important;
    border: none;
    padding: 0.6rem 1.8rem;
    border-radius: 10px;
    font-weight: 600;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    transform: translateY(-1px);
    box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
}
div.stButton > button:active {
    transform: translateY(1px);
}
</style>
""", unsafe_allow_html=True)

# ─── Cached loaders ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _db():
    from clinical_cdss.core.database import Neo4jConnection
    return Neo4jConnection()

@st.cache_data(ttl=60)
def _stats():
    from clinical_cdss.core.database import Neo4jConnection
    db = Neo4jConnection()
    try:
        q = "MATCH (n:{}) RETURN count(n) AS c"
        return {
            "Patients":  db.execute_query("MATCH (p:Patient) RETURN count(p) AS c")[0]["c"],
            "Records":   db.execute_query("MATCH (d:DailyRecord) RETURN count(d) AS c")[0]["c"],
            "Evidence":  db.execute_query("MATCH (e:EvidenceCase) RETURN count(e) AS c")[0]["c"],
            "Rules":     db.execute_query("MATCH (r:Diagnostic_Rule) RETURN count(r) AS c")[0]["c"],
            "Guideline": db.execute_query("MATCH (g:GuidelineChunk) RETURN count(g) AS c")[0]["c"],
        }
    finally:
        db.close()

@st.cache_data(ttl=60)
def _patients():
    from clinical_cdss.core.database import Neo4jConnection
    db = Neo4jConnection()
    try:
        rows = db.execute_query("MATCH (p:Patient) RETURN p.id AS id ORDER BY p.id LIMIT 2000")
        return [r["id"] for r in rows]
    finally:
        db.close()

def _summary(pid):
    from clinical_cdss.core.database import Neo4jConnection
    db = Neo4jConnection()
    try:
        rows = db.execute_query("""
            MATCH (p:Patient {id:$pid})
            OPTIONAL MATCH (p)-[:HAS_EVIDENCE]->(ec:EvidenceCase)
            RETURN p.age AS age, p.gender AS gender,
                   p.admission_day AS admission_day,
                   p.binary_label AS binary_label,
                   ec.plt_nadir AS plt_nadir, ec.hct_peak AS hct_peak,
                   ec.hflc_peak AS hflc_peak, ec.critical_day AS critical_day
            LIMIT 1
        """, {"pid": pid})
    finally:
        db.close()
    return dict(rows[0]) if rows else {}

def _records_df(pid):
    import pandas as pd
    from clinical_cdss.core.database import Neo4jConnection
    db = Neo4jConnection()
    try:
        rows = db.execute_query("""
            MATCH (p:Patient {id:$pid})-[:HAS_RECORD]->(d:DailyRecord)
            RETURN d.disease_day AS disease_day, d.day AS day,
                   d.plt AS plt, d.hct AS hct, d.wbc AS wbc, d.hflc AS hflc
            ORDER BY d.disease_day
        """, {"pid": pid})
    finally:
        db.close()
    return pd.DataFrame([dict(r) for r in rows])

def _matches_df(pid):
    import pandas as pd
    from clinical_cdss.core.database import Neo4jConnection
    db = Neo4jConnection()
    try:
        rows = db.execute_query("""
            MATCH (p:Patient {id:$pid})-[:HAS_EVIDENCE]->(ec:EvidenceCase)
                  -[m:MATCHES]->(r:Diagnostic_Rule)
            OPTIONAL MATCH (r)-[:DETERMINES]->(s:Severity)
            RETURN r.name AS rule, s.name AS severity,
                   m.coverage_score AS score,
                   m.sym_match AS sym_m, m.sym_total AS sym_t,
                   m.concept_match AS con_m, m.concept_total AS con_t
            ORDER BY score DESC LIMIT 10
        """, {"pid": pid})
    finally:
        db.close()
    return pd.DataFrame([dict(r) for r in rows])

def _similar(pid):
    from clinical_cdss.core.database import Neo4jConnection
    db = Neo4jConnection()
    try:
        return db.execute_query("""
            MATCH (p:Patient {id:$pid})-[sim:SIMILAR_TO]-(o:Patient)
            RETURN o.id AS id, sim.shared_symptoms AS shared, o.binary_label AS label
            ORDER BY shared DESC LIMIT 5
        """, {"pid": pid})
    finally:
        db.close()

@st.cache_resource(show_spinner=False)
def _rag():
    from clinical_cdss.rag.engine import MedicalGraphRAG
    return MedicalGraphRAG()

@st.cache_resource(show_spinner=False)
def _cdss():
    from clinical_cdss.gnn.predict import ClinicalCDSS
    return ClinicalCDSS(str(HGAT_MODEL))

@st.cache_resource(show_spinner=False)
def _temporal():
    if not TEMPORAL_MODEL.exists():
        return None
    from clinical_cdss.temporal.predict import TemporalProgressionPredictor
    return TemporalProgressionPredictor(str(TEMPORAL_MODEL))

# ─── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Clinical CDSS")
    st.caption("Dengue Hemorrhagic Fever")
    st.divider()

    # Stats
    try:
        s = _stats()
        cols = st.columns(2)
        for i, (k, v) in enumerate(s.items()):
            cols[i % 2].metric(k, v)
    except Exception as e:
        st.warning(f"Neo4j: {e}")

    st.divider()

    # Models
    st.markdown("**Models**")
    st.write(f"{'✅' if HGAT_MODEL.exists() else '❌'} HGAT")
    st.write(f"{'✅' if TEMPORAL_MODEL.exists() else '❌'} Temporal")
    if not HGAT_MODEL.exists():
        st.caption("Train: `python -m clinical_cdss.gnn.train`")

    st.divider()

    # Patient
    try:
        patients = _patients()
    except Exception as e:
        st.error(f"Cannot load patients: {e}")
        st.stop()

    if not patients:
        st.warning("No patients. Run ETL first.")
        st.stop()

    pid = st.selectbox("Patient", patients, label_visibility="collapsed")
    if st.session_state.get("_pid") != pid:
        st.session_state.update({"_pid": pid, "cdss": None, "fc": None})

    page = st.radio("", ["Overview", "Daily Update", "Diagnosis", "Graph XAI", "Forecasting"],
                    label_visibility="collapsed")

# ─── Patient header ────────────────────────────────────────────────────────
try:
    info = _summary(pid)
except Exception as e:
    st.error(f"Cannot load patient: {e}")
    st.stop()

gender = "Nam" if info.get("gender") == 1 else "Nữ"
lbl    = info.get("binary_label")
icon   = "🔴" if lbl == 1 else ("🟢" if lbl == 0 else "⚪")
lbl_t  = {0: "Non-shock", 1: "Shock DHF"}.get(lbl, "Unknown")

st.markdown(f"### {icon} {pid} &nbsp;<small style='color:#6b7280;font-size:.8rem;'>{lbl_t}</small>",
            unsafe_allow_html=True)

c = st.columns(6)
c[0].metric("Tuổi",       info.get("age", "–"))
c[1].metric("Giới",       gender)
c[2].metric("PLT nadir",  info.get("plt_nadir", "–"))
c[3].metric("HCT peak",   info.get("hct_peak",  "–"))
c[4].metric("HFLC peak",  info.get("hflc_peak", "–"))
c[5].metric("Critical day", info.get("critical_day", "–"))
st.divider()

# ══════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "Overview":
    import plotly.graph_objects as go

    left, right = st.columns([1.5, 1])

    with left:
        st.markdown('<div class="sec">Diễn tiến xét nghiệm</div>', unsafe_allow_html=True)
        try:
            df = _records_df(pid)
        except Exception as e:
            st.error(str(e)); df = None

        if df is not None and not df.empty:
            fig = go.Figure()
            if df["plt"].notna().any():
                fig.add_trace(go.Scatter(x=df["disease_day"], y=df["plt"],
                    name="PLT (k/µL)", mode="lines+markers",
                    line=dict(color="#ef4444", width=2.5), marker=dict(size=6)))
                fig.add_hline(y=100, line_dash="dash", line_color="#f59e0b",
                              annotation_text="PLT=100", annotation_position="top right")
            if df["hct"].notna().any():
                fig.add_trace(go.Scatter(x=df["disease_day"], y=df["hct"],
                    name="HCT (%)", mode="lines+markers", yaxis="y2",
                    line=dict(color="#3b82f6", width=2.5), marker=dict(size=6)))
            if df["hflc"].notna().any():
                fig.add_trace(go.Scatter(x=df["disease_day"], y=df["hflc"],
                    name="HFLC (%)", mode="lines+markers", yaxis="y2",
                    line=dict(color="#8b5cf6", width=2, dash="dot"), marker=dict(size=5)))
            fig.update_layout(
                height=330, margin=dict(l=8, r=8, t=10, b=8),
                paper_bgcolor="#fff", plot_bgcolor="#f9fafb",
                xaxis=dict(title="Disease day", gridcolor="#e5e7eb"),
                yaxis=dict(title="PLT", gridcolor="#e5e7eb"),
                yaxis2=dict(title="HCT/HFLC", overlaying="y", side="right"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("Chưa có dữ liệu xét nghiệm hàng ngày.")

    with right:
        st.markdown('<div class="sec">EvidenceCase Matches</div>', unsafe_allow_html=True)
        try:
            mdf = _matches_df(pid)
        except Exception as e:
            mdf = None; st.warning(str(e))

        if mdf is not None and not mdf.empty:
            import plotly.express as px
            fig2 = px.bar(mdf.head(8), x="score", y="rule", orientation="h",
                          color="score",
                          color_continuous_scale=["#dbeafe","#3b82f6","#1d4ed8"],
                          hover_data=["severity","sym_m","sym_t"])
            fig2.update_layout(
                height=280, margin=dict(l=8,r=8,t=8,b=8),
                paper_bgcolor="#fff", plot_bgcolor="#f9fafb",
                coloraxis_showscale=False,
                xaxis=dict(title="Score", gridcolor="#e5e7eb"),
                yaxis=dict(automargin=True, title=""),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="sec">Bệnh nhân tương tự</div>', unsafe_allow_html=True)
        try:
            sims = _similar(pid)
            if sims:
                for r in sims:
                    ico = "🔴" if r["label"] == 1 else "🟢"
                    st.markdown(
                        f'<div class="card"><h4>{ico} {r["id"]}</h4>'
                        f'<p>Triệu chứng chung: <b>{r["shared"]}</b></p></div>',
                        unsafe_allow_html=True)
            else:
                st.info("Không tìm thấy bệnh nhân tương tự.")
        except Exception as e:
            st.warning(str(e))

# ══════════════════════════════════════════════════════════════════════════
# DAILY UPDATE
# ══════════════════════════════════════════════════════════════════════════
elif page == "Daily Update":
    st.markdown('<div class="sec">Cập nhật dữ liệu ngày</div>', unsafe_allow_html=True)

    with st.form("daily"):
        r1 = st.columns(3)
        hday = r1[0].number_input("Hospital day", 1, 30, 1)
        dday = r1[1].number_input("Disease day",  1, 30, 1)
        auto = r1[2].checkbox("Auto disease day", value=False)
        r2 = st.columns(4)
        wbc  = r2[0].number_input("WBC",     0.0, step=0.1)
        plt_ = r2[1].number_input("PLT",     0.0, step=1.0)
        hct  = r2[2].number_input("HCT",     0.0, 100.0, step=0.1)
        hflc = r2[3].number_input("HFLC",    0.0, step=0.01)
        ok = st.form_submit_button("💾 Lưu & cập nhật graph", use_container_width=True)

    if ok:
        try:
            from clinical_cdss.clinical.daily_update import upsert_daily_record
            res = upsert_daily_record(
                patient_id=pid,
                hospital_day=int(hday),
                disease_day=None if auto else int(dday),
                wbc=wbc or None, plt=plt_ or None,
                hct=hct or None, hflc=hflc or None,
            )
            _stats.clear()
            if "error" in res:
                st.error(res["error"])
            else:
                st.success("Đã lưu và refresh EvidenceCase.")
                st.json(res)
        except Exception as e:
            st.error(str(e))

# ══════════════════════════════════════════════════════════════════════════
# DIAGNOSIS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Diagnosis":
    st.markdown('<div class="sec">Chẩn đoán & Báo cáo Lâm sàng</div>', unsafe_allow_html=True)

    # Show retrain warning if model has dimension mismatch
    if HGAT_MODEL.exists():
        try:
            cdss_obj = _cdss()
            if getattr(cdss_obj, "_dim_mismatch", None):
                st.warning(
                    f"⚠️ **HGAT model cần retrain** — Dimension mismatch: `{cdss_obj._dim_mismatch}`\n\n"
                    "Graph đã thay đổi kể từ lần train cuối. Chạy lại:\n"
                    "```\npython -m clinical_cdss.gnn.train --epochs 120 --patience 20\n```\n"
                    "Tạm thời dùng **RAG Coverage Score** thay thế."
                )
        except Exception:
            pass

    mode    = st.radio("Phương pháp", ["HGAT + RAG", "RAG Coverage Score"], horizontal=True)
    use_llm = st.checkbox("Dùng LLM local (Qwen)", value=False,
                          help="Bật khi cần báo cáo dạng văn xuôi. Cần GPU/RAM lớn.")

    if st.button("🔍 Phân tích", use_container_width=True):
        st.session_state["cdss"] = None
        with st.spinner("Đang phân tích..."):
            try:
                if mode.startswith("HGAT") and HGAT_MODEL.exists():
                    result = _cdss().diagnose(pid, use_llm=use_llm)
                else:
                    engine = _rag()
                    ctx    = engine.retrieve_context(pid)
                    if "error" in ctx:
                        result = ctx
                    else:
                        gl     = engine.retrieve_guideline_chunk(
                                     ctx.get("rule_name",""), ctx.get("concepts",[]))
                        report = engine.generate_response(ctx, guideline_chunk=gl,
                                                          use_llm=use_llm)
                        result = {
                            "method": "Coverage Score",
                            "confidence": ctx.get("coverage_score", 0.0),
                            "diagnosis": None,
                            "report": report,
                            "subgraph": {k:v for k,v in ctx.items() if k != "context_str"},
                        }
                st.session_state["cdss"] = result
            except Exception as e:
                st.session_state["cdss"] = {"error": str(e)}

    res = st.session_state.get("cdss")
    if res:
        if "error" in res:
            st.error(res["error"])
        else:
            method = res.get("method", "")
            conf   = res.get("confidence", 0.0)
            diag   = res.get("diagnosis")

            badge = "badge-hgat" if "HGAT" in method else "badge-rag"
            st.markdown(f'<span class="{badge}">{method}</span>', unsafe_allow_html=True)

            m1,m2,m3 = st.columns(3)
            m1.metric("Confidence",  f"{conf:.1%}")
            m2.metric("Diagnosis",   {0:"Non-shock",1:"Shock DHF"}.get(diag,"—"))
            m3.metric("Method",      method.split("(")[0].strip())

            st.markdown('<div class="sec">Báo cáo Lâm sàng</div>', unsafe_allow_html=True)
            report_txt = res.get("report","")
            st.markdown(report_txt)
            st.download_button("📥 Tải báo cáo", data=report_txt,
                               file_name=f"report_{pid}.txt", mime="text/plain")

            with st.expander("Subgraph / Raw"):
                st.json(res.get("subgraph",{}))

# ══════════════════════════════════════════════════════════════════════════
# GRAPH XAI
# ══════════════════════════════════════════════════════════════════════════
elif page == "Graph XAI":
    import plotly.express as px

    st.markdown('<div class="sec">EvidenceCase → Diagnostic Rule Coverage</div>',
                unsafe_allow_html=True)
    try:
        mdf = _matches_df(pid)
    except Exception as e:
        st.error(str(e)); mdf = None

    if mdf is not None and not mdf.empty:
        st.dataframe(mdf, hide_index=True, use_container_width=True)
        fig = px.bar(mdf.head(8), x="score", y="rule", orientation="h", color="score",
                     color_continuous_scale=["#dbeafe","#3b82f6","#1d4ed8"],
                     hover_data=["severity","sym_m","sym_t","con_m","con_t"])
        fig.update_layout(height=360, margin=dict(l=8,r=8,t=8,b=8),
                          paper_bgcolor="#fff", plot_bgcolor="#f9fafb",
                          coloraxis_showscale=False,
                          xaxis=dict(gridcolor="#e5e7eb"),
                          yaxis=dict(automargin=True, title=""))
        st.plotly_chart(fig, use_container_width=True)

        top = mdf.iloc[0]
        st.markdown(
            f'<div class="card"><h4>Best match: {top["rule"]}</h4>'
            f'<p>Severity: <b>{top["severity"]}</b> &nbsp;|&nbsp; '
            f'Score: <b>{top["score"]:.1%}</b> &nbsp;|&nbsp; '
            f'Sym: <b>{top["sym_m"]}/{top["sym_t"]}</b> &nbsp;|&nbsp; '
            f'Concept: <b>{top["con_m"]}/{top["con_t"]}</b></p></div>',
            unsafe_allow_html=True)
    else:
        st.info("Chưa có MATCHES edges. Chạy ETL để tạo EvidenceCase.")

    # HGAT attention nodes (nếu đã chạy diagnosis)
    res = st.session_state.get("cdss")
    if res and "subgraph" in res:
        nodes = res["subgraph"].get("attention_nodes", [])
        if nodes:
            import pandas as pd
            st.markdown('<div class="sec">HGAT Attention Nodes</div>', unsafe_allow_html=True)
            adf = pd.DataFrame(nodes, columns=["Node","Weight"])
            fig2 = px.bar(adf.sort_values("Weight"), x="Weight", y="Node",
                          orientation="h", color="Weight",
                          color_continuous_scale=["#eff6ff","#3b82f6","#1d4ed8"])
            fig2.update_layout(height=260, margin=dict(l=8,r=8,t=8,b=8),
                               paper_bgcolor="#fff", plot_bgcolor="#f9fafb",
                               coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# FORECASTING
# ══════════════════════════════════════════════════════════════════════════
elif page == "Forecasting":
    import plotly.express as px

    st.markdown('<div class="sec">Dự báo Tiến triển (Temporal Model)</div>',
                unsafe_allow_html=True)

    if not TEMPORAL_MODEL.exists():
        st.warning("Temporal model chưa train.\n\n"
                   "`python -m clinical_cdss.temporal.train --epochs 120`")
    else:
        if st.button("▶ Chạy dự báo", use_container_width=True):
            st.session_state["fc"] = None
            with st.spinner("Đang dự báo..."):
                try:
                    predictor = _temporal()
                    st.session_state["fc"] = predictor.predict(pid)
                except Exception as e:
                    st.session_state["fc"] = {"error": str(e)}

        fc = st.session_state.get("fc")
        if fc:
            if "error" in fc:
                st.error(fc["error"])
            else:
                a,b,c = st.columns(3)
                a.metric("Xác suất Shock",     f"{fc['shock_probability']:.1%}")
                b.metric("Xác suất Non-shock", f"{fc['non_shock_probability']:.1%}")
                c.metric("Nguy cơ tổng hợp",   f"{fc['forecast_risk']:.1%}")

                days = fc.get("day_attention",[])
                if days:
                    import pandas as pd
                    adf = pd.DataFrame(days)
                    fig = px.bar(adf, x="disease_day", y="attention", color="attention",
                                 color_continuous_scale=["#dbeafe","#3b82f6","#ef4444"],
                                 hover_data=["plt","hct","wbc","hflc"])
                    fig.update_layout(height=300, margin=dict(l=8,r=8,t=8,b=8),
                                      paper_bgcolor="#fff", plot_bgcolor="#f9fafb",
                                      coloraxis_showscale=False)
                    st.markdown('<div class="sec">Attention theo ngày bệnh</div>',
                                unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("Raw output"):
                    st.json(fc)
