import base64
from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st

from model import (
    KNN,
    KararAgaci,
    LojistikRegresyon,
    NaiveBayes,
    agaci_yaziya_cevir,
    hiperparametre_ara,
    hiperparametre_ara_knn,
    hiperparametre_ara_lojistik,
    hiperparametre_ara_naive_bayes,
    metrikleri_hesapla,
)


# =========================================================
# SAYFA AYARLARI
# =========================================================
st.set_page_config(
    page_title="De Humani Cordis — Fabrica ML Atlası",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

HEDEF_SUTUN = "heart_disease"
VERI_DOSYASI = Path(__file__).parent / "heart_disease_prediction_2026.csv"
HEART_ATLAS_DOSYASI = Path(__file__).parent / "heart_atlas.png"


@st.cache_data
def heart_atlas_data_uri():
    if not HEART_ATLAS_DOSYASI.exists():
        return ""
    veri = HEART_ATLAS_DOSYASI.read_bytes()
    kodlu = base64.b64encode(veri).decode("ascii")
    return f"data:image/png;base64,{kodlu}"


# =========================================================
# FABRICA · SCROLLTIDE ANATOMI ATLASI TASARIM SISTEMI
# =========================================================
def inject_custom_css():
    css = """<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg: #F4EFE4;
  --paper: #FAF7F0;
  --paper-deep: #EFE8D8;
  --ink: #1E1B18;
  --ink-muted: #5C5549;
  --ink-faint: #8C8273;
  --rule: #D8CFC0;
  --rule-strong: #B9AD99;
  --crimson: #9B2226;
  --crimson-soft: rgba(155, 34, 38, 0.10);
  --emerald: #2D6A4F;
  --emerald-bg: #ECF5F0;
  --emerald-border: #B7DFC9;
  --rose-bg: #FDF2F2;
  --rose-border: #F5C2C2;
}

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  color: var(--ink);
}

.stApp {
  background-color: var(--bg);
  background-image: radial-gradient(rgba(140, 130, 115, 0.14) 1px, transparent 1px);
  background-size: 22px 22px;
}

header[data-testid="stHeader"] {
  background: transparent !important;
}

#MainMenu, footer {
  visibility: hidden;
}

.block-container {
  padding-top: 1rem !important;
  padding-bottom: 2.5rem !important;
  max-width: 1260px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background-color: var(--paper) !important;
  border-right: 1px solid var(--rule);
}

section[data-testid="stSidebar"] .block-container {
  padding-top: 1.2rem !important;
}

.sidebar-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0.25rem 0.1rem 0.85rem 0.1rem;
  border-bottom: 1px solid var(--rule);
  margin-bottom: 0.85rem;
}

.sidebar-brand-title {
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-size: 1.35rem;
  font-weight: 600;
  color: var(--ink);
  line-height: 1.1;
}

.sidebar-brand-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-faint);
}

.sidebar-section-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.66rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-faint);
  margin: 0.5rem 0 0.4rem 0;
}

div.stButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button {
  background: var(--crimson) !important;
  border: 1px solid var(--crimson) !important;
  border-radius: 8px !important;
  color: #FAF7F0 !important;
  font-weight: 500 !important;
  padding: 0.55rem 0.9rem !important;
  transition: all 0.18s ease !important;
}

div.stButton > button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
  background: #7F1D1D !important;
  transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(155, 34, 38, 0.22) !important;
}

div.stButton > button[kind="secondary"] {
  background: var(--bg) !important;
  border: 1px solid var(--rule-strong) !important;
  border-radius: 8px !important;
  color: var(--ink) !important;
}

.mono-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.67rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-faint);
}

.mono-pill {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  font-weight: 500;
  padding: 4px 10px;
  border-radius: 99px;
  background: var(--bg);
  border: 1px solid var(--rule);
  color: var(--ink-muted);
}

/* Top Atlas Navigation Bar */
.fabrica-topbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  background: var(--paper);
  border: 1px solid var(--rule);
  border-radius: 12px;
  padding: 10px 16px;
  margin-bottom: 18px;
  box-shadow: 0 4px 16px rgba(30, 27, 24, 0.04);
}

.fabrica-topbar-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.74rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  color: var(--ink);
}

.fabrica-nav-links {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
}

.fabrica-nav-pill {
  display: inline-block;
  text-decoration: none !important;
  font-size: 0.8rem;
  font-weight: 500;
  color: var(--ink-muted) !important;
  background: transparent;
  border: 1px solid transparent;
  padding: 5px 12px;
  border-radius: 99px;
  transition: all 0.16s ease;
}

.fabrica-nav-pill:hover {
  background: var(--bg);
  border-color: var(--rule);
  color: var(--ink) !important;
}

.fabrica-nav-pill.active {
  background: var(--ink);
  color: var(--paper) !important;
  border-color: var(--ink);
}

/* Editorial Atlas Title */
.atlas-header {
  text-align: center;
  max-width: 780px;
  margin: 0 auto 18px;
}

.atlas-title {
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-size: clamp(2.1rem, 4vw, 3.1rem);
  font-weight: 600;
  line-height: 1.08;
  letter-spacing: -0.02em;
  color: var(--ink);
  margin: 6px 0 8px;
}

.atlas-title em {
  font-style: italic;
  color: var(--crimson);
  font-weight: 500;
}

.atlas-sub {
  font-size: 0.94rem;
  color: var(--ink-muted);
  line-height: 1.55;
  margin: 0;
}

/* Interactive Anatomical Heart Plate (Fabrica) */
.fabrica-plate {
  position: relative;
  width: 100%;
  height: 540px;
  background: var(--paper);
  border: 1px solid var(--rule-strong);
  outline: 4px solid var(--paper);
  outline-offset: -8px;
  border-radius: 16px;
  overflow: hidden;
  margin-bottom: 22px;
  box-shadow: 0 20px 45px -20px rgba(30, 27, 24, 0.16);
}

.plate-crosshair {
  position: absolute;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  color: var(--ink-faint);
  z-index: 3;
  pointer-events: none;
}

.ch-tl { top: 12px; left: 16px; }
.ch-tr { top: 12px; right: 16px; }
.ch-bl { bottom: 12px; left: 16px; }
.ch-br { bottom: 12px; right: 16px; }

.plate-ring-1, .plate-ring-2 {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  border-radius: 50%;
  border: 1px dashed rgba(140, 130, 115, 0.28);
  pointer-events: none;
  z-index: 1;
}

.plate-ring-1 { width: 320px; height: 320px; }
.plate-ring-2 { width: 430px; height: 430px; }

.fabrica-heart-wrap {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 330px;
  height: 330px;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2;
  pointer-events: none;
}

.fabrica-heart-img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  mix-blend-mode: multiply;
  filter: contrast(1.04);
}

.fabrica-svg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  z-index: 3;
  pointer-events: none;
}

@keyframes pinPulse {
  0%, 100% { r: 6; opacity: 0.95; }
  50% { r: 9; opacity: 0.55; }
}

.pin-halo {
  animation: pinPulse 2.4s ease-in-out infinite;
}

/* Clickable Callout Nodes */
.callout-node {
  position: absolute;
  z-index: 4;
  width: 245px;
  background: rgba(250, 247, 240, 0.94);
  -webkit-backdrop-filter: blur(8px);
  backdrop-filter: blur(8px);
  border: 1px solid var(--rule-strong);
  border-radius: 12px;
  padding: 13px 15px;
  text-decoration: none !important;
  color: var(--ink) !important;
  box-shadow: 0 8px 22px -10px rgba(30, 27, 24, 0.16);
  transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

.callout-node:hover {
  transform: translateY(-3px);
  border-color: var(--crimson);
  box-shadow: 0 14px 28px -10px rgba(155, 34, 38, 0.22);
}

.node-tl { top: 44px; left: 34px; }
.node-tr { top: 44px; right: 34px; }
.node-bl { bottom: 44px; left: 34px; }
.node-br { bottom: 44px; right: 34px; border-color: var(--crimson); background: rgba(253, 246, 244, 0.96); }

.callout-fig {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.66rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  color: var(--crimson);
  margin-bottom: 3px;
}

.callout-title {
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-size: 1.3rem;
  font-weight: 600;
  color: var(--ink);
  line-height: 1.15;
  margin-bottom: 4px;
}

.callout-desc {
  font-size: 0.77rem;
  color: var(--ink-muted);
  line-height: 1.4;
  margin-bottom: 6px;
}

.callout-cta {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--crimson);
}

/* Mobile fallback for Heart Plate */
@media (max-width: 900px) {
  .fabrica-plate {
    height: auto;
    padding: 20px 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
  }
  .fabrica-svg, .plate-ring-1, .plate-ring-2 { display: none; }
  .fabrica-heart-wrap {
    position: relative;
    top: auto;
    left: auto;
    transform: none;
    width: 220px;
    height: 220px;
  }
  .callout-node {
    position: relative;
    top: auto;
    left: auto;
    right: auto;
    bottom: auto;
    width: 100%;
  }
}

/* Archival Stat Strip & Cards */
.folio-strip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  background: var(--paper);
  border: 1px solid var(--rule);
  border-radius: 12px;
  margin-bottom: 20px;
  overflow: hidden;
}

.folio-cell {
  padding: 14px 16px;
  border-right: 1px solid var(--rule);
  text-align: center;
}

.folio-cell:last-child { border-right: none; }

.folio-val {
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-size: 1.7rem;
  font-weight: 600;
  color: var(--ink);
  margin-top: 2px;
}

.bento-grid {
  display: grid;
  gap: 14px;
  margin-bottom: 20px;
}

.bento-cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.bento-cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.bento-cols-5 { grid-template-columns: repeat(5, minmax(0, 1fr)); }

.folio-card {
  background: var(--paper);
  border: 1px solid var(--rule);
  border-radius: 12px;
  padding: 18px 20px;
}

.kpi-value {
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-size: 1.9rem;
  font-weight: 600;
  color: var(--ink);
  line-height: 1.1;
  margin-top: 5px;
}

.kpi-sub {
  font-size: 0.78rem;
  color: var(--ink-muted);
  margin-top: 4px;
}

.delta-badge {
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 99px;
  margin-top: 7px;
}

.delta-pos {
  color: var(--emerald);
  background: var(--emerald-bg);
  border: 1px solid var(--emerald-border);
}

.delta-neg {
  color: var(--crimson);
  background: var(--rose-bg);
  border: 1px solid var(--rose-border);
}

.back-link {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  text-decoration: none !important;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.76rem;
  font-weight: 600;
  color: var(--crimson) !important;
  margin-bottom: 10px;
}

.back-link:hover {
  text-decoration: underline !important;
}

.result-header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--rule);
  margin-bottom: 18px;
}

.result-title {
  margin: 0;
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-size: 2rem;
  font-weight: 600;
  color: var(--ink);
}

div[data-testid="stExpander"] {
  background: var(--paper);
  border: 1px solid var(--rule);
  border-radius: 12px;
}

.fabrica-footer {
  margin-top: 2rem;
  padding-top: 1.2rem;
  border-top: 1px solid var(--rule);
  display: grid;
  grid-template-columns: 1.3fr 1fr 1.2fr;
  gap: 1.4rem;
  font-size: 0.8rem;
  color: var(--ink-muted);
}

.footer-disclaimer {
  background: #FAF0DC;
  border: 1px solid #E6D3A7;
  color: #78551A;
  border-radius: 8px;
  padding: 0.6rem 0.8rem;
  font-size: 0.76rem;
  line-height: 1.45;
}

@media (max-width: 900px) {
  .folio-strip { grid-template-columns: 1fr 1fr; }
  .folio-cell:nth-child(2) { border-right: none; }
  .folio-cell:nth-child(1), .folio-cell:nth-child(2) { border-bottom: 1px solid var(--rule); }
  .bento-cols-3, .bento-cols-4, .bento-cols-5, .fabrica-footer { grid-template-columns: 1fr; }
}
</style>"""
    st.markdown(css, unsafe_allow_html=True)


def render_kpi_card(label, value, sub="", delta=None, delta_positive=True):
    delta_html = ""
    if delta is not None:
        cls = "delta-pos" if delta_positive else "delta-neg"
        delta_html = f'<div class="delta-badge {cls}">{delta}</div>'
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="folio-card">'
        f'<div class="mono-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{delta_html}'
        f'{sub_html}'
        f'</div>'
    )


# =========================================================
# VERI SETINI OKU, HAZIRLA VE OPTIMIZE MODELLERI ONBELLEKLE
# =========================================================
@st.cache_data
def veri_yukle():
    df = pd.read_csv(VERI_DOSYASI)

    if HEDEF_SUTUN not in df.columns:
        raise ValueError(
            f"Veri setinde '{HEDEF_SUTUN}' isimli hedef sutun bulunamadi."
        )

    df = df.dropna().copy()
    df[HEDEF_SUTUN] = df[HEDEF_SUTUN].astype(int)

    return df


def veri_hazirla(df, egitim_yuzdesi, dogrulama_gerekli):
    """Ayni rastgele tohumla test sizintisi olmadan egitim/dogrulama/test bolmesi olusturur."""
    X_ham = df.drop(columns=[HEDEF_SUTUN])
    y = df[HEDEF_SUTUN].to_numpy(dtype=int)

    rng = np.random.default_rng(42)
    indeksler = rng.permutation(len(df))
    havuz_sayisi = int(len(df) * egitim_yuzdesi / 100)
    havuz_indeksleri = indeksler[:havuz_sayisi]
    test_indeksleri = indeksler[havuz_sayisi:]

    if dogrulama_gerekli:
        egitim_sayisi = int(len(havuz_indeksleri) * 0.8)
        egitim_indeksleri = havuz_indeksleri[:egitim_sayisi]
        dogrulama_indeksleri = havuz_indeksleri[egitim_sayisi:]
    else:
        egitim_indeksleri = havuz_indeksleri
        dogrulama_indeksleri = np.array([], dtype=int)

    X_egitim_ham = X_ham.iloc[egitim_indeksleri].copy()
    X_test_ham = X_ham.iloc[test_indeksleri].copy()
    y_egitim = y[egitim_indeksleri]
    y_test = y[test_indeksleri]

    X_egitim_df = pd.get_dummies(X_egitim_ham, dtype=np.uint8)
    X_test_df = pd.get_dummies(X_test_ham, dtype=np.uint8).reindex(
        columns=X_egitim_df.columns, fill_value=0
    )

    ozellik_isimleri = X_egitim_df.columns.tolist()
    X_egitim = X_egitim_df.to_numpy(dtype=float)
    X_test = X_test_df.to_numpy(dtype=float)

    if dogrulama_gerekli:
        X_dogrulama_ham = X_ham.iloc[dogrulama_indeksleri].copy()
        y_dogrulama = y[dogrulama_indeksleri]
        X_dogrulama_df = pd.get_dummies(
            X_dogrulama_ham, dtype=np.uint8
        ).reindex(columns=X_egitim_df.columns, fill_value=0)
        X_dogrulama = X_dogrulama_df.to_numpy(dtype=float)
    else:
        X_dogrulama = None
        y_dogrulama = None

    return (
        X_egitim,
        y_egitim,
        X_dogrulama,
        y_dogrulama,
        X_test,
        y_test,
        ozellik_isimleri,
    )


def test_seti_ham_getir(df, egitim_yuzdesi=80):
    """Ayni rastgele tohumla ayrilan test setindeki gercek hasta satirlarini dondurur."""
    rng = np.random.default_rng(42)
    indeksler = rng.permutation(len(df))
    havuz_sayisi = int(len(df) * egitim_yuzdesi / 100)
    test_indeksleri = indeksler[havuz_sayisi:]
    return df.iloc[test_indeksleri].reset_index(drop=True)


def tek_kisi_vektorle(kisi_sozlugu, ozellik_isimleri):
    """Tek bir kisinin 25 ozelligini modelin bekledigi sayisal matrise cevirir."""
    tek_df = pd.DataFrame([kisi_sozlugu])
    encoded = pd.get_dummies(tek_df, dtype=np.uint8).reindex(
        columns=ozellik_isimleri, fill_value=0
    )
    return encoded.to_numpy(dtype=float)


@st.cache_resource
def optimize_model_calistir(algoritma_kod, egitim_yuzdesi=80):
    """Kalp haritasindaki noktalara tiklandiginda hizli sonuc gostermek icin optimize modeli calistirir."""
    df_lokal = veri_yukle()
    (
        X_egitim,
        y_egitim,
        X_dogrulama,
        y_dogrulama,
        X_test,
        y_test,
        ozellik_isimleri,
    ) = veri_hazirla(df_lokal, egitim_yuzdesi, dogrulama_gerekli=True)

    t0 = time.time()
    ekstra = {}
    if algoritma_kod == "karar_agaci":
        model, ayar, sonuclar_df = hiperparametre_ara(
            X_egitim,
            y_egitim,
            X_dogrulama,
            y_dogrulama,
            ozellik_isimleri,
            hedef_metrik="accuracy",
            budama_uygula=True,
        )
        ekstra["agac_metni"] = "\n".join(
            agaci_yaziya_cevir(model.agac, model.ozellik_isimleri)
        )
        ekstra["budanan"] = model.budanan_dugum_sayisi
    elif algoritma_kod == "lojistik":
        model, ayar, sonuclar_df = hiperparametre_ara_lojistik(
            X_egitim,
            y_egitim,
            X_dogrulama,
            y_dogrulama,
            ozellik_isimleri,
            hedef_metrik="accuracy",
        )
        ekstra["ozellik_df"] = model.ozellik_agirliklari_df()
        ekstra["kayip_gecmisi"] = model.kayip_gecmisi
        ekstra["tamamlanan_iterasyon"] = model.tamamlanan_iterasyon
    elif algoritma_kod == "naive_bayes":
        model, ayar, sonuclar_df = hiperparametre_ara_naive_bayes(
            X_egitim,
            y_egitim,
            X_dogrulama,
            y_dogrulama,
            ozellik_isimleri,
            hedef_metrik="accuracy",
        )
        ekstra["ozellik_df"] = model.ozellik_ayirt_edicilik_df()
    else:
        model, ayar, sonuclar_df = hiperparametre_ara_knn(
            X_egitim,
            y_egitim,
            X_dogrulama,
            y_dogrulama,
            ozellik_isimleri,
            hedef_metrik="accuracy",
        )
        ekstra["ozellik_df"] = model.ozellik_agirliklari_df()
        ekstra["referans_sayisi"] = len(model.X_ref_s)

    test_olasilik = model.predict_proba(X_test)
    tahmin = (test_olasilik >= 0.5).astype(int)
    metrikler = metrikleri_hesapla(
        y_test, tahmin, olasiliklar=test_olasilik
    )
    sure = time.time() - t0

    return {
        "model": model,
        "ozellik_isimleri": ozellik_isimleri,
        "ayar": ayar,
        "sonuclar_df": sonuclar_df,
        "metrikler": metrikler,
        "sure": sure,
        "n_train": len(y_egitim),
        "n_val": len(y_dogrulama),
        "n_test": len(y_test),
        "ekstra": ekstra,
    }


def goster_tekli_sonuc(algoritma_ad, algoritma_kod, sonuc, egitim_yuzdesi):
    metrikler = sonuc["metrikler"]
    ayar = sonuc["ayar"]
    ekstra = sonuc["ekstra"]

    st.markdown(
        '<a class="back-link" href="?view=home" target="_self">← Kalp Atlasına Dön (Ana Sayfa)</a>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="result-header">
<h2 class="result-title">{algoritma_ad} — Optimize Test Sonuçları</h2>
<span class="mono-pill">Eğitim: {sonuc['n_train']:,} | Doğrulama: {sonuc['n_val']:,} | Test: {sonuc['n_test']:,} • {sonuc['sure']:.2f} sn</span>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="bento-grid bento-cols-5">'
        + render_kpi_card("Accuracy", f"%{metrikler['accuracy'] * 100:.2f}")
        + render_kpi_card("Precision", f"%{metrikler['precision'] * 100:.2f}")
        + render_kpi_card("Recall", f"%{metrikler['recall'] * 100:.2f}")
        + render_kpi_card("F1 Skoru", f"%{metrikler['f1'] * 100:.2f}")
        + render_kpi_card("ROC-AUC", f"{metrikler['roc_auc']:.4f}")
        + "</div>",
        unsafe_allow_html=True,
    )

    col_cm, col_cfg = st.columns([1, 1.2])
    with col_cm:
        st.markdown("#### Karışıklık Matrisi")
        st.dataframe(
            pd.DataFrame(
                [[metrikler["tn"], metrikler["fp"]],
                 [metrikler["fn"], metrikler["tp"]]],
                index=["Gerçek: Sağlıklı (0)", "Gerçek: Kalp Hastası (1)"],
                columns=["Tahmin: Sağlıklı (0)", "Tahmin: Kalp Hastası (1)"],
            ),
            use_container_width=True,
        )
        st.caption(
            "0 = Sağlıklı, 1 = Kalp Hastası. Sol üst ve sağ alt kutular doğru tahminleri; "
            "diğer iki kutu ise hatalı tahminleri gösterir."
        )

    with col_cfg:
        st.markdown("#### En İyi Bulunan Yapılandırma")
        if algoritma_kod == "karar_agaci":
            ozet = (
                f"• Yöntem: <strong>32-Bin Histogram + {ayar['kriter'].upper()}</strong><br>"
                f"• Derinlik: <strong>{ayar['derinlik']}</strong> | Min Bölünme: <strong>{ayar['min_bolunme']}</strong> | Min Yaprak: <strong>{ayar['min_yaprak']}</strong><br>"
                f"• Budanan Düğüm: <strong>{ekstra['budanan']}</strong> | Eğitim/Test: <strong>%{egitim_yuzdesi} / %{100 - egitim_yuzdesi}</strong>"
            )
        elif algoritma_kod == "lojistik":
            ozet = (
                f"• Optimizasyon: <strong>{ayar['optimizasyon'].upper()} + Z-Skor Ölçekleme</strong><br>"
                f"• Öğrenme Oranı: <strong>{ayar['ogrenme_orani']}</strong> | L2 (lambda): <strong>{ayar['l2_lambda']}</strong><br>"
                f"• Tamamlanan İterasyon: <strong>{ekstra['tamamlanan_iterasyon']} / 300</strong> | Eğitim/Test: <strong>%{egitim_yuzdesi} / %{100 - egitim_yuzdesi}</strong>"
            )
        elif algoritma_kod == "naive_bayes":
            ozet = (
                f"• Yöntem: <strong>{ayar['yontem'].upper()} Naive Bayes</strong><br>"
                f"• Laplace (alpha): <strong>{ayar['alpha']}</strong> | Varyans Yumuşatma: <strong>{ayar['var_smoothing']}</strong><br>"
                f"• Log-Loss: <strong>{metrikler['log_loss']:.4f}</strong> | Eğitim/Test: <strong>%{egitim_yuzdesi} / %{100 - egitim_yuzdesi}</strong>"
            )
        else:
            ozet = (
                f"• Mesafe: <strong>{ayar['mesafe'].upper()}</strong> | K: <strong>{ayar['k']}</strong> | Oylama: <strong>{ayar['agirliklandirma'].upper()}</strong><br>"
                f"• Özellik Ağırlıklandırma: <strong>{'Açık (Korelasyon)' if ayar['ozellik_agirlikli'] else 'Kapalı'}</strong> | Referans: <strong>{ekstra['referans_sayisi']:,}</strong><br>"
                f"• Eğitim/Test: <strong>%{egitim_yuzdesi} / %{100 - egitim_yuzdesi}</strong>"
            )
        st.markdown(
            f'<div class="folio-card"><div style="font-size:0.88rem; line-height:1.9; color:var(--ink-muted);">{ozet}</div></div>',
            unsafe_allow_html=True,
        )

    if sonuc["sonuclar_df"] is not None:
        with st.expander("Grid Search — En iyi 10 hiperparametre kombinasyonu"):
            st.dataframe(sonuc["sonuclar_df"].head(10), use_container_width=True)

    if algoritma_kod == "karar_agaci":
        with st.expander("Oluşturulan karar ağacı dallarını göster"):
            st.code(ekstra["agac_metni"], language="text")
    elif algoritma_kod == "lojistik":
        with st.expander("En etkili özellik ağırlıkları (İlk 15)"):
            st.dataframe(ekstra["ozellik_df"].head(15), use_container_width=True)
        if ekstra["kayip_gecmisi"]:
            with st.expander("Eğitim kayıp (Log-Loss) grafiği"):
                kayip_df = pd.DataFrame(
                    ekstra["kayip_gecmisi"],
                    columns=["İterasyon", "Log-Loss"],
                ).set_index("İterasyon")
                st.line_chart(kayip_df)
    elif algoritma_kod == "naive_bayes":
        with st.expander("En ayırt edici özellikler (İlk 15)"):
            st.dataframe(ekstra["ozellik_df"].head(15), use_container_width=True)
    else:
        with st.expander("En etkili özellik ağırlıkları (İlk 15)"):
            st.dataframe(ekstra["ozellik_df"].head(15), use_container_width=True)


# =========================================================
# UYGULAMA BASLANGICI
# =========================================================
inject_custom_css()
heart_uri = heart_atlas_data_uri()

try:
    df = veri_yukle()
except FileNotFoundError:
    st.error(
        "CSV dosyası bulunamadı. app.py ile "
        "heart_disease_prediction_2026.csv aynı klasörde olmalı."
    )
    st.stop()
except Exception as hata:
    st.error(f"Veri okunamadı: {hata}")
    st.stop()

kayit_sayisi_metin = f"{len(df):,}".replace(",", ".")
ozellik_sayisi_metin = str(len(df.columns) - 1)

gecerli_gorunumler = {"home", "karar_agaci", "lojistik", "naive_bayes", "knn", "karsilastir", "tahmin"}
aktif_gorunum = st.query_params.get("view", "home")
if aktif_gorunum not in gecerli_gorunumler:
    aktif_gorunum = "home"


# =========================================================
# SOL KONTROL PANELI (SIDEBAR)
# =========================================================
st.sidebar.markdown(
    """<div class="sidebar-brand">
<div style="font-size:1.45rem;">🫀</div>
<div>
<div class="sidebar-brand-title">Fabrica Atlas</div>
<div class="sidebar-brand-sub">Saf NumPy ML Laboratuvarı</div>
</div>
</div>""",
    unsafe_allow_html=True,
)

if st.sidebar.button("🏠 Ana Sayfa (Kalp Atlası)", use_container_width=True):
    st.query_params["view"] = "home"
    st.rerun()

karsilastir_btn = st.sidebar.button(
    "⚡ Dört Modeli Karşılaştır",
    type="primary",
    use_container_width=True,
    help="Optimize Karar Ağacı, Lojistik Regresyon, Naive Bayes ve KNN modellerini aynı test setinde yarıştırır.",
)
if karsilastir_btn:
    st.query_params["view"] = "karsilastir"
    st.rerun()

if st.sidebar.button(
    "🩺 Hasta Tahmin Sayfası",
    use_container_width=True,
    help="Kişi özelliklerini seçerek 4 algoritmadan canlı kalp hastalığı tahmini alır.",
):
    st.query_params["view"] = "tahmin"
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown(
    '<div class="sidebar-section-label">Özel Parametre Çalıştır</div>',
    unsafe_allow_html=True,
)

algoritma = st.sidebar.radio(
    "Algoritma",
    ["Karar Ağacı", "Lojistik Regresyon", "Naive Bayes", "KNN"],
    help="Tüm algoritmalar hazır ML kütüphanesi olmadan saf NumPy ile kodlanmıştır.",
)

mod = st.sidebar.radio(
    "Çalışma modu",
    ["Manuel ayarlar", "Otomatik optimizasyon (Grid Search)"],
    help="Grid Search, tüm hiperparametre kombinasyonlarını doğrulama setinde tarar.",
)

with st.sidebar.form("model_ayarlari"):
    st.markdown("**Parametre Ayarları**")

    if algoritma == "Karar Ağacı":
        if mod == "Manuel ayarlar":
            kriter = st.selectbox("Bölünme kriteri", ["gini", "entropy"])
            derinlik = st.slider("Karar ağacının derinliği", 1, 12, 6, 1)
            min_bolunme = st.slider("Dallanma için minimum kayıt", 50, 1000, 200, 50)
            min_yaprak = st.slider("Her dalda minimum kayıt", 10, 200, 50, 10)
            egitim_yuzdesi = st.slider("Eğitim verisi oranı (%)", 60, 90, 80, 5)
            budama_uygula = st.checkbox("Hata azaltmalı budama (Pruning)", value=True)
        else:
            egitim_yuzdesi = st.slider("Eğitim verisi oranı (%)", 60, 90, 80, 5)
            budama_uygula = st.checkbox("Hata azaltmalı budama (Pruning)", value=True)
    elif algoritma == "Lojistik Regresyon":
        budama_uygula = False
        if mod == "Manuel ayarlar":
            optimizasyon = st.selectbox("Optimizasyon algoritması", ["adam", "sgd", "momentum", "rmsprop"])
            ogrenme_orani = st.select_slider("Öğrenme oranı", options=[0.005, 0.01, 0.05, 0.1, 0.5], value=0.05)
            iterasyon = st.slider("Maksimum iterasyon", 50, 1000, 300, 50)
            l2_lambda = st.select_slider("L2 düzenlileme (lambda)", options=[0.0, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
            egitim_yuzdesi = st.slider("Eğitim verisi oranı (%)", 60, 90, 80, 5)
        else:
            egitim_yuzdesi = st.slider("Eğitim verisi oranı (%)", 60, 90, 80, 5)
    elif algoritma == "Naive Bayes":
        budama_uygula = False
        if mod == "Manuel ayarlar":
            nb_yontem = st.selectbox("Dağılım yöntemi", ["hibrit", "gaussian", "histogram"])
            nb_alpha = st.select_slider("Laplace düzeltme (alpha)", options=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0], value=1.0)
            nb_var_smoothing = st.select_slider("Varyans yumuşatma", options=[1e-9, 1e-6, 1e-3, 1e-1], value=1e-6)
            nb_bin_sayisi = st.select_slider("Histogram kutu sayısı", options=[8, 12, 16, 24], value=16)
            egitim_yuzdesi = st.slider("Eğitim verisi oranı (%)", 60, 90, 80, 5)
        else:
            egitim_yuzdesi = st.slider("Eğitim verisi oranı (%)", 60, 90, 80, 5)
    else:
        budama_uygula = False
        if mod == "Manuel ayarlar":
            knn_k = st.slider("Komşu sayısı (K)", 3, 51, 25, 2)
            knn_mesafe = st.selectbox("Mesafe metriği", ["kosinus", "oklid"])
            knn_agirliklandirma = st.selectbox("Komşu oylaması", ["mesafe", "esit"])
            knn_ozellik_agirlikli = st.checkbox("Korelasyon özellik ağırlıklandırma", value=True)
            knn_maks_ref = st.select_slider("Referans örnek sayısı", options=[5000, 10000, 15000, 25000], value=15000)
            egitim_yuzdesi = st.slider("Eğitim verisi oranı (%)", 60, 90, 80, 5)
        else:
            knn_maks_ref = st.select_slider("Referans örnek sayısı", options=[5000, 10000, 15000, 25000], value=15000)
            egitim_yuzdesi = st.slider("Eğitim verisi oranı (%)", 60, 90, 80, 5)

    calistir = st.form_submit_button("Seçili Ayarla Çalıştır", type="primary", use_container_width=True)


# =========================================================
# UST ATLAS NAVIGASYON CUBUGU
# =========================================================
nav_ogeleri = [
    ("home", "Ana Sayfa"),
    ("karar_agaci", "Karar Ağacı"),
    ("lojistik", "Lojistik Regresyon"),
    ("naive_bayes", "Naive Bayes"),
    ("knn", "KNN"),
    ("tahmin", "🩺 Hasta Tahmini"),
    ("karsilastir", "⚡ 4'lü Karşılaştırma"),
]
nav_html = "".join(
    f'<a href="?view={kod}" target="_self" class="fabrica-nav-pill {"active" if (not calistir and aktif_gorunum == kod) else ""}">{ad}</a>'
    for kod, ad in nav_ogeleri
)
st.markdown(
    f"""<div class="fabrica-topbar">
<div class="fabrica-topbar-brand">
<span>TABULA I</span>
<span style="color:var(--rule-strong);">//</span>
<span>CARDIOML FABRICA</span>
</div>
<div class="fabrica-nav-links">{nav_html}</div>
</div>""",
    unsafe_allow_html=True,
)


# =========================================================
# GORUNUM A: SOL FORM ILE OZEL CALISTIRMA
# =========================================================
if calistir:
    baslangic = time.time()
    dogrulama_gerekli = (
        mod == "Otomatik optimizasyon (Grid Search)"
        or (algoritma == "Karar Ağacı" and budama_uygula)
    )
    with st.spinner(f"{algoritma} seçili parametrelerle eğitiliyor..."):
        (
            X_egitim,
            y_egitim,
            X_dogrulama,
            y_dogrulama,
            X_test,
            y_test,
            ozellik_isimleri,
        ) = veri_hazirla(df, egitim_yuzdesi, dogrulama_gerekli)

        ekstra = {}
        sonuclar_df = None
        if algoritma == "Karar Ağacı":
            alg_kod = "karar_agaci"
            if mod == "Otomatik optimizasyon (Grid Search)":
                model, ayar, sonuclar_df = hiperparametre_ara(
                    X_egitim, y_egitim, X_dogrulama, y_dogrulama, ozellik_isimleri,
                    hedef_metrik="accuracy", budama_uygula=budama_uygula,
                )
            else:
                model = KararAgaci(derinlik=derinlik, min_bolunme=min_bolunme, min_yaprak=min_yaprak, kriter=kriter)
                model.fit(X_egitim, y_egitim, ozellik_isimleri)
                if budama_uygula:
                    model.buda(X_dogrulama, y_dogrulama)
                ayar = {"kriter": kriter, "derinlik": derinlik, "min_bolunme": min_bolunme, "min_yaprak": min_yaprak}
            ekstra["agac_metni"] = "\n".join(agaci_yaziya_cevir(model.agac, model.ozellik_isimleri))
            ekstra["budanan"] = model.budanan_dugum_sayisi
        elif algoritma == "Lojistik Regresyon":
            alg_kod = "lojistik"
            if mod == "Otomatik optimizasyon (Grid Search)":
                model, ayar, sonuclar_df = hiperparametre_ara_lojistik(
                    X_egitim, y_egitim, X_dogrulama, y_dogrulama, ozellik_isimleri, hedef_metrik="accuracy",
                )
            else:
                model = LojistikRegresyon(ogrenme_orani=ogrenme_orani, iterasyon=iterasyon, l2_lambda=l2_lambda, optimizasyon=optimizasyon)
                model.fit(X_egitim, y_egitim, ozellik_isimleri)
                ayar = {"optimizasyon": optimizasyon, "ogrenme_orani": ogrenme_orani, "l2_lambda": l2_lambda}
            ekstra["ozellik_df"] = model.ozellik_agirliklari_df()
            ekstra["kayip_gecmisi"] = model.kayip_gecmisi
            ekstra["tamamlanan_iterasyon"] = model.tamamlanan_iterasyon
        elif algoritma == "Naive Bayes":
            alg_kod = "naive_bayes"
            if mod == "Otomatik optimizasyon (Grid Search)":
                model, ayar, sonuclar_df = hiperparametre_ara_naive_bayes(
                    X_egitim, y_egitim, X_dogrulama, y_dogrulama, ozellik_isimleri, hedef_metrik="accuracy",
                )
            else:
                model = NaiveBayes(yontem=nb_yontem, var_smoothing=nb_var_smoothing, alpha=nb_alpha, bin_sayisi=nb_bin_sayisi)
                model.fit(X_egitim, y_egitim, ozellik_isimleri)
                ayar = {"yontem": nb_yontem, "alpha": nb_alpha, "var_smoothing": nb_var_smoothing}
            ekstra["ozellik_df"] = model.ozellik_ayirt_edicilik_df()
        else:
            alg_kod = "knn"
            if mod == "Otomatik optimizasyon (Grid Search)":
                model, ayar, sonuclar_df = hiperparametre_ara_knn(
                    X_egitim, y_egitim, X_dogrulama, y_dogrulama, ozellik_isimleri,
                    hedef_metrik="accuracy", maks_referans=knn_maks_ref,
                )
            else:
                model = KNN(
                    k=knn_k,
                    mesafe=knn_mesafe,
                    agirliklandirma=knn_agirliklandirma,
                    ozellik_agirlikli=knn_ozellik_agirlikli,
                    maks_referans=knn_maks_ref,
                )
                model.fit(X_egitim, y_egitim, ozellik_isimleri)
                ayar = {
                    "k": knn_k,
                    "mesafe": knn_mesafe,
                    "agirliklandirma": knn_agirliklandirma,
                    "ozellik_agirlikli": knn_ozellik_agirlikli,
                }
            ekstra["ozellik_df"] = model.ozellik_agirliklari_df()
            ekstra["referans_sayisi"] = len(model.X_ref_s)

        test_olasilik = model.predict_proba(X_test)
        tahmin = (test_olasilik >= 0.5).astype(int)
        metrikler = metrikleri_hesapla(y_test, tahmin, olasiliklar=test_olasilik)
        sonuc = {
            "ayar": ayar,
            "sonuclar_df": sonuclar_df,
            "metrikler": metrikler,
            "sure": time.time() - baslangic,
            "n_train": len(y_egitim),
            "n_val": len(y_dogrulama) if dogrulama_gerekli else 0,
            "n_test": len(y_test),
            "ekstra": ekstra,
        }
    goster_tekli_sonuc(algoritma, alg_kod, sonuc, egitim_yuzdesi)


# =========================================================
# GORUNUM B: ANA SAYFA (ETKILESIMLI ANATOMIK KALP ATLASI)
# =========================================================
elif aktif_gorunum == "home":
    st.markdown(
        """<div class="atlas-header">
<div class="mono-label">DE HUMANI CORDIS // İNTERAKTİF MAKİNE ÖĞRENMESİ ATLASI</div>
<h1 class="atlas-title">Kalp Hastalığı Tahmini için <em>Anatomik</em> Model Haritası</h1>
<p class="atlas-sub">Kalbin üzerindeki noktalı hatlardan birine tıklayarak o algoritmanın optimize edilmiş test sonuçlarını inceleyebilir veya tüm modelleri aynı test verisi üzerinde karşılaştırabilirsin.</p>
</div>""",
        unsafe_allow_html=True,
    )

    heart_img_html = (
        f'<img src="{heart_uri}" alt="Anatomik Kalp Atlası" class="fabrica-heart-img">'
        if heart_uri
        else '<div style="font-size:6rem;">🫀</div>'
    )

    st.markdown(
        f"""<div class="fabrica-plate">
<div class="plate-crosshair ch-tl">+</div>
<div class="plate-crosshair ch-tr">+</div>
<div class="plate-crosshair ch-bl">+</div>
<div class="plate-crosshair ch-br">+</div>
<div class="plate-ring-1"></div>
<div class="plate-ring-2"></div>
<div class="fabrica-heart-wrap">{heart_img_html}</div>
<svg class="fabrica-svg" viewBox="0 0 1000 540" preserveAspectRatio="none">
<polyline points="445,195 360,125 275,125" fill="none" stroke="#9B2226" stroke-width="1.6" stroke-dasharray="4 5"/>
<circle cx="445" cy="195" r="7" fill="rgba(155,34,38,0.22)" class="pin-halo"/>
<circle cx="445" cy="195" r="4" fill="#9B2226"/>
<polyline points="550,185 635,125 725,125" fill="none" stroke="#9B2226" stroke-width="1.6" stroke-dasharray="4 5"/>
<circle cx="550" cy="185" r="7" fill="rgba(155,34,38,0.22)" class="pin-halo"/>
<circle cx="550" cy="185" r="4" fill="#9B2226"/>
<polyline points="455,320 365,415 275,415" fill="none" stroke="#9B2226" stroke-width="1.6" stroke-dasharray="4 5"/>
<circle cx="455" cy="320" r="7" fill="rgba(155,34,38,0.22)" class="pin-halo"/>
<circle cx="455" cy="320" r="4" fill="#9B2226"/>
<polyline points="545,335 635,415 725,415" fill="none" stroke="#9B2226" stroke-width="1.6" stroke-dasharray="4 5"/>
<circle cx="545" cy="335" r="7" fill="rgba(155,34,38,0.22)" class="pin-halo"/>
<circle cx="545" cy="335" r="4" fill="#9B2226"/>
</svg>
<a href="?view=karar_agaci" target="_self" class="callout-node node-tl">
<div class="callout-fig">FIG. 01 // AĞAÇ MİMARİSİ</div>
<div class="callout-title">Karar Ağacı</div>
<div class="callout-desc">32-Bin Histogram, Gini/Entropy, Budama ve 24 Grid Search</div>
<div class="callout-cta">Sonuçları İncele →</div>
</a>
<a href="?view=lojistik" target="_self" class="callout-node node-tr">
<div class="callout-fig">FIG. 02 // DOĞRUSAL MODEL</div>
<div class="callout-title">Lojistik Regresyon</div>
<div class="callout-desc">SGD, Momentum, RMSProp, Adam ve L2 ile 27 Grid Search</div>
<div class="callout-cta">Sonuçları İncele →</div>
</a>
<a href="?view=naive_bayes" target="_self" class="callout-node node-bl">
<div class="callout-fig">FIG. 03 // OLASILIK TEOREMİ</div>
<div class="callout-title">Naive Bayes</div>
<div class="callout-desc">Gaussian, Hibrit, Histogram ve Laplace ile 24 Grid Search</div>
<div class="callout-cta">Sonuçları İncele →</div>
</a>
<a href="?view=knn" target="_self" class="callout-node node-br">
<div class="callout-fig">FIG. 04 // KOMŞULUK MODELİ</div>
<div class="callout-title">K-En Yakın Komşu (KNN)</div>
<div class="callout-desc">Öklid &amp; Kosinüs, Korelasyon Ağırlıklı, 32 Grid Search</div>
<div class="callout-cta">Sonuçları İncele →</div>
</a>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<div class="folio-strip">
<div class="folio-cell">
<div class="mono-label">TOPLAM KAYIT</div>
<div class="folio-val">{kayit_sayisi_metin}</div>
</div>
<div class="folio-cell">
<div class="mono-label">TAHMİN ÖZELLİĞİ</div>
<div class="folio-val">{ozellik_sayisi_metin}</div>
</div>
<div class="folio-cell">
<div class="mono-label">ALGORİTMA</div>
<div class="folio-val">4</div>
</div>
<div class="folio-cell">
<div class="mono-label">HEDEF SÜTUN</div>
<div class="folio-val" style="font-size:1.35rem; padding-top:4px;">{HEDEF_SUTUN}</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    with st.expander("Veri setinden ilk 10 kaydı görüntüle"):
        st.dataframe(df.head(10), use_container_width=True)


# =========================================================
# GORUNUM C: KALP NOKTALARINDAN SECILEN TEKLI MODEL
# =========================================================
elif aktif_gorunum in ("karar_agaci", "lojistik", "naive_bayes", "knn"):
    isim_haritasi = {
        "karar_agaci": "Karar Ağacı",
        "lojistik": "Lojistik Regresyon",
        "naive_bayes": "Naive Bayes",
        "knn": "K-En Yakın Komşu (KNN)",
    }
    with st.spinner(f"{isim_haritasi[aktif_gorunum]} optimize ediliyor ve sonuçlar hazırlanıyor..."):
        sonuc = optimize_model_calistir(aktif_gorunum, egitim_yuzdesi=80)
    goster_tekli_sonuc(isim_haritasi[aktif_gorunum], aktif_gorunum, sonuc, 80)


# =========================================================
# GORUNUM C-2: HASTA TAHMIN SAYFASI (SOLDAN BAGIMSIZ)
# =========================================================
elif aktif_gorunum == "tahmin":
    st.markdown(
        '<a class="back-link" href="?view=home" target="_self">← Kalp Atlasına Dön (Ana Sayfa)</a>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class="result-header">
<h2 class="result-title">🩺 Hasta Kalp Hastalığı Risk Tahmini</h2>
<span class="mono-pill">4 Optimize Algoritma • Tahmin vs Gerçek Sonuç</span>
</div>""",
        unsafe_allow_html=True,
    )

    with st.spinner("Optimize modeller hazırlanıyor..."):
        ka_res = optimize_model_calistir("karar_agaci", egitim_yuzdesi=80)
        lr_res = optimize_model_calistir("lojistik", egitim_yuzdesi=80)
        nb_res = optimize_model_calistir("naive_bayes", egitim_yuzdesi=80)
        knn_res = optimize_model_calistir("knn", egitim_yuzdesi=80)

    ozellik_isimleri = ka_res["ozellik_isimleri"]
    tab_gercek, tab_manuel = st.tabs([
        "📋 Test Setinden Gerçek Hasta (Tahmin vs Gerçek)",
        "🧪 Kendi Kişini Oluştur (25 Özellik Seç)",
    ])

    with tab_gercek:
        df_test_ham = test_seti_ham_getir(df, egitim_yuzdesi=80)
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            secilen_cinsiyet = st.selectbox(
                "Cinsiyet filtresi",
                ["Tümü", "Female", "Male"],
                key="hasta_cinsiyet_filtre",
            )
        with f_col2:
            secilen_durum = st.selectbox(
                "Gerçek durum filtresi",
                ["Tümü", "Kalp Hastası (1)", "Sağlıklı (0)"],
                key="hasta_durum_filtre",
            )

        filtreli_df = df_test_ham.copy()
        if secilen_cinsiyet != "Tümü":
            filtreli_df = filtreli_df[filtreli_df["sex"] == secilen_cinsiyet]
        if secilen_durum == "Kalp Hastası (1)":
            filtreli_df = filtreli_df[filtreli_df[HEDEF_SUTUN] == 1]
        elif secilen_durum == "Sağlıklı (0)":
            filtreli_df = filtreli_df[filtreli_df[HEDEF_SUTUN] == 0]

        with f_col3:
            hasta_sira = st.number_input(
                f"Hasta sıra no (0 - {max(len(filtreli_df) - 1, 0)})",
                min_value=0,
                max_value=max(len(filtreli_df) - 1, 0),
                value=0,
                step=1,
                key="hasta_sira_no",
            )

        if len(filtreli_df) == 0:
            st.warning("Bu filtreye uygun test hastası bulunamadı.")
        else:
            secilen_satir = filtreli_df.iloc[int(hasta_sira)]
            gercek_deger = int(secilen_satir[HEDEF_SUTUN])
            kisi_dict = secilen_satir.drop(labels=[HEDEF_SUTUN]).to_dict()
            X_kisi = tek_kisi_vektorle(kisi_dict, ozellik_isimleri)

            gercek_metin = "Kalp Hastası (1)" if gercek_deger == 1 else "Sağlıklı (0)"
            if gercek_deger == 1:
                st.error(f"**Seçilen Hastanın Gerçek Sonucu:** {gercek_metin} — Yaş: {int(kisi_dict['age'])}, Cinsiyet: {kisi_dict['sex']}, Tansiyon: {int(kisi_dict['resting_bp'])}, Kolesterol: {int(kisi_dict['cholesterol'])}")
            else:
                st.success(f"**Seçilen Hastanın Gerçek Sonucu:** {gercek_metin} — Yaş: {int(kisi_dict['age'])}, Cinsiyet: {kisi_dict['sex']}, Tansiyon: {int(kisi_dict['resting_bp'])}, Kolesterol: {int(kisi_dict['cholesterol'])}")

            kartlar_html = []
            for ad, res in [
                ("🌳 Karar Ağacı", ka_res),
                ("📈 Lojistik Regresyon", lr_res),
                ("🎲 Naive Bayes", nb_res),
                ("🧭 KNN", knn_res),
            ]:
                olasilik = float(res["model"].predict_proba(X_kisi)[0])
                tahmin = 1 if olasilik >= 0.5 else 0
                tahmin_metin = "Riskli (1)" if tahmin == 1 else "Sağlıklı (0)"
                dogru_mu = (tahmin == gercek_deger)
                kartlar_html.append(
                    render_kpi_card(
                        ad,
                        tahmin_metin,
                        f"Risk Olasılığı: %{olasilik * 100:.1f} • Gerçek: {gercek_deger}",
                        delta="✓ Doğru Tahmin" if dogru_mu else "✗ Yanlış Tahmin",
                        delta_positive=dogru_mu,
                    )
                )
            st.markdown(
                '<div class="bento-grid bento-cols-4">' + "".join(kartlar_html) + "</div>",
                unsafe_allow_html=True,
            )

            with st.expander("Seçilen hastanın tüm 25 özelliğini görüntüle"):
                ozellik_tablo = pd.DataFrame({
                    "Özellik": list(kisi_dict.keys()),
                    "Değer": [str(v) for v in kisi_dict.values()],
                })
                st.dataframe(ozellik_tablo, use_container_width=True)

    with tab_manuel:
        with st.form("hasta_ozellik_formu"):
            st.markdown("#### 1. Demografik & Yaşam Tarzı")
            d1, d2, d3 = st.columns(3)
            with d1:
                h_age = st.slider("Yaş", 29, 80, 53)
                h_sex = st.selectbox("Cinsiyet", ["Female", "Male"])
                h_bmi = st.number_input("BMI (Vücut Kitle İndeksi)", 15.0, 54.6, 28.0, step=0.5)
                h_region = st.selectbox("Bölge", ["Midwest", "Northeast", "Southeast", "Southwest", "West"])
            with d2:
                h_income = st.selectbox("Gelir Düzeyi", ["Middle", "Low", "High"])
                h_smoking = st.selectbox("Sigara Kullanımı", ["Never", "Former", "Current"])
                h_alcohol = st.selectbox("Alkol Tüketimi", ["Non-drinker", "Moderate", "Heavy"])
                h_activity = st.selectbox("Fiziksel Aktivite", ["Active", "Light", "Sedentary", "Very Active"])
            with d3:
                h_diet = st.slider("Diyet Kalitesi (1-10)", 1, 10, 6)
                h_sleep = st.slider("Uyku Süresi (saat)", 4.0, 10.0, 7.0, step=0.1)
                h_stress = st.slider("Stres Düzeyi (1-10)", 1, 10, 6)

            st.markdown("#### 2. Tıbbi Geçmiş")
            t1, t2, t3 = st.columns(3)
            with t1:
                h_diabetes = 1 if st.checkbox("Diyabet (Şeker Hastalığı)") else 0
                h_hypertension = 1 if st.checkbox("Hipertansiyon (Yüksek Tansiyon)") else 0
            with t2:
                h_family = 1 if st.checkbox("Ailede Kalp Hastalığı Öyküsü") else 0
                h_prev = 1 if st.checkbox("Daha Önce Kalp Olayı Geçirdi") else 0
            with t3:
                h_fbs = 1 if st.checkbox("Yüksek Açlık Kan Şekeri") else 0
                h_angina = 1 if st.checkbox("Egzersizle Göğüs Ağrısı (Angina)") else 0

            st.markdown("#### 3. Klinik & EKG Ölçümleri")
            k1, k2, k3 = st.columns(3)
            with k1:
                h_bp = st.slider("Dinlenme Tansiyonu (mmHg)", 80, 200, 140)
                h_chol = st.slider("Kolesterol (mg/dL)", 100, 400, 218)
                h_max_hr = st.slider("Maksimum Kalp Hızı", 107, 202, 174)
            with k2:
                h_ecg = st.selectbox("Dinlenme EKG", ["Normal", "Left Ventricular Hypertrophy", "ST-T Abnormality"])
                h_st_dep = st.slider("ST Çökmesi (st_depression)", 0.0, 6.2, 1.1, step=0.1)
                h_st_slope = st.selectbox("ST Eğimi (st_slope)", ["Upsloping", "Flat", "Downsloping"])
            with k3:
                h_avg_hr = st.slider("Ort. Dinlenme Nabzı", 40.0, 110.0, 70.0, step=0.5)
                h_hrv = st.slider("HRV Skoru", 5.0, 100.0, 53.0, step=0.5)

            tahmin_yap = st.form_submit_button(
                "🩺 4 Algoritmayla Kalp Hastalığı Tahmini Yap",
                type="primary",
                use_container_width=True,
            )

        if tahmin_yap:
            manuel_dict = {
                "age": h_age,
                "sex": h_sex,
                "bmi": h_bmi,
                "region": h_region,
                "income_level": h_income,
                "smoking_status": h_smoking,
                "alcohol_intake": h_alcohol,
                "physical_activity": h_activity,
                "diet_quality": h_diet,
                "sleep_hours": h_sleep,
                "stress_level": h_stress,
                "diabetes": h_diabetes,
                "hypertension": h_hypertension,
                "family_history": h_family,
                "previous_heart_event": h_prev,
                "resting_bp": h_bp,
                "cholesterol": h_chol,
                "fasting_blood_sugar": h_fbs,
                "max_heart_rate": h_max_hr,
                "resting_ecg": h_ecg,
                "exercise_angina": h_angina,
                "st_depression": h_st_dep,
                "st_slope": h_st_slope,
                "avg_resting_hr": h_avg_hr,
                "hrv_score": h_hrv,
            }
            X_manuel = tek_kisi_vektorle(manuel_dict, ozellik_isimleri)

            olasiliklar = []
            tahminler_bin = []
            kartlar_manuel = []
            for ad, res in [
                ("🌳 Karar Ağacı", ka_res),
                ("📈 Lojistik Regresyon", lr_res),
                ("🎲 Naive Bayes", nb_res),
                ("🧭 KNN", knn_res),
            ]:
                olasilik = float(res["model"].predict_proba(X_manuel)[0])
                tahmin = 1 if olasilik >= 0.5 else 0
                olasiliklar.append(olasilik)
                tahminler_bin.append(tahmin)
                kartlar_manuel.append(
                    render_kpi_card(
                        ad,
                        "Riskli (1)" if tahmin == 1 else "Sağlıklı (0)",
                        f"Kalp Hastalığı Olasılığı: %{olasilik * 100:.1f}",
                        delta="Yüksek Risk" if tahmin == 1 else "Düşük Risk",
                        delta_positive=(tahmin == 0),
                    )
                )

            riskli_sayisi = sum(tahminler_bin)
            ort_olasilik = float(np.mean(olasiliklar)) * 100
            if riskli_sayisi >= 2:
                st.error(f"**Ortak Model Kararı:** {riskli_sayisi}/4 algoritma bu profilde **Kalp Hastalığı Riski (1)** öngörüyor (Ortalama Risk Olasılığı: **%{ort_olasilik:.1f}**).")
            else:
                st.success(f"**Ortak Model Kararı:** {4 - riskli_sayisi}/4 algoritma bu profili **Sağlıklı (0)** olarak değerlendiriyor (Ortalama Risk Olasılığı: **%{ort_olasilik:.1f}**).")

            st.markdown(
                '<div class="bento-grid bento-cols-4">' + "".join(kartlar_manuel) + "</div>",
                unsafe_allow_html=True,
            )


# =========================================================
# GORUNUM D: 4'LU KARSILASTIRMA
# =========================================================
else:
    with st.spinner("Her dört optimize model çalıştırılıyor ve karşılaştırılıyor..."):
        ka_res = optimize_model_calistir("karar_agaci", egitim_yuzdesi=80)
        lr_res = optimize_model_calistir("lojistik", egitim_yuzdesi=80)
        nb_res = optimize_model_calistir("naive_bayes", egitim_yuzdesi=80)
        knn_res = optimize_model_calistir("knn", egitim_yuzdesi=80)

    ka_m, lr_m, nb_m, knn_m = ka_res["metrikler"], lr_res["metrikler"], nb_res["metrikler"], knn_res["metrikler"]
    ka_a, lr_a, nb_a, knn_a = ka_res["ayar"], lr_res["ayar"], nb_res["ayar"], knn_res["ayar"]
    toplam_sure = ka_res["sure"] + lr_res["sure"] + nb_res["sure"] + knn_res["sure"]

    st.markdown(
        '<a class="back-link" href="?view=home" target="_self">← Kalp Atlasına Dön (Ana Sayfa)</a>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="result-header">
<h2 class="result-title">⚡ Dörtlü Optimize Model Karşılaştırması</h2>
<span class="mono-pill">Eğitim: {ka_res['n_train']:,} | Doğrulama: {ka_res['n_val']:,} | Test: {ka_res['n_test']:,} • {toplam_sure:.2f} sn</span>
</div>""",
        unsafe_allow_html=True,
    )

    en_iyi_acc = max(ka_m["accuracy"], lr_m["accuracy"], nb_m["accuracy"], knn_m["accuracy"])
    st.markdown(
        '<div class="bento-grid bento-cols-4">'
        + render_kpi_card(
            "🌳 Karar Ağacı",
            f"%{ka_m['accuracy'] * 100:.2f}",
            f"F1: %{ka_m['f1'] * 100:.2f} • AUC: {ka_m['roc_auc']:.4f} • {ka_res['sure']:.2f} sn",
            delta="★ En Yüksek Doğruluk" if ka_m["accuracy"] == en_iyi_acc else f"{ka_a['kriter'].upper()} d={ka_a['derinlik']}",
            delta_positive=(ka_m["accuracy"] == en_iyi_acc),
        )
        + render_kpi_card(
            "📈 Lojistik Regresyon",
            f"%{lr_m['accuracy'] * 100:.2f}",
            f"F1: %{lr_m['f1'] * 100:.2f} • AUC: {lr_m['roc_auc']:.4f} • {lr_res['sure']:.2f} sn",
            delta="★ En Yüksek Doğruluk" if lr_m["accuracy"] == en_iyi_acc else f"{lr_a['optimizasyon'].upper()} lr={lr_a['ogrenme_orani']}",
            delta_positive=(lr_m["accuracy"] == en_iyi_acc),
        )
        + render_kpi_card(
            "🎲 Naive Bayes",
            f"%{nb_m['accuracy'] * 100:.2f}",
            f"F1: %{nb_m['f1'] * 100:.2f} • AUC: {nb_m['roc_auc']:.4f} • {nb_res['sure']:.2f} sn",
            delta="★ En Yüksek Doğruluk" if nb_m["accuracy"] == en_iyi_acc else f"{nb_a['yontem'].upper()} α={nb_a['alpha']}",
            delta_positive=(nb_m["accuracy"] == en_iyi_acc),
        )
        + render_kpi_card(
            "🧭 KNN",
            f"%{knn_m['accuracy'] * 100:.2f}",
            f"F1: %{knn_m['f1'] * 100:.2f} • AUC: {knn_m['roc_auc']:.4f} • {knn_res['sure']:.2f} sn",
            delta="★ En Yüksek Doğruluk" if knn_m["accuracy"] == en_iyi_acc else f"{knn_a['mesafe'].upper()} k={knn_a['k']}",
            delta_positive=(knn_m["accuracy"] == en_iyi_acc),
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Tüm Performans Metrikleri (Test Seti)")
    karsilastirma_satirlari = [
        ("Accuracy (%)", ka_m["accuracy"] * 100, lr_m["accuracy"] * 100, nb_m["accuracy"] * 100, knn_m["accuracy"] * 100, True),
        ("Precision (%)", ka_m["precision"] * 100, lr_m["precision"] * 100, nb_m["precision"] * 100, knn_m["precision"] * 100, True),
        ("Recall / Duyarlılık (%)", ka_m["recall"] * 100, lr_m["recall"] * 100, nb_m["recall"] * 100, knn_m["recall"] * 100, True),
        ("Specificity / Özgüllük (%)", ka_m["specificity"] * 100, lr_m["specificity"] * 100, nb_m["specificity"] * 100, knn_m["specificity"] * 100, True),
        ("F1 Skoru (%)", ka_m["f1"] * 100, lr_m["f1"] * 100, nb_m["f1"] * 100, knn_m["f1"] * 100, True),
        ("ROC-AUC", ka_m["roc_auc"], lr_m["roc_auc"], nb_m["roc_auc"], knn_m["roc_auc"], True),
        ("Log-Loss (düşük = iyi)", ka_m["log_loss"], lr_m["log_loss"], nb_m["log_loss"], knn_m["log_loss"], False),
        ("Süre (saniye, düşük = hızlı)", ka_res["sure"], lr_res["sure"], nb_res["sure"], knn_res["sure"], False),
    ]
    tablo_satirlari = []
    for ad, ka_v, lr_v, nb_v, knn_v, yuksek_iyi in karsilastirma_satirlari:
        adaylar = {"Karar Ağacı": ka_v, "Lojistik Regresyon": lr_v, "Naive Bayes": nb_v, "KNN": knn_v}
        en_iyi_ad = max(adaylar, key=adaylar.get) if yuksek_iyi else min(adaylar, key=adaylar.get)
        tablo_satirlari.append({
            "Metrik": ad,
            "Karar Ağacı": round(ka_v, 4),
            "Lojistik Regresyon": round(lr_v, 4),
            "Naive Bayes": round(nb_v, 4),
            "KNN": round(knn_v, 4),
            "En İyi": en_iyi_ad,
        })
    st.dataframe(pd.DataFrame(tablo_satirlari), use_container_width=True)

    st.markdown("#### Görsel Performans Grafiği")
    grafik_df = pd.DataFrame(
        {
            "Karar Ağacı": [ka_m["accuracy"], ka_m["precision"], ka_m["recall"], ka_m["specificity"], ka_m["f1"], ka_m["roc_auc"]],
            "Lojistik Regresyon": [lr_m["accuracy"], lr_m["precision"], lr_m["recall"], lr_m["specificity"], lr_m["f1"], lr_m["roc_auc"]],
            "Naive Bayes": [nb_m["accuracy"], nb_m["precision"], nb_m["recall"], nb_m["specificity"], nb_m["f1"], nb_m["roc_auc"]],
            "KNN": [knn_m["accuracy"], knn_m["precision"], knn_m["recall"], knn_m["specificity"], knn_m["f1"], knn_m["roc_auc"]],
        },
        index=["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"],
    )
    st.bar_chart(grafik_df)

    st.markdown("#### Karışıklık Matrisleri")
    cm_r1_c1, cm_r1_c2 = st.columns(2)
    with cm_r1_c1:
        st.caption(f"Karar Ağacı ({ka_a['kriter'].upper()}, d={ka_a['derinlik']}, budanan={ka_res['ekstra']['budanan']})")
        st.dataframe(pd.DataFrame([[ka_m["tn"], ka_m["fp"]], [ka_m["fn"], ka_m["tp"]]], index=["Gerçek: Sağlıklı (0)", "Gerçek: Kalp Hastası (1)"], columns=["Tahmin: Sağlıklı (0)", "Tahmin: Kalp Hastası (1)"]), use_container_width=True)
    with cm_r1_c2:
        st.caption(f"Lojistik Regresyon ({lr_a['optimizasyon'].upper()}, lr={lr_a['ogrenme_orani']}, l2={lr_a['l2_lambda']})")
        st.dataframe(pd.DataFrame([[lr_m["tn"], lr_m["fp"]], [lr_m["fn"], lr_m["tp"]]], index=["Gerçek: Sağlıklı (0)", "Gerçek: Kalp Hastası (1)"], columns=["Tahmin: Sağlıklı (0)", "Tahmin: Kalp Hastası (1)"]), use_container_width=True)

    cm_r2_c1, cm_r2_c2 = st.columns(2)
    with cm_r2_c1:
        st.caption(f"Naive Bayes ({nb_a['yontem'].upper()}, alpha={nb_a['alpha']})")
        st.dataframe(pd.DataFrame([[nb_m["tn"], nb_m["fp"]], [nb_m["fn"], nb_m["tp"]]], index=["Gerçek: Sağlıklı (0)", "Gerçek: Kalp Hastası (1)"], columns=["Tahmin: Sağlıklı (0)", "Tahmin: Kalp Hastası (1)"]), use_container_width=True)
    with cm_r2_c2:
        st.caption(f"KNN ({knn_a['mesafe'].upper()}, k={knn_a['k']}, {knn_a['agirliklandirma']}, ağırlıklı={knn_a['ozellik_agirlikli']})")
        st.dataframe(pd.DataFrame([[knn_m["tn"], knn_m["fp"]], [knn_m["fn"], knn_m["tp"]]], index=["Gerçek: Sağlıklı (0)", "Gerçek: Kalp Hastası (1)"], columns=["Tahmin: Sağlıklı (0)", "Tahmin: Kalp Hastası (1)"]), use_container_width=True)
    st.caption(
        "0 = Sağlıklı, 1 = Kalp Hastası. Sol üst ve sağ alt kutular doğru tahminleri; "
        "diğer iki kutu ise hatalı tahminleri gösterir."
    )

    with st.expander("Karar Ağacı — En iyi 10 Grid Search sonucu"):
        st.dataframe(ka_res["sonuclar_df"].head(10), use_container_width=True)
    with st.expander("Lojistik Regresyon — En iyi 10 Grid Search sonucu"):
        st.dataframe(lr_res["sonuclar_df"].head(10), use_container_width=True)
    with st.expander("Naive Bayes — En iyi 10 Grid Search sonucu"):
        st.dataframe(nb_res["sonuclar_df"].head(10), use_container_width=True)
    with st.expander("KNN — En iyi 10 Grid Search sonucu"):
        st.dataframe(knn_res["sonuclar_df"].head(10), use_container_width=True)
    with st.expander("Lojistik Regresyon — En etkili özellik ağırlıkları (İlk 15)"):
        st.dataframe(lr_res["ekstra"]["ozellik_df"].head(15), use_container_width=True)
    with st.expander("KNN — Korelasyon özellik ağırlıkları (İlk 15)"):
        st.dataframe(knn_res["ekstra"]["ozellik_df"].head(15), use_container_width=True)
    with st.expander("Naive Bayes — En ayırt edici özellikler (İlk 15)"):
        st.dataframe(nb_res["ekstra"]["ozellik_df"].head(15), use_container_width=True)
    with st.expander("Karar Ağacı — Oluşturulan dallar"):
        st.code(ka_res["ekstra"]["agac_metni"], language="text")


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """<div class="fabrica-footer">
<div>
<div class="mono-label" style="margin-bottom:6px;">CardioML Fabrica</div>
<div>
Hazır makine öğrenmesi kütüphanesi kullanılmadan, yalnızca NumPy ve Pandas ile
sıfırdan geliştirilmiş anatomik ML atlası.
</div>
</div>
<div>
<div class="mono-label" style="margin-bottom:6px;">Algoritmalar</div>
<div>
• Karar Ağacı (Histogram + Budama)<br>
• Lojistik Regresyon (SGD, Momentum, RMSProp, Adam)<br>
• Naive Bayes (Gaussian, Hibrit, Histogram)<br>
• KNN (Öklid, Kosinüs, Korelasyon Ağırlıklı)
</div>
</div>
<div>
<div class="mono-label" style="margin-bottom:6px;">Bilgilendirme</div>
<div class="footer-disclaimer">
Bu proje eğitim amaçlıdır. Üretilen tahminler ve skorlar tıbbi tanı veya
tedavi kararı için kullanılamaz.
</div>
</div>
</div>""",
    unsafe_allow_html=True,
)
