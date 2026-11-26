import numpy as np
import pandas as pd


# =========================================================
# ORTAK YARDIMCILAR: OLCEKLEME VE METRIKLER
# =========================================================
def z_olcekle(X):
    """Egitim verisi uzerinde Z-skor olcekleme uygular."""
    ortalama = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (X - ortalama) / std, ortalama, std


def roc_auc_hesapla(gercek, olasiliklar):
    """Saf NumPy ile sira tabanli (Mann-Whitney U) ROC-AUC hesaplar."""
    gercek = np.asarray(gercek, dtype=int)
    olasiliklar = np.asarray(olasiliklar, dtype=float)
    poz = int(np.sum(gercek == 1))
    neg = len(gercek) - poz
    if poz == 0 or neg == 0:
        return 0.0

    sirali = np.argsort(olasiliklar)
    sp, sy = olasiliklar[sirali], gercek[sirali]
    siralar = np.arange(1, len(olasiliklar) + 1, dtype=float)
    farkli = np.diff(sp) != 0
    baslar = np.concatenate(([0], np.where(farkli)[0] + 1))
    bitisler = np.concatenate((np.where(farkli)[0] + 1, [len(olasiliklar)]))
    for b, e in zip(baslar, bitisler):
        if e - b > 1:
            siralar[b:e] = 0.5 * (b + 1 + e)
    return float((np.sum(siralar[sy == 1]) - poz * (poz + 1) / 2.0) / (poz * neg))


def metrikleri_hesapla(gercek, tahmin, olasiliklar=None):
    """Siniflandirma basari metriklerini hesaplar."""
    tn = int(np.sum((gercek == 0) & (tahmin == 0)))
    fp = int(np.sum((gercek == 0) & (tahmin == 1)))
    fn = int(np.sum((gercek == 1) & (tahmin == 0)))
    tp = int(np.sum((gercek == 1) & (tahmin == 1)))

    acc = (tp + tn) / len(gercek) if len(gercek) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    roc_auc, log_loss = None, None
    if olasiliklar is not None:
        p_f64 = np.asarray(olasiliklar, dtype=float)
        roc_auc = roc_auc_hesapla(gercek, p_f64)
        pc = np.clip(p_f64, 1e-12, 1.0 - 1e-12)
        log_loss = float(-np.mean(gercek * np.log(pc) + (1 - gercek) * np.log(1.0 - pc)))

    return {
        "accuracy": acc, "precision": prec, "recall": rec,
        "specificity": spec, "f1": f1, "roc_auc": roc_auc,
        "log_loss": log_loss, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def _satir_olustur(ayar, y_dogrulama, olasilik):
    """Grid Search adayinin dogrulama metriklerini tek sozlukte birlestirir."""
    met = metrikleri_hesapla(y_dogrulama, (olasilik >= 0.5).astype(int), olasiliklar=olasilik)
    satir = dict(ayar)
    for k in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "log_loss"):
        satir[k] = met[k]
    return satir, met


def _sonuc_tablosu(sonuclar, hedef_metrik):
    return pd.DataFrame(sonuclar).sort_values(by=hedef_metrik, ascending=False).reset_index(drop=True)


def roc_egrisi_hesapla(gercek, olasiliklar, nokta_sayisi=120):
    """Saf NumPy ile ROC egrisi (FPR, TPR) noktalarini hesaplar."""
    y = np.asarray(gercek, dtype=int)
    p = np.asarray(olasiliklar, dtype=float)
    n_pos = int(np.sum(y == 1))
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    sirali = np.argsort(-p)
    sy = y[sirali]
    tp = np.cumsum(sy == 1)
    fp = np.cumsum(sy == 0)

    fpr = np.concatenate(([0.0], fp / n_neg, [1.0]))
    tpr = np.concatenate(([0.0], tp / n_pos, [1.0]))
    if len(fpr) > nokta_sayisi:
        indeks = np.linspace(0, len(fpr) - 1, nokta_sayisi, dtype=int)
        fpr, tpr = fpr[indeks], tpr[indeks]
    return fpr, tpr


def capraz_dogrulama(model_uretici, X, y, ozellik_isimleri, k_kat=5, tohum=42):
    """K-katli capraz dogrulama uygular ve katman bazli sonuclar ile ozet dondurur."""
    rng = np.random.default_rng(tohum)
    indeksler = rng.permutation(len(X))
    katlar = np.array_split(indeksler, k_kat)
    satirlar = []

    for i, test_idx in enumerate(katlar, start=1):
        egitim_idx = np.concatenate([katlar[j] for j in range(k_kat) if j != i - 1])
        model = model_uretici().fit(X[egitim_idx], y[egitim_idx], ozellik_isimleri)
        olasilik = model.predict_proba(X[test_idx])
        satir, _ = _satir_olustur({"kat": i}, y[test_idx], olasilik)
        satirlar.append(satir)

    kat_df = pd.DataFrame(satirlar)
    ozet = {}
    for m in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        ozet[f"{m}_ort"] = float(kat_df[m].mean())
        ozet[f"{m}_std"] = float(kat_df[m].std(ddof=1))
    return kat_df, ozet


# =========================================================
# 1. KARAR AGACI (HISTOGRAM + BUDAMA)
# =========================================================
class KararAgaci:
    def __init__(self, derinlik, min_bolunme, min_yaprak, kriter="gini", bin_sayisi=32):
        self.derinlik = derinlik
        self.min_bolunme = min_bolunme
        self.min_yaprak = min_yaprak
        self.kriter = kriter
        self.bin_sayisi = bin_sayisi
        self.agac = None
        self.ozellik_isimleri = []
        self.budanan_dugum_sayisi = 0

    def safsizlik(self, pozitif, toplam):
        p = pozitif / np.maximum(toplam, 1)
        if self.kriter == "entropy":
            pc = np.clip(p, 1e-12, 1.0 - 1e-12)
            return -(pc * np.log2(pc) + (1.0 - pc) * np.log2(1.0 - pc))
        return 2.0 * p * (1.0 - p)

    @staticmethod
    def binle(X, bin_sayisi=32):
        n, d = X.shape
        X_bin = np.empty((n, d), dtype=np.uint8)
        esik_listesi = []
        qs = np.linspace(0.0, 1.0, bin_sayisi + 1)[1:-1]
        for i in range(d):
            col = X[:, i]
            uniq = np.unique(col)
            esikler = (uniq[:-1] + uniq[1:]) / 2.0 if len(uniq) <= bin_sayisi else np.unique(np.quantile(col, qs))
            X_bin[:, i] = np.searchsorted(esikler, col, side="right").astype(np.uint8)
            esik_listesi.append(esikler)
        return X_bin, esik_listesi

    def en_iyi_bolunme(self, X_bin, y, esik_listesi):
        n = len(y)
        pos_tot = int(np.sum(y))
        ana = float(self.safsizlik(pos_tot, n))
        pos_mask = y == 1
        best = (0.0, None, None)

        for i, esikler in enumerate(esik_listesi):
            nb = len(esikler) + 1
            if nb <= 1:
                continue
            col = X_bin[:, i]
            cnt = np.bincount(col, minlength=nb)
            pos = np.bincount(col[pos_mask], minlength=nb)
            sol_n = np.cumsum(cnt)[:-1]
            sag_n = n - sol_n
            gecerli = (sol_n >= self.min_yaprak) & (sag_n >= self.min_yaprak)
            if not np.any(gecerli):
                continue
            sol_p = np.cumsum(pos)[:-1]
            kazanc = np.where(
                gecerli,
                ana - ((sol_n / n) * self.safsizlik(sol_p, sol_n) + (sag_n / n) * self.safsizlik(pos_tot - sol_p, sag_n)),
                -1.0,
            )
            b = int(np.argmax(kazanc))
            if kazanc[b] > best[0]:
                best = (float(kazanc[b]), i, b)

        if best[1] is None:
            return None
        return best[1], best[2], float(esik_listesi[best[1]][best[2]]), best[0]

    def dugum_olustur(self, X_bin, y, esik_listesi, seviye=0):
        n = len(y)
        pos = int(np.sum(y))
        olasilik = pos / n
        yaprak = {"yaprak": True, "tahmin": 1 if olasilik >= 0.5 else 0, "olasilik": olasilik, "kayit": n}

        if seviye >= self.derinlik or n < self.min_bolunme or pos == 0 or pos == n:
            return yaprak
        bol = self.en_iyi_bolunme(X_bin, y, esik_listesi)
        if bol is None:
            return yaprak
        oz, b_idx, esik, kazanc = bol
        sol_m = X_bin[:, oz] <= b_idx
        if np.all(sol_m) or not np.any(sol_m):
            return yaprak

        return {
            "yaprak": False, "ozellik": oz, "esik": esik, "kazanc": kazanc,
            "tahmin": yaprak["tahmin"], "olasilik": olasilik, "kayit": n,
            "sol": self.dugum_olustur(X_bin[sol_m], y[sol_m], esik_listesi, seviye + 1),
            "sag": self.dugum_olustur(X_bin[~sol_m], y[~sol_m], esik_listesi, seviye + 1),
        }

    def fit(self, X, y, ozellik_isimleri, on_binlenmis=None):
        self.ozellik_isimleri = list(ozellik_isimleri)
        self.budanan_dugum_sayisi = 0
        X_bin, esik_listesi = self.binle(X, self.bin_sayisi) if on_binlenmis is None else on_binlenmis
        self.agac = self.dugum_olustur(X_bin, y, esik_listesi)
        return self

    def buda(self, X_dogrulama, y_dogrulama):
        def _buda(dugum, idx):
            if dugum["yaprak"] or len(idx) == 0:
                return int(np.sum(y_dogrulama[idx] != dugum["tahmin"]))
            sol_mu = X_dogrulama[idx, dugum["ozellik"]] <= dugum["esik"]
            alt_hata = _buda(dugum["sol"], idx[sol_mu]) + _buda(dugum["sag"], idx[~sol_mu])
            yaprak_hata = int(np.sum(y_dogrulama[idx] != dugum["tahmin"]))
            if yaprak_hata <= alt_hata:
                dugum["yaprak"] = True
                dugum.pop("sol", None)
                dugum.pop("sag", None)
                self.budanan_dugum_sayisi += 1
                return yaprak_hata
            return alt_hata
        _buda(self.agac, np.arange(len(y_dogrulama)))
        return self

    def _olasilik_doldur(self, dugum, X, idx, sonuc):
        if len(idx) == 0:
            return
        if dugum["yaprak"]:
            sonuc[idx] = dugum["olasilik"]
            return
        sol_mu = X[idx, dugum["ozellik"]] <= dugum["esik"]
        self._olasilik_doldur(dugum["sol"], X, idx[sol_mu], sonuc)
        self._olasilik_doldur(dugum["sag"], X, idx[~sol_mu], sonuc)

    def predict_proba(self, X):
        sonuc = np.zeros(len(X), dtype=float)
        self._olasilik_doldur(self.agac, X, np.arange(len(X)), sonuc)
        return sonuc

    def predict(self, X, karar_esigi=0.5):
        return (self.predict_proba(X) >= karar_esigi).astype(int)


def agaci_yaziya_cevir(dugum, ozellik_isimleri, seviye=0, dal_adi="Kok"):
    bosluk = "  " * seviye
    if dugum["yaprak"]:
        return [f"{bosluk}{dal_adi} -> Tahmin: {dugum['tahmin']} (olasilik: %{dugum['olasilik'] * 100:.1f}, kayit: {dugum['kayit']})"]
    satirlar = [f"{bosluk}{dal_adi}: {ozellik_isimleri[dugum['ozellik']]} <= {dugum['esik']:.3f} (kazanc: {dugum['kazanc']:.4f})"]
    satirlar.extend(agaci_yaziya_cevir(dugum["sol"], ozellik_isimleri, seviye + 1, "Evet"))
    satirlar.extend(agaci_yaziya_cevir(dugum["sag"], ozellik_isimleri, seviye + 1, "Hayir"))
    return satirlar


def hiperparametre_ara(X_egitim, y_egitim, X_dogrulama, y_dogrulama, ozellik_isimleri, hedef_metrik="accuracy", budama_uygula=True, bin_sayisi=32):
    on_bin = KararAgaci.binle(X_egitim, bin_sayisi)
    sonuclar, en_iyi_model, en_iyi_ayar, en_iyi_skor = [], None, None, -1.0

    for kriter in ("gini", "entropy"):
        for derinlik in (4, 6, 8, 10):
            for min_yaprak in (30, 60, 100):
                min_bol = max(min_yaprak * 2, 100)
                model = KararAgaci(derinlik, min_bol, min_yaprak, kriter=kriter, bin_sayisi=bin_sayisi)
                model.fit(X_egitim, y_egitim, ozellik_isimleri, on_binlenmis=on_bin)
                if budama_uygula:
                    model.buda(X_dogrulama, y_dogrulama)
                satir, met = _satir_olustur(
                    {"kriter": kriter, "derinlik": derinlik, "min_bolunme": min_bol, "min_yaprak": min_yaprak, "budanan_dugum": model.budanan_dugum_sayisi},
                    y_dogrulama, model.predict_proba(X_dogrulama),
                )
                sonuclar.append(satir)
                if met[hedef_metrik] > en_iyi_skor:
                    en_iyi_skor, en_iyi_model, en_iyi_ayar = met[hedef_metrik], model, satir

    return en_iyi_model, en_iyi_ayar, _sonuc_tablosu(sonuclar, hedef_metrik)


# =========================================================
# 2. LOJISTIK REGRESYON (SGD, MOMENTUM, RMSPROP, ADAM)
# =========================================================
class LojistikRegresyon:
    def __init__(self, ogrenme_orani=0.05, iterasyon=300, l2_lambda=1.0, optimizasyon="adam", tol=1e-5):
        if optimizasyon not in ("adam", "sgd", "momentum", "rmsprop"):
            raise ValueError(f"Gecersiz optimizasyon algoritmasi: {optimizasyon}")
        self.ogrenme_orani = ogrenme_orani
        self.iterasyon = iterasyon
        self.l2_lambda = l2_lambda
        self.optimizasyon = optimizasyon
        self.tol = tol
        self.agirliklar = None
        self.bias = 0.0
        self.ortalama = None
        self.std = None
        self.kayip_gecmisi = []
        self.ozellik_isimleri = []
        self.tamamlanan_iterasyon = 0

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))

    @staticmethod
    def olcekleme_istatistikleri(X):
        return z_olcekle(X)

    def fit(self, X, y, ozellik_isimleri, on_olceklenmis=None):
        self.ozellik_isimleri = list(ozellik_isimleri)
        X_s, self.ortalama, self.std = z_olcekle(X) if on_olceklenmis is None else on_olceklenmis
        n, d = X_s.shape
        self.agirliklar, self.bias = np.zeros(d, dtype=float), 0.0
        self.kayip_gecmisi, self.tamamlanan_iterasyon = [], 0

        mw, vw = np.zeros(d, dtype=float), np.zeros(d, dtype=float)
        mb, vb = 0.0, 0.0
        b1, b2, eps, lr = 0.9, 0.999, 1e-8, self.ogrenme_orani

        for t in range(1, self.iterasyon + 1):
            p = self.sigmoid(X_s @ self.agirliklar + self.bias)
            hata = p - y
            dw = (X_s.T @ hata) / n + (self.l2_lambda / n) * self.agirliklar
            db = float(np.mean(hata))

            if self.optimizasyon == "sgd":
                self.agirliklar -= lr * dw
                self.bias -= lr * db
            elif self.optimizasyon == "momentum":
                mw, mb = b1 * mw + dw, b1 * mb + db
                self.agirliklar -= lr * mw
                self.bias -= lr * mb
            elif self.optimizasyon == "rmsprop":
                vw, vb = b1 * vw + (1.0 - b1) * (dw ** 2), b1 * vb + (1.0 - b1) * (db ** 2)
                self.agirliklar -= lr * dw / (np.sqrt(vw) + eps)
                self.bias -= lr * db / (np.sqrt(vb) + eps)
            else:
                mw, vw = b1 * mw + (1.0 - b1) * dw, b2 * vw + (1.0 - b2) * (dw ** 2)
                mb, vb = b1 * mb + (1.0 - b1) * db, b2 * vb + (1.0 - b2) * (db ** 2)
                self.agirliklar -= lr * (mw / (1.0 - b1 ** t)) / (np.sqrt(vw / (1.0 - b2 ** t)) + eps)
                self.bias -= lr * (mb / (1.0 - b1 ** t)) / (np.sqrt(vb / (1.0 - b2 ** t)) + eps)

            self.tamamlanan_iterasyon = t
            if t % 10 == 0 or t == 1:
                pc = np.clip(p, 1e-12, 1.0 - 1e-12)
                kayip = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1.0 - pc)) + 0.5 * (self.l2_lambda / n) * np.sum(self.agirliklar ** 2))
                self.kayip_gecmisi.append((t, kayip))
                if len(self.kayip_gecmisi) >= 2 and abs(self.kayip_gecmisi[-2][1] - kayip) < self.tol:
                    break
        return self

    def predict_proba(self, X):
        return self.sigmoid(((X - self.ortalama) / self.std) @ self.agirliklar + self.bias)

    def predict(self, X, karar_esigi=0.5):
        return (self.predict_proba(X) >= karar_esigi).astype(int)

    def ozellik_agirliklari_df(self):
        return pd.DataFrame({
            "ozellik": self.ozellik_isimleri,
            "agirlik": self.agirliklar,
            "mutlak_etki": np.abs(self.agirliklar),
        }).sort_values(by="mutlak_etki", ascending=False).reset_index(drop=True)

    def kisi_acikla(self, X_tek):
        """Tek bir hasta icin ozellik bazli log-odds katkilarini (XAI) hesaplar."""
        x_s = (X_tek[0] - self.ortalama) / self.std
        katkilar = x_s * self.agirliklar
        return pd.DataFrame({
            "ozellik": self.ozellik_isimleri,
            "katki": katkilar,
            "mutlak_katki": np.abs(katkilar),
            "yon": np.where(katkilar >= 0, "Riski Artırıyor", "Riski Azaltıyor"),
        }).sort_values(by="mutlak_katki", ascending=False).reset_index(drop=True)


def hiperparametre_ara_lojistik(X_egitim, y_egitim, X_dogrulama, y_dogrulama, ozellik_isimleri, hedef_metrik="accuracy"):
    on_olcek = z_olcekle(X_egitim)
    sonuclar, en_iyi_model, en_iyi_ayar, en_iyi_skor = [], None, None, -1.0
    arama = [("sgd", (0.1, 0.5)), ("momentum", (0.1, 0.5)), ("rmsprop", (0.01, 0.05)), ("adam", (0.01, 0.05, 0.1))]

    for opt, lr_list in arama:
        for lr in lr_list:
            for l2 in (0.0, 1.0, 10.0):
                model = LojistikRegresyon(lr, 300, l2, optimizasyon=opt).fit(X_egitim, y_egitim, ozellik_isimleri, on_olceklenmis=on_olcek)
                satir, met = _satir_olustur(
                    {"optimizasyon": opt, "ogrenme_orani": lr, "l2_lambda": l2, "tamamlanan_iterasyon": model.tamamlanan_iterasyon},
                    y_dogrulama, model.predict_proba(X_dogrulama),
                )
                sonuclar.append(satir)
                if met[hedef_metrik] > en_iyi_skor:
                    en_iyi_skor, en_iyi_model, en_iyi_ayar = met[hedef_metrik], model, satir

    return en_iyi_model, en_iyi_ayar, _sonuc_tablosu(sonuclar, hedef_metrik)


# =========================================================
# 3. NAIVE BAYES (GAUSSIAN, HIBRIT, HISTOGRAM)
# =========================================================
class NaiveBayes:
    def __init__(self, yontem="hibrit", var_smoothing=1e-6, alpha=1.0, bin_sayisi=16):
        if yontem not in ("gaussian", "hibrit", "histogram"):
            raise ValueError(f"Gecersiz Naive Bayes yontemi: {yontem}")
        self.yontem = yontem
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.bin_sayisi = bin_sayisi
        self.ozellik_isimleri = []
        self.log_prior = None
        self.cont_idx = np.array([], dtype=int)
        self.bin_idx = np.array([], dtype=int)
        self.means = self.vars = self.bin_probs = None
        self.esik_listesi, self.log_bin_probs = [], []

    def fit(self, X, y, ozellik_isimleri, on_binlenmis=None):
        self.ozellik_isimleri = list(ozellik_isimleri)
        n, d = X.shape
        self.log_prior = np.array([np.log(max(float(np.mean(y == 0)), 1e-12)), np.log(max(float(np.mean(y == 1)), 1e-12))])

        if self.yontem in ("gaussian", "hibrit"):
            bin_mask = np.zeros(d, dtype=bool) if self.yontem == "gaussian" else np.array([set(np.unique(X[:, i])).issubset({0.0, 1.0}) for i in range(d)])
            self.cont_idx, self.bin_idx = np.where(~bin_mask)[0], np.where(bin_mask)[0]
            eps = self.var_smoothing * max(float(np.max(np.var(X, axis=0))) if n > 0 else 1.0, 1e-12)
            self.means = np.zeros((2, len(self.cont_idx)), dtype=float)
            self.vars = np.ones((2, len(self.cont_idx)), dtype=float)
            self.bin_probs = np.full((2, len(self.bin_idx)), 0.5, dtype=float)
            for c in (0, 1):
                Xc = X[y == c]
                if len(self.cont_idx):
                    self.means[c] = np.mean(Xc[:, self.cont_idx], axis=0)
                    self.vars[c] = np.var(Xc[:, self.cont_idx], axis=0) + eps
                if len(self.bin_idx):
                    self.bin_probs[c] = (np.sum(Xc[:, self.bin_idx], axis=0) + self.alpha) / (len(Xc) + 2.0 * self.alpha)
        else:
            X_bin, self.esik_listesi = KararAgaci.binle(X, self.bin_sayisi) if on_binlenmis is None else on_binlenmis
            self.log_bin_probs = []
            for i, esikler in enumerate(self.esik_listesi):
                k = len(esikler) + 1
                lp = np.zeros((2, k), dtype=float)
                for c in (0, 1):
                    cnt = np.bincount(X_bin[y == c, i], minlength=k)
                    lp[c] = np.log((cnt + self.alpha) / (np.sum(cnt) + k * self.alpha))
                self.log_bin_probs.append(lp)
        return self

    def predict_proba(self, X):
        log_post = np.tile(self.log_prior, (len(X), 1))
        if self.yontem in ("gaussian", "hibrit"):
            if len(self.cont_idx):
                Xc = X[:, self.cont_idx]
                for c in (0, 1):
                    v, m = self.vars[c], self.means[c]
                    log_post[:, c] += -0.5 * np.sum(np.log(2.0 * np.pi * v) + ((Xc - m) ** 2) / v, axis=1)
            if len(self.bin_idx):
                Xb = X[:, self.bin_idx]
                for c in (0, 1):
                    p = np.clip(self.bin_probs[c], 1e-12, 1.0 - 1e-12)
                    log_post[:, c] += Xb @ np.log(p) + (1.0 - Xb) @ np.log(1.0 - p)
        else:
            for i, esikler in enumerate(self.esik_listesi):
                b = np.searchsorted(esikler, X[:, i], side="right")
                lp = self.log_bin_probs[i]
                log_post[:, 0] += lp[0, b]
                log_post[:, 1] += lp[1, b]
        exp_lp = np.exp(log_post - np.max(log_post, axis=1, keepdims=True))
        return exp_lp[:, 1] / np.sum(exp_lp, axis=1)

    def predict(self, X, karar_esigi=0.5):
        return (self.predict_proba(X) >= karar_esigi).astype(int)

    def ozellik_ayirt_edicilik_df(self):
        satirlar = []
        if self.yontem in ("gaussian", "hibrit"):
            for pos, fi in enumerate(self.cont_idx):
                s = max(float(np.sqrt(0.5 * (self.vars[0, pos] + self.vars[1, pos]))), 1e-12)
                satirlar.append({"ozellik": self.ozellik_isimleri[fi], "tip": "Gaussian", "ayirt_edicilik": float(abs(self.means[1, pos] - self.means[0, pos]) / s)})
            for pos, fi in enumerate(self.bin_idx):
                satirlar.append({"ozellik": self.ozellik_isimleri[fi], "tip": "Bernoulli", "ayirt_edicilik": float(abs(self.bin_probs[1, pos] - self.bin_probs[0, pos]))})
        else:
            for fi, lp in enumerate(self.log_bin_probs):
                satirlar.append({"ozellik": self.ozellik_isimleri[fi], "tip": "Histogram", "ayirt_edicilik": 0.5 * float(np.sum(np.abs(np.exp(lp[1]) - np.exp(lp[0]))))})
        return pd.DataFrame(satirlar).sort_values(by="ayirt_edicilik", ascending=False).reset_index(drop=True)


def hiperparametre_ara_naive_bayes(X_egitim, y_egitim, X_dogrulama, y_dogrulama, ozellik_isimleri, hedef_metrik="accuracy"):
    sonuclar, en_iyi_model, en_iyi_ayar, en_iyi_skor = [], None, None, -1.0
    adaylar = [("gaussian", vs, 1.0, 16, None) for vs in (1e-9, 1e-6, 1e-3, 1e-1)]
    adaylar += [("hibrit", vs, a, 16, None) for vs in (1e-9, 1e-6, 1e-3) for a in (0.1, 0.5, 1.0, 2.0)]
    for bs in (8, 16):
        on_bin = KararAgaci.binle(X_egitim, bs)
        adaylar += [("histogram", 1e-6, a, bs, on_bin) for a in (0.1, 0.5, 1.0, 2.0)]

    for yontem, vs, alpha, bs, on_bin in adaylar:
        model = NaiveBayes(yontem, vs, alpha, bs).fit(X_egitim, y_egitim, ozellik_isimleri, on_binlenmis=on_bin)
        satir, met = _satir_olustur(
            {"yontem": yontem, "var_smoothing": vs, "alpha": alpha, "bin_sayisi": bs if yontem == "histogram" else None},
            y_dogrulama, model.predict_proba(X_dogrulama),
        )
        sonuclar.append(satir)
        if met[hedef_metrik] > en_iyi_skor:
            en_iyi_skor, en_iyi_model, en_iyi_ayar = met[hedef_metrik], model, satir

    return en_iyi_model, en_iyi_ayar, _sonuc_tablosu(sonuclar, hedef_metrik)


# =========================================================
# 4. K-EN YAKIN KOMSU (KNN)
# =========================================================
class KNN:
    def __init__(self, k=25, mesafe="kosinus", agirliklandirma="mesafe", ozellik_agirlikli=True, maks_referans=15000, batch_size=1024):
        if mesafe not in ("oklid", "kosinus"):
            raise ValueError(f"Gecersiz mesafe metrigi: {mesafe}")
        if agirliklandirma not in ("esit", "mesafe"):
            raise ValueError(f"Gecersiz agirliklandirma: {agirliklandirma}")
        self.k, self.mesafe, self.agirliklandirma = int(k), mesafe, agirliklandirma
        self.ozellik_agirlikli = ozellik_agirlikli
        self.maks_referans, self.batch_size = int(maks_referans), int(batch_size)
        self.ozellik_isimleri = []
        self.ortalama = self.std = self.ozellik_agirliklari = None
        self.X_ref_s = self.y_ref = self.ref_sq = self.X_ref_norm = None

    def fit(self, X, y, ozellik_isimleri):
        self.ozellik_isimleri = list(ozellik_isimleri)
        n_ref = min(len(X), self.maks_referans)
        X_sub, y_sub = X[:n_ref], y[:n_ref]
        X_s_f64, self.ortalama, self.std = z_olcekle(X_sub)
        X_s = X_s_f64.astype(np.float32)

        if self.ozellik_agirlikli:
            yc = (y_sub - np.mean(y_sub)).astype(np.float32)
            corr = np.abs((X_s.T @ yc) / (len(y_sub) * (np.std(yc) + 1e-8)))
            self.ozellik_agirliklari = np.sqrt(np.maximum(corr, 1e-4)).astype(np.float32)
        else:
            self.ozellik_agirliklari = np.ones(X_s.shape[1], dtype=np.float32)

        self.X_ref_s = X_s * self.ozellik_agirliklari
        self.y_ref = np.asarray(y_sub, dtype=np.int8)
        if self.mesafe == "kosinus":
            self.X_ref_norm = self.X_ref_s / (np.linalg.norm(self.X_ref_s, axis=1, keepdims=True) + 1e-8)
            self.ref_sq = None
        else:
            self.ref_sq = np.sum(self.X_ref_s ** 2, axis=1)
            self.X_ref_norm = None
        return self

    def _en_yakin_k_bul(self, X, k_max):
        X_q = ((X - self.ortalama) / self.std).astype(np.float32) * self.ozellik_agirliklari
        n_q, n_ref = len(X_q), len(self.X_ref_s)
        k_eff = min(k_max, n_ref)
        top_d = np.empty((n_q, k_eff), dtype=np.float32)
        top_y = np.empty((n_q, k_eff), dtype=np.int8)
        X_q_norm = X_q / (np.linalg.norm(X_q, axis=1, keepdims=True) + 1e-8) if self.mesafe == "kosinus" else None

        for s in range(0, n_q, self.batch_size):
            e = min(s + self.batch_size, n_q)
            if self.mesafe == "kosinus":
                dists = 1.0 - (X_q_norm[s:e] @ self.X_ref_norm.T)
            else:
                qb = X_q[s:e]
                dists = np.maximum(np.sum(qb ** 2, axis=1, keepdims=True) + self.ref_sq[None, :] - 2.0 * (qb @ self.X_ref_s.T), 0.0)
            idx = np.argpartition(dists, k_eff, axis=1)[:, :k_eff] if k_eff < n_ref else np.tile(np.arange(n_ref), (e - s, 1))
            r_idx = np.arange(e - s)[:, None]
            bd = dists[r_idx, idx]
            ord_k = np.argsort(bd, axis=1)
            sorted_bd = bd[r_idx, ord_k]
            top_d[s:e] = np.maximum(sorted_bd, 0.0) if self.mesafe == "kosinus" else np.sqrt(sorted_bd)
            top_y[s:e] = self.y_ref[idx[r_idx, ord_k]]
        return top_d, top_y

    @staticmethod
    def _oyla(top_d, top_y, agirliklandirma):
        if agirliklandirma == "esit":
            return np.mean(top_y, axis=1, dtype=float)
        w = 1.0 / (top_d.astype(float) + 1e-6)
        return np.sum(w * top_y.astype(float), axis=1) / np.sum(w, axis=1)

    def predict_proba(self, X):
        top_d, top_y = self._en_yakin_k_bul(X, self.k)
        return self._oyla(top_d, top_y, self.agirliklandirma)

    def predict(self, X, karar_esigi=0.5):
        return (self.predict_proba(X) >= karar_esigi).astype(int)

    def ozellik_agirliklari_df(self):
        return pd.DataFrame({
            "ozellik": self.ozellik_isimleri,
            "agirlik": self.ozellik_agirliklari.astype(float),
        }).sort_values(by="agirlik", ascending=False).reset_index(drop=True)


def hiperparametre_ara_knn(X_egitim, y_egitim, X_dogrulama, y_dogrulama, ozellik_isimleri, hedef_metrik="accuracy", maks_referans=15000):
    k_adaylari = (7, 15, 25, 35)
    k_max = max(k_adaylari)
    sonuclar, en_iyi_ayar, en_iyi_skor = [], None, -1.0

    for mesafe in ("oklid", "kosinus"):
        for oz_ag in (False, True):
            baz = KNN(k_max, mesafe, "mesafe", oz_ag, maks_referans).fit(X_egitim, y_egitim, ozellik_isimleri)
            top_d, top_y = baz._en_yakin_k_bul(X_dogrulama, k_max)
            for k in k_adaylari:
                for ag in ("esit", "mesafe"):
                    satir, met = _satir_olustur(
                        {"mesafe": mesafe, "ozellik_agirlikli": oz_ag, "k": k, "agirliklandirma": ag, "maks_referans": len(baz.X_ref_s)},
                        y_dogrulama, KNN._oyla(top_d[:, :k], top_y[:, :k], ag),
                    )
                    sonuclar.append(satir)
                    if met[hedef_metrik] > en_iyi_skor:
                        en_iyi_skor, en_iyi_ayar = met[hedef_metrik], satir

    en_iyi_model = KNN(
        en_iyi_ayar["k"], en_iyi_ayar["mesafe"], en_iyi_ayar["agirliklandirma"],
        en_iyi_ayar["ozellik_agirlikli"], maks_referans,
    ).fit(X_egitim, y_egitim, ozellik_isimleri)
    return en_iyi_model, en_iyi_ayar, _sonuc_tablosu(sonuclar, hedef_metrik)


# =========================================================
# 5. K-MEANS KUMELEME (DENETIMSIZ NEGATIF REFERANS)
# =========================================================
class KMeansSiniflandirici:
    """Etiketsiz ham Oklid kumeleme ile bilerek zayif negatif referans modeli."""

    def __init__(self, k=2, maks_iter=25, baslatma="rastgele", tohum=42):
        if baslatma not in ("rastgele", "kmeans++"):
            raise ValueError(f"Gecersiz baslatma yontemi: {baslatma}")
        self.k = int(k)
        self.maks_iter = int(maks_iter)
        self.baslatma = baslatma
        self.tohum = int(tohum)
        self.ozellik_isimleri = []
        self.merkezler = None
        self.kume_olasiliklari = None
        self.kume_boyutlari = None
        self.tamamlanan_iterasyon = 0
        self.inertia = 0.0

    @staticmethod
    def _mesafe_kare(X, merkezler):
        return np.maximum(
            np.sum(X ** 2, axis=1, keepdims=True)
            + np.sum(merkezler ** 2, axis=1)[None, :]
            - 2.0 * (X @ merkezler.T),
            0.0,
        )

    def _merkez_baslat(self, X, rng):
        n = len(X)
        if self.baslatma == "rastgele":
            return X[rng.choice(n, self.k, replace=False)].copy()

        alt_n = min(n, 15000)
        X_alt = X[rng.choice(n, alt_n, replace=False)] if n > alt_n else X
        merkezler = [X_alt[int(rng.integers(0, len(X_alt)))].copy()]
        en_yakin_d2 = np.sum((X_alt - merkezler[0]) ** 2, axis=1)
        for _ in range(1, self.k):
            toplam = float(np.sum(en_yakin_d2))
            if toplam <= 1e-12:
                sec = int(rng.integers(0, len(X_alt)))
            else:
                sec = int(rng.choice(len(X_alt), p=en_yakin_d2 / toplam))
            yeni_m = X_alt[sec].copy()
            merkezler.append(yeni_m)
            d2_yeni = np.sum((X_alt - yeni_m) ** 2, axis=1)
            en_yakin_d2 = np.minimum(en_yakin_d2, d2_yeni)
        return np.array(merkezler, dtype=float)

    def fit(self, X, y, ozellik_isimleri):
        self.ozellik_isimleri = list(ozellik_isimleri)
        rng = np.random.default_rng(self.tohum)
        merkezler = self._merkez_baslat(X, rng)
        onceki_etiketler = None
        self.tamamlanan_iterasyon = 0

        for it in range(1, self.maks_iter + 1):
            etiketler = np.argmin(self._mesafe_kare(X, merkezler), axis=1)
            self.tamamlanan_iterasyon = it
            if onceki_etiketler is not None and np.array_equal(etiketler, onceki_etiketler):
                break
            onceki_etiketler = etiketler
            merkezler = np.array(
                [X[etiketler == c].mean(axis=0) if np.any(etiketler == c) else merkezler[c] for c in range(self.k)],
                dtype=float,
            )

        self.merkezler = merkezler
        d2 = self._mesafe_kare(X, self.merkezler)
        etiketler = np.argmin(d2, axis=1)
        self.inertia = float(np.sum(d2[np.arange(len(X)), etiketler]))

        varsayilan_p = float(np.mean(y))
        self.kume_olasiliklari = np.array(
            [float(np.mean(y[etiketler == c])) if np.any(etiketler == c) else varsayilan_p for c in range(self.k)],
            dtype=float,
        )
        self.kume_boyutlari = np.array([int(np.sum(etiketler == c)) for c in range(self.k)], dtype=int)
        return self

    def predict_proba(self, X):
        en_yakin = np.argmin(self._mesafe_kare(X, self.merkezler), axis=1)
        return self.kume_olasiliklari[en_yakin]

    def predict(self, X, karar_esigi=0.5):
        return (self.predict_proba(X) >= karar_esigi).astype(int)

    def kume_ozeti_df(self):
        return pd.DataFrame({
            "kume": np.arange(1, self.k + 1),
            "kayit_sayisi": self.kume_boyutlari,
            "hasta_orani": np.round(self.kume_olasiliklari, 4),
            "atanan_sinif": (self.kume_olasiliklari >= 0.5).astype(int),
        })


def hiperparametre_ara_kmeans(X_egitim, y_egitim, X_dogrulama, y_dogrulama, ozellik_isimleri, hedef_metrik="accuracy"):
    sonuclar, en_iyi_model, en_iyi_ayar, en_iyi_skor = [], None, None, -1.0

    for k in (2, 3, 4, 6):
        for baslatma in ("rastgele", "kmeans++"):
            for maks_iter in (15, 30):
                model = KMeansSiniflandirici(k=k, maks_iter=maks_iter, baslatma=baslatma).fit(
                    X_egitim, y_egitim, ozellik_isimleri
                )
                satir, met = _satir_olustur(
                    {"k": k, "baslatma": baslatma, "maks_iter": maks_iter, "tamamlanan_iterasyon": model.tamamlanan_iterasyon},
                    y_dogrulama, model.predict_proba(X_dogrulama),
                )
                sonuclar.append(satir)
                if met[hedef_metrik] > en_iyi_skor:
                    en_iyi_skor, en_iyi_model, en_iyi_ayar = met[hedef_metrik], model, satir

    return en_iyi_model, en_iyi_ayar, _sonuc_tablosu(sonuclar, hedef_metrik)
