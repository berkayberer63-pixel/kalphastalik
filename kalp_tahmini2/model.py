import numpy as np
import pandas as pd


# =========================================================
# KARAR AGACI: HAZIR ML MODELI KULLANILMIYOR
# =========================================================
class KararAgaci:
    def __init__(
        self,
        derinlik,
        min_bolunme,
        min_yaprak,
        kriter="gini",
        bin_sayisi=32
    ):
        self.derinlik = derinlik
        self.min_bolunme = min_bolunme
        self.min_yaprak = min_yaprak
        self.kriter = kriter
        self.bin_sayisi = bin_sayisi
        self.agac = None
        self.ozellik_isimleri = []
        self.budanan_dugum_sayisi = 0

    def safsizlik(self, pozitif, toplam):
        """Gini veya Entropy safsizligini vektorize olarak hesaplar."""
        p = pozitif / np.maximum(toplam, 1)
        if self.kriter == "entropy":
            p_clip = np.clip(p, 1e-12, 1.0 - 1e-12)
            return -(
                p_clip * np.log2(p_clip)
                + (1.0 - p_clip) * np.log2(1.0 - p_clip)
            )
        return 2.0 * p * (1.0 - p)

    @staticmethod
    def binle(X, bin_sayisi=32):
        """Ozellik sutunlarini bir kez histogram kutularina ayirir."""
        kayit_sayisi, ozellik_sayisi = X.shape
        X_bin = np.empty((kayit_sayisi, ozellik_sayisi), dtype=np.uint8)
        esik_listesi = []
        ceyrekler = np.linspace(0.0, 1.0, bin_sayisi + 1)[1:-1]

        for i in range(ozellik_sayisi):
            sutun = X[:, i]
            benzersiz = np.unique(sutun)
            if len(benzersiz) <= bin_sayisi:
                esikler = (benzersiz[:-1] + benzersiz[1:]) / 2.0
            else:
                esikler = np.unique(np.quantile(sutun, ceyrekler))
            X_bin[:, i] = np.searchsorted(
                esikler, sutun, side="right"
            ).astype(np.uint8)
            esik_listesi.append(esikler)

        return X_bin, esik_listesi

    def en_iyi_bolunme(self, X_bin, y, esik_listesi):
        """Histogram kutulari ve kumulatif toplamlarla en iyi bolunmeyi bulur."""
        kayit_sayisi = len(y)
        toplam_pozitif = int(np.sum(y))
        ana_safsizlik = float(self.safsizlik(toplam_pozitif, kayit_sayisi))
        pozitif_maskesi = y == 1

        en_iyi_kazanc = 0.0
        en_iyi_ozellik = None
        en_iyi_bin = None

        for i, esikler in enumerate(esik_listesi):
            kutu_sayisi = len(esikler) + 1
            if kutu_sayisi <= 1:
                continue

            sutun_bin = X_bin[:, i]
            toplam_kutu = np.bincount(sutun_bin, minlength=kutu_sayisi)
            pozitif_kutu = np.bincount(
                sutun_bin[pozitif_maskesi], minlength=kutu_sayisi
            )

            sol_sayisi = np.cumsum(toplam_kutu)[:-1]
            sag_sayisi = kayit_sayisi - sol_sayisi
            gecerli = (sol_sayisi >= self.min_yaprak) & (
                sag_sayisi >= self.min_yaprak
            )
            if not np.any(gecerli):
                continue

            sol_pozitif = np.cumsum(pozitif_kutu)[:-1]
            sag_pozitif = toplam_pozitif - sol_pozitif

            kazanclar = ana_safsizlik - (
                (sol_sayisi / kayit_sayisi)
                * self.safsizlik(sol_pozitif, sol_sayisi)
                + (sag_sayisi / kayit_sayisi)
                * self.safsizlik(sag_pozitif, sag_sayisi)
            )
            kazanclar = np.where(gecerli, kazanclar, -1.0)
            b_idx = int(np.argmax(kazanclar))

            if kazanclar[b_idx] > en_iyi_kazanc:
                en_iyi_kazanc = float(kazanclar[b_idx])
                en_iyi_ozellik = i
                en_iyi_bin = b_idx

        if en_iyi_ozellik is None:
            return None

        return (
            en_iyi_ozellik,
            en_iyi_bin,
            float(esik_listesi[en_iyi_ozellik][en_iyi_bin]),
            en_iyi_kazanc,
        )

    def dugum_olustur(self, X_bin, y, esik_listesi, seviye=0):
        kayit_sayisi = len(y)
        pozitif_sayisi = int(np.sum(y))
        olasilik = pozitif_sayisi / kayit_sayisi
        tahmin = 1 if olasilik >= 0.5 else 0

        yaprak = {
            "yaprak": True,
            "tahmin": tahmin,
            "olasilik": olasilik,
            "kayit": kayit_sayisi,
        }

        if (
            seviye >= self.derinlik
            or kayit_sayisi < self.min_bolunme
            or pozitif_sayisi == 0
            or pozitif_sayisi == kayit_sayisi
        ):
            return yaprak

        bolunme = self.en_iyi_bolunme(X_bin, y, esik_listesi)
        if bolunme is None:
            return yaprak

        ozellik, bin_indeksi, esik, kazanc = bolunme
        sol_maskesi = X_bin[:, ozellik] <= bin_indeksi

        if np.all(sol_maskesi) or not np.any(sol_maskesi):
            return yaprak

        return {
            "yaprak": False,
            "ozellik": ozellik,
            "esik": esik,
            "kazanc": kazanc,
            "tahmin": tahmin,
            "olasilik": olasilik,
            "kayit": kayit_sayisi,
            "sol": self.dugum_olustur(
                X_bin[sol_maskesi], y[sol_maskesi], esik_listesi, seviye + 1
            ),
            "sag": self.dugum_olustur(
                X_bin[~sol_maskesi], y[~sol_maskesi], esik_listesi, seviye + 1
            ),
        }

    def fit(self, X, y, ozellik_isimleri, on_binlenmis=None):
        self.ozellik_isimleri = list(ozellik_isimleri)
        self.budanan_dugum_sayisi = 0
        if on_binlenmis is None:
            X_bin, esik_listesi = self.binle(X, self.bin_sayisi)
        else:
            X_bin, esik_listesi = on_binlenmis
        self.agac = self.dugum_olustur(X_bin, y, esik_listesi)
        return self

    def buda(self, X_dogrulama, y_dogrulama):
        """Dogrulama seti uzerinde hata azaltmali budama (Reduced Error Pruning) uygular."""
        def _buda(dugum, indeksler):
            if dugum["yaprak"] or len(indeksler) == 0:
                return int(np.sum(y_dogrulama[indeksler] != dugum["tahmin"]))

            sol_mu = X_dogrulama[indeksler, dugum["ozellik"]] <= dugum["esik"]
            alt_hata = _buda(dugum["sol"], indeksler[sol_mu]) + _buda(
                dugum["sag"], indeksler[~sol_mu]
            )
            yaprak_hata = int(
                np.sum(y_dogrulama[indeksler] != dugum["tahmin"])
            )

            if yaprak_hata <= alt_hata:
                dugum["yaprak"] = True
                dugum.pop("sol", None)
                dugum.pop("sag", None)
                self.budanan_dugum_sayisi += 1
                return yaprak_hata
            return alt_hata

        _buda(self.agac, np.arange(len(y_dogrulama)))
        return self

    def olasilik_doldur(self, dugum, X, satir_indeksleri, sonuc):
        if len(satir_indeksleri) == 0:
            return

        if dugum["yaprak"]:
            sonuc[satir_indeksleri] = dugum["olasilik"]
            return

        sol_mu = X[satir_indeksleri, dugum["ozellik"]] <= dugum["esik"]
        self.olasilik_doldur(
            dugum["sol"], X, satir_indeksleri[sol_mu], sonuc
        )
        self.olasilik_doldur(
            dugum["sag"], X, satir_indeksleri[~sol_mu], sonuc
        )

    def predict_proba(self, X):
        sonuc = np.zeros(len(X), dtype=float)
        self.olasilik_doldur(self.agac, X, np.arange(len(X)), sonuc)
        return sonuc

    def predict(self, X, karar_esigi=0.5):
        return (self.predict_proba(X) >= karar_esigi).astype(int)


# =========================================================
# AGACI METIN OLARAK GOSTER
# =========================================================
def agaci_yaziya_cevir(dugum, ozellik_isimleri, seviye=0, dal_adi="Kok"):
    satirlar = []
    bosluk = "  " * seviye

    if dugum["yaprak"]:
        satirlar.append(
            f"{bosluk}{dal_adi} -> Tahmin: {dugum['tahmin']} "
            f"(olasilik: %{dugum['olasilik'] * 100:.1f}, kayit: {dugum['kayit']})"
        )
        return satirlar

    ozellik = ozellik_isimleri[dugum["ozellik"]]
    satirlar.append(
        f"{bosluk}{dal_adi}: {ozellik} <= {dugum['esik']:.3f} "
        f"(kazanc: {dugum['kazanc']:.4f})"
    )
    satirlar.extend(
        agaci_yaziya_cevir(
            dugum["sol"], ozellik_isimleri, seviye + 1, "Evet"
        )
    )
    satirlar.extend(
        agaci_yaziya_cevir(
            dugum["sag"], ozellik_isimleri, seviye + 1, "Hayir"
        )
    )
    return satirlar


# =========================================================
# BASARI METRIKLERINI HESAPLA VE OPTIMIZASYON YARDIMCILARI
# =========================================================
def roc_auc_hesapla(gercek, olasiliklar):
    """Saf NumPy ile sira tabanli (Mann-Whitney U) ROC-AUC hesaplar."""
    gercek = np.asarray(gercek, dtype=int)
    olasiliklar = np.asarray(olasiliklar, dtype=float)
    pozitif_sayisi = int(np.sum(gercek == 1))
    negatif_sayisi = len(gercek) - pozitif_sayisi
    if pozitif_sayisi == 0 or negatif_sayisi == 0:
        return 0.0

    sirali_indeks = np.argsort(olasiliklar)
    sirali_p = olasiliklar[sirali_indeks]
    sirali_y = gercek[sirali_indeks]
    siralar = np.arange(1, len(olasiliklar) + 1, dtype=float)

    farkli = np.diff(sirali_p) != 0
    baslangiclar = np.concatenate(([0], np.where(farkli)[0] + 1))
    bitisler = np.concatenate((np.where(farkli)[0] + 1, [len(olasiliklar)]))
    for bas, bit in zip(baslangiclar, bitisler):
        if bit - bas > 1:
            siralar[bas:bit] = 0.5 * (bas + 1 + bit)

    pozitif_sira_toplami = float(np.sum(siralar[sirali_y == 1]))
    return float(
        (pozitif_sira_toplami - pozitif_sayisi * (pozitif_sayisi + 1) / 2.0)
        / (pozitif_sayisi * negatif_sayisi)
    )


def metrikleri_hesapla(gercek, tahmin, olasiliklar=None):
    tn = int(np.sum((gercek == 0) & (tahmin == 0)))
    fp = int(np.sum((gercek == 0) & (tahmin == 1)))
    fn = int(np.sum((gercek == 1) & (tahmin == 0)))
    tp = int(np.sum((gercek == 1) & (tahmin == 1)))

    accuracy = (tp + tn) / len(gercek) if len(gercek) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    roc_auc = None
    log_loss = None
    if olasiliklar is not None:
        olasiliklar_f64 = np.asarray(olasiliklar, dtype=float)
        roc_auc = roc_auc_hesapla(gercek, olasiliklar_f64)
        p_clip = np.clip(olasiliklar_f64, 1e-12, 1.0 - 1e-12)
        log_loss = float(
            -np.mean(gercek * np.log(p_clip) + (1 - gercek) * np.log(1.0 - p_clip))
        )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": log_loss,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def en_iyi_esik_bul(y_dogrulama, olasiliklar, hedef_metrik="f1"):
    """Dogrulama olasiliklari uzerinde secilen metrigi maksimize eden esigi bulur."""
    en_iyi_esik = 0.5
    en_iyi_skor = -1.0

    for esik in np.linspace(0.25, 0.75, 51):
        tahmin = (olasiliklar >= esik).astype(int)
        metrik = metrikleri_hesapla(y_dogrulama, tahmin)
        skor = metrik[hedef_metrik]
        if skor > en_iyi_skor:
            en_iyi_skor = skor
            en_iyi_esik = float(esik)

    return en_iyi_esik, en_iyi_skor


def hiperparametre_ara(
    X_egitim,
    y_egitim,
    X_dogrulama,
    y_dogrulama,
    ozellik_isimleri,
    hedef_metrik="accuracy",
    budama_uygula=True,
    bin_sayisi=32
):
    """On-binlenmis egitim verisiyle hizli Grid Search uygular."""
    on_binlenmis = KararAgaci.binle(X_egitim, bin_sayisi)
    sonuclar = []
    en_iyi_model = None
    en_iyi_ayar = None
    en_iyi_skor = -1.0

    for kriter in ["gini", "entropy"]:
        for derinlik in [4, 6, 8, 10]:
            for min_yaprak in [30, 60, 100]:
                min_bolunme = max(min_yaprak * 2, 100)
                model = KararAgaci(
                    derinlik=derinlik,
                    min_bolunme=min_bolunme,
                    min_yaprak=min_yaprak,
                    kriter=kriter,
                    bin_sayisi=bin_sayisi,
                )
                model.fit(
                    X_egitim,
                    y_egitim,
                    ozellik_isimleri,
                    on_binlenmis=on_binlenmis,
                )
                if budama_uygula:
                    model.buda(X_dogrulama, y_dogrulama)

                val_olasilik = model.predict_proba(X_dogrulama)
                val_tahmin = (val_olasilik >= 0.5).astype(int)
                metrik = metrikleri_hesapla(
                    y_dogrulama, val_tahmin, olasiliklar=val_olasilik
                )
                satir = {
                    "kriter": kriter,
                    "derinlik": derinlik,
                    "min_bolunme": min_bolunme,
                    "min_yaprak": min_yaprak,
                    "budanan_dugum": model.budanan_dugum_sayisi,
                    "accuracy": metrik["accuracy"],
                    "precision": metrik["precision"],
                    "recall": metrik["recall"],
                    "specificity": metrik["specificity"],
                    "f1": metrik["f1"],
                    "roc_auc": metrik["roc_auc"],
                }
                sonuclar.append(satir)

                if metrik[hedef_metrik] > en_iyi_skor:
                    en_iyi_skor = metrik[hedef_metrik]
                    en_iyi_model = model
                    en_iyi_ayar = satir

    sonuclar_df = pd.DataFrame(sonuclar).sort_values(
        by=hedef_metrik, ascending=False
    ).reset_index(drop=True)
    return en_iyi_model, en_iyi_ayar, sonuclar_df


# =========================================================
# LOJISTIK REGRESYON: HAZIR ML MODELI KULLANILMIYOR
# =========================================================
class LojistikRegresyon:
    def __init__(
        self,
        ogrenme_orani=0.05,
        iterasyon=300,
        l2_lambda=1.0,
        optimizasyon="adam",
        tol=1e-5
    ):
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
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def olcekleme_istatistikleri(X):
        ortalama = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        X_s = (X - ortalama) / std
        return X_s, ortalama, std

    def _olcekle(self, X):
        return (X - self.ortalama) / self.std

    def fit(self, X, y, ozellik_isimleri, on_olceklenmis=None):
        self.ozellik_isimleri = list(ozellik_isimleri)
        if on_olceklenmis is None:
            X_s, self.ortalama, self.std = self.olcekleme_istatistikleri(X)
        else:
            X_s, self.ortalama, self.std = on_olceklenmis

        n, d = X_s.shape
        self.agirliklar = np.zeros(d, dtype=float)
        self.bias = 0.0
        self.kayip_gecmisi = []
        self.tamamlanan_iterasyon = 0

        m_w = np.zeros(d, dtype=float)
        v_w = np.zeros(d, dtype=float)
        m_b = 0.0
        v_b = 0.0
        b1, b2, eps = 0.9, 0.999, 1e-8

        for t in range(1, self.iterasyon + 1):
            z = X_s @ self.agirliklar + self.bias
            p = self.sigmoid(z)
            hata = p - y

            dw = (X_s.T @ hata) / n + (self.l2_lambda / n) * self.agirliklar
            db = float(np.mean(hata))

            if self.optimizasyon == "sgd":
                self.agirliklar -= self.ogrenme_orani * dw
                self.bias -= self.ogrenme_orani * db
            elif self.optimizasyon == "momentum":
                m_w = b1 * m_w + dw
                m_b = b1 * m_b + db
                self.agirliklar -= self.ogrenme_orani * m_w
                self.bias -= self.ogrenme_orani * m_b
            elif self.optimizasyon == "rmsprop":
                v_w = b1 * v_w + (1.0 - b1) * (dw ** 2)
                v_b = b1 * v_b + (1.0 - b1) * (db ** 2)
                self.agirliklar -= (
                    self.ogrenme_orani * dw / (np.sqrt(v_w) + eps)
                )
                self.bias -= (
                    self.ogrenme_orani * db / (np.sqrt(v_b) + eps)
                )
            else:
                m_w = b1 * m_w + (1.0 - b1) * dw
                v_w = b2 * v_w + (1.0 - b2) * (dw ** 2)
                m_b = b1 * m_b + (1.0 - b1) * db
                v_b = b2 * v_b + (1.0 - b2) * (db ** 2)

                m_w_hat = m_w / (1.0 - b1 ** t)
                v_w_hat = v_w / (1.0 - b2 ** t)
                m_b_hat = m_b / (1.0 - b1 ** t)
                v_b_hat = v_b / (1.0 - b2 ** t)

                self.agirliklar -= (
                    self.ogrenme_orani * m_w_hat / (np.sqrt(v_w_hat) + eps)
                )
                self.bias -= (
                    self.ogrenme_orani * m_b_hat / (np.sqrt(v_b_hat) + eps)
                )

            self.tamamlanan_iterasyon = t

            if t % 10 == 0 or t == 1:
                p_clip = np.clip(p, 1e-12, 1.0 - 1e-12)
                kayip = float(
                    -np.mean(
                        y * np.log(p_clip) + (1 - y) * np.log(1.0 - p_clip)
                    )
                    + 0.5
                    * (self.l2_lambda / n)
                    * np.sum(self.agirliklar ** 2)
                )
                self.kayip_gecmisi.append((t, kayip))
                if (
                    len(self.kayip_gecmisi) >= 2
                    and abs(self.kayip_gecmisi[-2][1] - kayip) < self.tol
                ):
                    break

        return self

    def predict_proba(self, X):
        X_s = self._olcekle(X)
        return self.sigmoid(X_s @ self.agirliklar + self.bias)

    def predict(self, X, karar_esigi=0.5):
        return (self.predict_proba(X) >= karar_esigi).astype(int)

    def ozellik_agirliklari_df(self):
        return pd.DataFrame({
            "ozellik": self.ozellik_isimleri,
            "agirlik": self.agirliklar,
            "mutlak_etki": np.abs(self.agirliklar),
        }).sort_values(by="mutlak_etki", ascending=False).reset_index(drop=True)


def hiperparametre_ara_lojistik(
    X_egitim,
    y_egitim,
    X_dogrulama,
    y_dogrulama,
    ozellik_isimleri,
    hedef_metrik="accuracy"
):
    """On-olceklenmis egitim verisiyle 4 optimizasyon algoritmasini (SGD, Momentum, RMSProp, Adam) tarar."""
    on_olceklenmis = LojistikRegresyon.olcekleme_istatistikleri(X_egitim)
    sonuclar = []
    en_iyi_model = None
    en_iyi_ayar = None
    en_iyi_skor = -1.0

    arama_uzayi = [
        ("sgd", [0.1, 0.5]),
        ("momentum", [0.1, 0.5]),
        ("rmsprop", [0.01, 0.05]),
        ("adam", [0.01, 0.05, 0.1]),
    ]

    for optimizasyon, lr_listesi in arama_uzayi:
        for ogrenme_orani in lr_listesi:
            for l2_lambda in [0.0, 1.0, 10.0]:
                model = LojistikRegresyon(
                    ogrenme_orani=ogrenme_orani,
                    iterasyon=300,
                    l2_lambda=l2_lambda,
                    optimizasyon=optimizasyon,
                )
                model.fit(
                    X_egitim,
                    y_egitim,
                    ozellik_isimleri,
                    on_olceklenmis=on_olceklenmis,
                )

                val_olasilik = model.predict_proba(X_dogrulama)
                val_tahmin = (val_olasilik >= 0.5).astype(int)
                metrik = metrikleri_hesapla(
                    y_dogrulama, val_tahmin, olasiliklar=val_olasilik
                )
                satir = {
                    "optimizasyon": optimizasyon,
                    "ogrenme_orani": ogrenme_orani,
                    "l2_lambda": l2_lambda,
                    "tamamlanan_iterasyon": model.tamamlanan_iterasyon,
                    "accuracy": metrik["accuracy"],
                    "precision": metrik["precision"],
                    "recall": metrik["recall"],
                    "specificity": metrik["specificity"],
                    "f1": metrik["f1"],
                    "roc_auc": metrik["roc_auc"],
                }
                sonuclar.append(satir)

                if metrik[hedef_metrik] > en_iyi_skor:
                    en_iyi_skor = metrik[hedef_metrik]
                    en_iyi_model = model
                    en_iyi_ayar = satir

    sonuclar_df = pd.DataFrame(sonuclar).sort_values(
        by=hedef_metrik, ascending=False
    ).reset_index(drop=True)
    return en_iyi_model, en_iyi_ayar, sonuclar_df


# =========================================================
# NAIVE BAYES: HAZIR ML MODELI KULLANILMIYOR
# =========================================================
class NaiveBayes:
    def __init__(
        self,
        yontem="hibrit",
        var_smoothing=1e-6,
        alpha=1.0,
        bin_sayisi=16
    ):
        if yontem not in ("gaussian", "hibrit", "histogram"):
            raise ValueError(f"Gecersiz Naive Bayes yontemi: {yontem}")
        self.yontem = yontem
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.bin_sayisi = bin_sayisi
        self.ozellik_isimleri = []
        self.log_prior = None
        self.binary_mask = None
        self.cont_idx = np.array([], dtype=int)
        self.bin_idx = np.array([], dtype=int)
        self.means = None
        self.vars = None
        self.bin_probs = None
        self.esik_listesi = []
        self.log_bin_probs = []

    def fit(self, X, y, ozellik_isimleri, on_binlenmis=None):
        self.ozellik_isimleri = list(ozellik_isimleri)
        n, d = X.shape
        self.log_prior = np.array([
            np.log(max(float(np.mean(y == 0)), 1e-12)),
            np.log(max(float(np.mean(y == 1)), 1e-12)),
        ])

        if self.yontem == "gaussian":
            self.binary_mask = np.zeros(d, dtype=bool)
        else:
            self.binary_mask = np.array([
                set(np.unique(X[:, i])).issubset({0.0, 1.0})
                for i in range(d)
            ])

        if self.yontem in ("gaussian", "hibrit"):
            self.cont_idx = np.where(~self.binary_mask)[0]
            self.bin_idx = np.where(self.binary_mask)[0]
            max_var = float(np.max(np.var(X, axis=0))) if n > 0 else 1.0
            eps = self.var_smoothing * max(max_var, 1e-12)

            self.means = np.zeros((2, len(self.cont_idx)), dtype=float)
            self.vars = np.ones((2, len(self.cont_idx)), dtype=float)
            self.bin_probs = np.full((2, len(self.bin_idx)), 0.5, dtype=float)

            for c in (0, 1):
                Xc = X[y == c]
                if len(self.cont_idx):
                    self.means[c] = np.mean(Xc[:, self.cont_idx], axis=0)
                    self.vars[c] = np.var(Xc[:, self.cont_idx], axis=0) + eps
                if len(self.bin_idx):
                    self.bin_probs[c] = (
                        np.sum(Xc[:, self.bin_idx], axis=0) + self.alpha
                    ) / (len(Xc) + 2.0 * self.alpha)
        else:
            if on_binlenmis is None:
                X_bin, self.esik_listesi = KararAgaci.binle(X, self.bin_sayisi)
            else:
                X_bin, self.esik_listesi = on_binlenmis

            self.log_bin_probs = []
            for i, esikler in enumerate(self.esik_listesi):
                k = len(esikler) + 1
                lp = np.zeros((2, k), dtype=float)
                for c in (0, 1):
                    cnt = np.bincount(X_bin[y == c, i], minlength=k)
                    lp[c] = np.log(
                        (cnt + self.alpha) / (np.sum(cnt) + k * self.alpha)
                    )
                self.log_bin_probs.append(lp)

        return self

    def predict_proba(self, X):
        n = len(X)
        log_post = np.tile(self.log_prior, (n, 1))

        if self.yontem in ("gaussian", "hibrit"):
            if len(self.cont_idx):
                Xc = X[:, self.cont_idx]
                for c in (0, 1):
                    v = self.vars[c]
                    m = self.means[c]
                    log_post[:, c] += -0.5 * np.sum(
                        np.log(2.0 * np.pi * v) + ((Xc - m) ** 2) / v,
                        axis=1,
                    )
            if len(self.bin_idx):
                Xb = X[:, self.bin_idx]
                for c in (0, 1):
                    p = np.clip(self.bin_probs[c], 1e-12, 1.0 - 1e-12)
                    log_post[:, c] += (
                        Xb @ np.log(p) + (1.0 - Xb) @ np.log(1.0 - p)
                    )
        else:
            for i, esikler in enumerate(self.esik_listesi):
                b = np.searchsorted(esikler, X[:, i], side="right")
                lp = self.log_bin_probs[i]
                log_post[:, 0] += lp[0, b]
                log_post[:, 1] += lp[1, b]

        max_lp = np.max(log_post, axis=1, keepdims=True)
        exp_lp = np.exp(log_post - max_lp)
        return exp_lp[:, 1] / np.sum(exp_lp, axis=1)

    def predict(self, X, karar_esigi=0.5):
        return (self.predict_proba(X) >= karar_esigi).astype(int)

    def ozellik_ayirt_edicilik_df(self):
        satirlar = []
        if self.yontem in ("gaussian", "hibrit"):
            for idx_pos, feat_idx in enumerate(self.cont_idx):
                ort_std = np.sqrt(
                    0.5 * (self.vars[0, idx_pos] + self.vars[1, idx_pos])
                )
                fark = (
                    abs(self.means[1, idx_pos] - self.means[0, idx_pos])
                    / max(float(ort_std), 1e-12)
                )
                satirlar.append({
                    "ozellik": self.ozellik_isimleri[feat_idx],
                    "tip": "Gaussian",
                    "ayirt_edicilik": float(fark),
                })
            for idx_pos, feat_idx in enumerate(self.bin_idx):
                fark = abs(
                    self.bin_probs[1, idx_pos] - self.bin_probs[0, idx_pos]
                )
                satirlar.append({
                    "ozellik": self.ozellik_isimleri[feat_idx],
                    "tip": "Bernoulli",
                    "ayirt_edicilik": float(fark),
                })
        else:
            for feat_idx, lp in enumerate(self.log_bin_probs):
                p0 = np.exp(lp[0])
                p1 = np.exp(lp[1])
                tv = 0.5 * float(np.sum(np.abs(p1 - p0)))
                satirlar.append({
                    "ozellik": self.ozellik_isimleri[feat_idx],
                    "tip": "Histogram",
                    "ayirt_edicilik": tv,
                })

        return pd.DataFrame(satirlar).sort_values(
            by="ayirt_edicilik", ascending=False
        ).reset_index(drop=True)


def hiperparametre_ara_naive_bayes(
    X_egitim,
    y_egitim,
    X_dogrulama,
    y_dogrulama,
    ozellik_isimleri,
    hedef_metrik="accuracy"
):
    """Gaussian, Hibrit ve Histogram Naive Bayes kombinasyonlarini tarar."""
    sonuclar = []
    en_iyi_model = None
    en_iyi_ayar = None
    en_iyi_skor = -1.0

    adaylar = []
    for vs in [1e-9, 1e-6, 1e-3, 1e-1]:
        adaylar.append(("gaussian", vs, 1.0, 16, None))
    for vs in [1e-9, 1e-6, 1e-3]:
        for alpha in [0.1, 0.5, 1.0, 2.0]:
            adaylar.append(("hibrit", vs, alpha, 16, None))
    for bs in [8, 16]:
        on_bin = KararAgaci.binle(X_egitim, bs)
        for alpha in [0.1, 0.5, 1.0, 2.0]:
            adaylar.append(("histogram", 1e-6, alpha, bs, on_bin))

    for yontem, vs, alpha, bs, on_bin in adaylar:
        model = NaiveBayes(
            yontem=yontem,
            var_smoothing=vs,
            alpha=alpha,
            bin_sayisi=bs,
        )
        model.fit(X_egitim, y_egitim, ozellik_isimleri, on_binlenmis=on_bin)
        val_olasilik = model.predict_proba(X_dogrulama)
        val_tahmin = (val_olasilik >= 0.5).astype(int)
        metrik = metrikleri_hesapla(
            y_dogrulama, val_tahmin, olasiliklar=val_olasilik
        )
        satir = {
            "yontem": yontem,
            "var_smoothing": vs,
            "alpha": alpha,
            "bin_sayisi": bs if yontem == "histogram" else None,
            "accuracy": metrik["accuracy"],
            "precision": metrik["precision"],
            "recall": metrik["recall"],
            "specificity": metrik["specificity"],
            "f1": metrik["f1"],
            "roc_auc": metrik["roc_auc"],
            "log_loss": metrik["log_loss"],
        }
        sonuclar.append(satir)

        if metrik[hedef_metrik] > en_iyi_skor:
            en_iyi_skor = metrik[hedef_metrik]
            en_iyi_model = model
            en_iyi_ayar = satir

    sonuclar_df = pd.DataFrame(sonuclar).sort_values(
        by=hedef_metrik, ascending=False
    ).reset_index(drop=True)
    return en_iyi_model, en_iyi_ayar, sonuclar_df


# =========================================================
# K-EN YAKIN KOMSU (KNN): HAZIR ML MODELI KULLANILMIYOR
# =========================================================
class KNN:
    def __init__(
        self,
        k=25,
        mesafe="kosinus",
        agirliklandirma="mesafe",
        ozellik_agirlikli=True,
        maks_referans=15000,
        batch_size=1024
    ):
        if mesafe not in ("oklid", "kosinus"):
            raise ValueError(f"Gecersiz mesafe metrigi: {mesafe}")
        if agirliklandirma not in ("esit", "mesafe"):
            raise ValueError(f"Gecersiz agirliklandirma: {agirliklandirma}")
        self.k = int(k)
        self.mesafe = mesafe
        self.agirliklandirma = agirliklandirma
        self.ozellik_agirlikli = ozellik_agirlikli
        self.maks_referans = int(maks_referans)
        self.batch_size = int(batch_size)
        self.ozellik_isimleri = []
        self.ortalama = None
        self.std = None
        self.ozellik_agirliklari = None
        self.X_ref_s = None
        self.y_ref = None
        self.ref_sq = None
        self.X_ref_norm = None

    def fit(self, X, y, ozellik_isimleri):
        self.ozellik_isimleri = list(ozellik_isimleri)
        n_ref = min(len(X), self.maks_referans)
        X_sub = X[:n_ref]
        y_sub = y[:n_ref]

        self.ortalama = np.mean(X_sub, axis=0)
        std = np.std(X_sub, axis=0)
        self.std = np.where(std < 1e-8, 1.0, std)

        X_s = ((X_sub - self.ortalama) / self.std).astype(np.float32)

        if self.ozellik_agirlikli:
            yc = (y_sub - np.mean(y_sub)).astype(np.float32)
            corr = np.abs(
                (X_s.T @ yc) / (len(y_sub) * (np.std(yc) + 1e-8))
            )
            self.ozellik_agirliklari = np.sqrt(
                np.maximum(corr, 1e-4)
            ).astype(np.float32)
        else:
            self.ozellik_agirliklari = np.ones(X_s.shape[1], dtype=np.float32)

        self.X_ref_s = X_s * self.ozellik_agirliklari
        self.y_ref = np.asarray(y_sub, dtype=np.int8)

        if self.mesafe == "kosinus":
            normlar = np.linalg.norm(
                self.X_ref_s, axis=1, keepdims=True
            ) + 1e-8
            self.X_ref_norm = self.X_ref_s / normlar
            self.ref_sq = None
        else:
            self.ref_sq = np.sum(self.X_ref_s ** 2, axis=1)
            self.X_ref_norm = None

        return self

    def _en_yakin_k_bul(self, X, k_max):
        X_q_s = (
            ((X - self.ortalama) / self.std).astype(np.float32)
            * self.ozellik_agirliklari
        )
        n_q = len(X_q_s)
        k_eff = min(k_max, len(self.X_ref_s))
        top_d = np.empty((n_q, k_eff), dtype=np.float32)
        top_y = np.empty((n_q, k_eff), dtype=np.int8)

        if self.mesafe == "kosinus":
            X_q_norm = X_q_s / (
                np.linalg.norm(X_q_s, axis=1, keepdims=True) + 1e-8
            )
            for s in range(0, n_q, self.batch_size):
                e = min(s + self.batch_size, n_q)
                dists = 1.0 - (X_q_norm[s:e] @ self.X_ref_norm.T)
                if k_eff < len(self.X_ref_s):
                    idx = np.argpartition(dists, k_eff, axis=1)[:, :k_eff]
                else:
                    idx = np.tile(np.arange(len(self.X_ref_s)), (e - s, 1))
                row_idx = np.arange(e - s)[:, None]
                bd = dists[row_idx, idx]
                ord_k = np.argsort(bd, axis=1)
                top_d[s:e] = np.maximum(bd[row_idx, ord_k], 0.0)
                top_y[s:e] = self.y_ref[idx[row_idx, ord_k]]
        else:
            for s in range(0, n_q, self.batch_size):
                e = min(s + self.batch_size, n_q)
                q_b = X_q_s[s:e]
                q_sq = np.sum(q_b ** 2, axis=1, keepdims=True)
                dists = np.maximum(
                    q_sq + self.ref_sq[None, :] - 2.0 * (q_b @ self.X_ref_s.T),
                    0.0,
                )
                if k_eff < len(self.X_ref_s):
                    idx = np.argpartition(dists, k_eff, axis=1)[:, :k_eff]
                else:
                    idx = np.tile(np.arange(len(self.X_ref_s)), (e - s, 1))
                row_idx = np.arange(e - s)[:, None]
                bd = dists[row_idx, idx]
                ord_k = np.argsort(bd, axis=1)
                top_d[s:e] = np.sqrt(bd[row_idx, ord_k])
                top_y[s:e] = self.y_ref[idx[row_idx, ord_k]]

        return top_d, top_y

    def predict_proba(self, X):
        top_d, top_y = self._en_yakin_k_bul(X, self.k)
        if self.agirliklandirma == "esit":
            return np.mean(top_y, axis=1, dtype=float)
        w = 1.0 / (top_d.astype(float) + 1e-6)
        return np.sum(w * top_y.astype(float), axis=1) / np.sum(w, axis=1)

    def predict(self, X, karar_esigi=0.5):
        return (self.predict_proba(X) >= karar_esigi).astype(int)

    def ozellik_agirliklari_df(self):
        return pd.DataFrame({
            "ozellik": self.ozellik_isimleri,
            "agirlik": self.ozellik_agirliklari.astype(float),
        }).sort_values(by="agirlik", ascending=False).reset_index(drop=True)


def hiperparametre_ara_knn(
    X_egitim,
    y_egitim,
    X_dogrulama,
    y_dogrulama,
    ozellik_isimleri,
    hedef_metrik="accuracy",
    maks_referans=15000
):
    """Tek gecisli coklu-K Grid Search ile 32 KNN kombinasyonunu tarar."""
    k_adaylari = [7, 15, 25, 35]
    k_max = max(k_adaylari)
    sonuclar = []
    en_iyi_ayar = None
    en_iyi_skor = -1.0

    for mesafe in ["oklid", "kosinus"]:
        for ozellik_agirlikli in [False, True]:
            baz_model = KNN(
                k=k_max,
                mesafe=mesafe,
                agirliklandirma="mesafe",
                ozellik_agirlikli=ozellik_agirlikli,
                maks_referans=maks_referans,
            ).fit(X_egitim, y_egitim, ozellik_isimleri)

            top_d, top_y = baz_model._en_yakin_k_bul(X_dogrulama, k_max)

            for k in k_adaylari:
                td_k = top_d[:, :k]
                ty_k = top_y[:, :k]
                for agirliklandirma in ["esit", "mesafe"]:
                    if agirliklandirma == "esit":
                        val_olasilik = np.mean(ty_k, axis=1, dtype=float)
                    else:
                        w = 1.0 / (td_k.astype(float) + 1e-6)
                        val_olasilik = np.sum(
                            w * ty_k.astype(float), axis=1
                        ) / np.sum(w, axis=1)

                    val_tahmin = (val_olasilik >= 0.5).astype(int)
                    metrik = metrikleri_hesapla(
                        y_dogrulama, val_tahmin, olasiliklar=val_olasilik
                    )
                    satir = {
                        "mesafe": mesafe,
                        "ozellik_agirlikli": ozellik_agirlikli,
                        "k": k,
                        "agirliklandirma": agirliklandirma,
                        "maks_referans": len(baz_model.X_ref_s),
                        "accuracy": metrik["accuracy"],
                        "precision": metrik["precision"],
                        "recall": metrik["recall"],
                        "specificity": metrik["specificity"],
                        "f1": metrik["f1"],
                        "roc_auc": metrik["roc_auc"],
                        "log_loss": metrik["log_loss"],
                    }
                    sonuclar.append(satir)

                    if metrik[hedef_metrik] > en_iyi_skor:
                        en_iyi_skor = metrik[hedef_metrik]
                        en_iyi_ayar = satir

    en_iyi_model = KNN(
        k=en_iyi_ayar["k"],
        mesafe=en_iyi_ayar["mesafe"],
        agirliklandirma=en_iyi_ayar["agirliklandirma"],
        ozellik_agirlikli=en_iyi_ayar["ozellik_agirlikli"],
        maks_referans=maks_referans,
    ).fit(X_egitim, y_egitim, ozellik_isimleri)

    sonuclar_df = pd.DataFrame(sonuclar).sort_values(
        by=hedef_metrik, ascending=False
    ).reset_index(drop=True)
    return en_iyi_model, en_iyi_ayar, sonuclar_df
