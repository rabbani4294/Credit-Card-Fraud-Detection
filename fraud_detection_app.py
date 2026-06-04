"""
CreditGuard AI — Credit Card Fraud Detection System
Professional Tkinter GUI  |  XGBoost + SMOTE  |  ROC-AUC 0.9835

USAGE
-----
Place this file in the same folder as:
  • fraud_model.pkl       (XGBoost model)
  • amount_scaler.pkl     (StandardScaler for Amount)

Run:  python fraud_detection_app.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib
import os, sys, threading, math


# ─────────────────────────── THEME ────────────────────────────
BG_DARK  = "#0A0E1A"
BG_CARD  = "#111827"
BG_INPUT = "#1A2235"
BORDER   = "#1E3A5F"
ACCENT   = "#00D4FF"
ACCENT2  = "#7C3AED"
SUCCESS  = "#10B981"
DANGER   = "#EF4444"
WARNING  = "#F59E0B"
TXT_PRI  = "#F1F5F9"
TXT_SEC  = "#94A3B8"
TXT_MUT  = "#475569"
MONO     = ("Courier New", 10)


# ─────────────────────────── FEATURES ─────────────────────────
FEATURES = [
    "Time",
    "V1","V2","V3","V4","V5","V6","V7",
    "V8","V9","V10","V11","V12","V13","V14",
    "V15","V16","V17","V18","V19","V20","V21",
    "V22","V23","V24","V25","V26","V27","V28",
    "Amount",
]

HINTS = {"Time": "Seconds since first transaction in dataset",
         "Amount": "Transaction USD (will be scaled automatically)"}
for _i in range(1, 29):
    HINTS[f"V{_i}"] = f"PCA-anonymised component V{_i}"

FRAUD_SAMPLE = {
    "Time":"406","V1":"-2.3122","V2":"1.9520","V3":"-1.6099","V4":"3.9979",
    "V5":"-0.5222","V6":"-1.4265","V7":"-2.5374","V8":"1.3917",
    "V9":"-2.7701","V10":"-2.7723","V11":"3.2020","V12":"-2.8999",
    "V13":"-0.5952","V14":"-4.2893","V15":"0.3897","V16":"-1.1407",
    "V17":"-2.8301","V18":"-0.0168","V19":"0.4170","V20":"0.1269",
    "V21":"0.5172","V22":"-0.0350","V23":"-0.4652","V24":"0.3202",
    "V25":"0.0445","V26":"0.1778","V27":"0.2611","V28":"-0.1433",
    "Amount":"239.93",
}

NORMAL_SAMPLE = {
    "Time":"0","V1":"-1.3598","V2":"-0.0728","V3":"2.5363","V4":"1.3782",
    "V5":"-0.3383","V6":"0.4624","V7":"0.2396","V8":"0.0987",
    "V9":"0.3638","V10":"0.0908","V11":"-0.5516","V12":"-0.6178",
    "V13":"-0.9914","V14":"-0.3112","V15":"1.4682","V16":"-0.4704",
    "V17":"0.2080","V18":"0.0258","V19":"0.4040","V20":"0.2514",
    "V21":"-0.0183","V22":"0.2778","V23":"-0.1105","V24":"0.0669",
    "V25":"0.1285","V26":"-0.1891","V27":"0.1336","V28":"-0.0211",
    "Amount":"149.62",
}


# ─────────────────────────── UTILS ────────────────────────────
def resource_path(filename):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, filename)


def hex_blend(fg, bg, alpha):
    def c(f, b): return int(int(f, 16)*alpha + int(b, 16)*(1-alpha))
    return (f"#{c(fg[1:3],bg[1:3]):02x}"
            f"{c(fg[3:5],bg[3:5]):02x}"
            f"{c(fg[5:7],bg[5:7]):02x}")


# ─────────────────────────── PULSE RING ───────────────────────
class PulseRing(tk.Canvas):
    SZ = 140
    def __init__(self, parent, **kw):
        super().__init__(parent, width=self.SZ, height=self.SZ,
                         bg=BG_CARD, highlightthickness=0, **kw)
        self._color = ACCENT
        self._phase = 0
        self._active = False
        self._job = None

    def set_color(self, color):
        self._color = color

    def start(self):
        self._active = True
        self._tick()

    def stop(self):
        self._active = False
        if self._job:
            self.after_cancel(self._job)
            self._job = None
        self.delete("all")

    def _tick(self):
        if not self._active:
            return
        self.delete("all")
        cx = cy = self.SZ // 2
        base_r = self.SZ // 4
        for i in range(3):
            offset = (self._phase + i*20) % 60
            r = base_r + offset
            a = max(0.0, 1.0 - offset/60.0)
            col = hex_blend(self._color, BG_CARD, a)
            self.create_oval(cx-r, cy-r, cx+r, cy+r, outline=col, width=2, fill="")
        r2 = base_r - 5
        self.create_oval(cx-r2, cy-r2, cx+r2, cy+r2, fill=self._color, outline="")
        self._phase = (self._phase + 3) % 60
        self._job = self.after(40, self._tick)


# ─────────────────────────── FEATURE ENTRY ────────────────────
class FeatureEntry(tk.Frame):
    def __init__(self, parent, label, hint="", **kw):
        super().__init__(parent, bg=BG_CARD, **kw)
        self.columnconfigure(1, weight=1)

        tk.Label(self, text=label, font=("Consolas", 10, "bold"),
                 fg=ACCENT, bg=BG_CARD, width=13, anchor="e"
                 ).grid(row=0, column=0, padx=(8, 10), pady=2)

        self._bf = tk.Frame(self, bg=BORDER, padx=1, pady=1)
        self._bf.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self._bf.columnconfigure(0, weight=1)

        self.var = tk.StringVar()
        self._e = tk.Entry(self._bf, textvariable=self.var,
                           font=MONO, bg=BG_INPUT, fg=TXT_PRI,
                           insertbackground=ACCENT, relief="flat",
                           bd=0, highlightthickness=0)
        self._e.grid(sticky="ew", padx=6, pady=5)

        if hint:
            tk.Label(self, text=hint, font=("Consolas", 7),
                     fg=TXT_MUT, bg=BG_CARD).grid(
                row=1, column=1, sticky="w", padx=(0, 8), pady=(0, 1))

        self._e.bind("<FocusIn>",  lambda _: self._bf.config(bg=ACCENT))
        self._e.bind("<FocusOut>", lambda _: self._bf.config(bg=BORDER))

    def get(self): return self.var.get().strip()
    def set(self, v): self.var.set(v)
    def clear(self): self.var.set("")
    def focus(self): self._e.focus_set()


# ─────────────────────────── APP ──────────────────────────────
class FraudDetectionApp:
    BAR_W = 220

    def __init__(self, root):
        self.root = root
        self.model = self.scaler = None
        self.entries = {}

        root.title("CreditGuard AI — Fraud Detection System")
        root.geometry("1020x840")
        root.minsize(860, 700)
        root.configure(bg=BG_DARK)
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Vertical.TScrollbar", background=BG_INPUT,
                        troughcolor=BG_CARD, bordercolor=BG_CARD,
                        arrowcolor=TXT_MUT, relief="flat", gripcount=0)
        style.map("Vertical.TScrollbar", background=[("active", BORDER)])

        self._build_header()
        self._build_body()
        self._build_footer()
        self._load_model_async()

    # ── Header ────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=BG_CARD, height=90)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        hdr.columnconfigure(1, weight=1)

        ic = tk.Canvas(hdr, width=60, height=60, bg=BG_CARD, highlightthickness=0)
        ic.grid(row=0, column=0, padx=(20, 12), pady=15)
        self._draw_shield(ic, 30, 30, 25)

        tf = tk.Frame(hdr, bg=BG_CARD)
        tf.grid(row=0, column=1, sticky="w")
        tk.Label(tf, text="CreditGuard AI", font=("Georgia", 23, "bold"),
                 fg=TXT_PRI, bg=BG_CARD).pack(anchor="w")
        tk.Label(tf,
                 text="Credit Card Fraud Detection  ·  XGBoost Classifier  ·  SMOTE Balanced",
                 font=("Consolas", 8), fg=TXT_SEC, bg=BG_CARD).pack(anchor="w")

        self._status = tk.Label(hdr, text="⏳  Loading model…",
                                font=("Consolas", 9), fg=WARNING, bg=BG_CARD, padx=16)
        self._status.grid(row=0, column=2, padx=12)
        tk.Frame(self.root, bg=BORDER, height=1).grid(row=0, column=0, sticky="sew")

    # ── Body ─────────────────────────────────────────────────
    def _build_body(self):
        body = tk.Frame(self.root, bg=BG_DARK)
        body.grid(row=1, column=0, sticky="nsew", padx=16, pady=14)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        self._build_form(body)
        self._build_result(body)

    # ── Form panel ───────────────────────────────────────────
    def _build_form(self, parent):
        card = tk.Frame(parent, bg=BG_CARD)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)

        ph = tk.Frame(card, bg=BG_CARD)
        ph.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 0))
        tk.Label(ph, text="Transaction Feature Input",
                 font=("Georgia", 13, "bold"), fg=TXT_PRI, bg=BG_CARD).pack(side="left")
        tk.Label(ph, text="30 / 30 required",
                 font=("Consolas", 8), fg=TXT_MUT, bg=BG_CARD).pack(side="right")
        tk.Frame(card, bg=BORDER, height=1).grid(row=0, column=0, sticky="sew",
                                                  padx=12, pady=(38, 0))

        wrap = tk.Frame(card, bg=BG_CARD)
        wrap.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)

        canvas = tk.Canvas(wrap, bg=BG_CARD, highlightthickness=0)
        vsb    = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        sf     = tk.Frame(canvas, bg=BG_CARD)

        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=vsb.set)
        sf.columnconfigure(0, weight=1)

        win = canvas.create_window((0, 0), window=sf, anchor="nw")
        sf.bind("<Configure>",
                lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

        for idx, feat in enumerate(FEATURES):
            fe = FeatureEntry(sf, feat, HINTS.get(feat, ""))
            fe.grid(row=idx, column=0, sticky="ew", padx=4, pady=1)
            self.entries[feat] = fe

        btn_row = tk.Frame(card, bg=BG_CARD)
        btn_row.grid(row=2, column=0, sticky="ew", padx=14, pady=(8, 4))
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)

        self._btn(btn_row, "🔍  Predict", ACCENT, "#00222F",
                  self._on_predict).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self._btn(btn_row, "↺  Clear All", TXT_MUT, BG_INPUT,
                  self._on_clear).grid(row=0, column=1, sticky="ew", padx=(5, 0))

        smp = tk.Frame(card, bg=BG_CARD)
        smp.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 12))
        smp.columnconfigure(0, weight=1)
        smp.columnconfigure(1, weight=1)

        self._btn(smp, "⚡ Load Fraud Sample", DANGER, "#220A0A",
                  lambda: self._load_sample(FRAUD_SAMPLE)
                  ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self._btn(smp, "✅ Load Normal Sample", SUCCESS, "#0A2215",
                  lambda: self._load_sample(NORMAL_SAMPLE)
                  ).grid(row=0, column=1, sticky="ew", padx=(5, 0))

    # ── Result panel ─────────────────────────────────────────
    def _build_result(self, parent):
        card = tk.Frame(parent, bg=BG_CARD)
        card.grid(row=0, column=1, sticky="nsew")
        card.columnconfigure(0, weight=1)

        tk.Label(card, text="Analysis Result",
                 font=("Georgia", 13, "bold"), fg=TXT_PRI, bg=BG_CARD
                 ).grid(row=0, column=0, sticky="w", padx=14, pady=(14, 0))
        tk.Frame(card, bg=BORDER, height=1).grid(row=0, column=0, sticky="sew",
                                                   padx=12, pady=(38, 0))

        rw = tk.Frame(card, bg=BG_CARD)
        rw.grid(row=1, column=0, pady=(22, 6))
        self._pulse = PulseRing(rw)
        self._pulse.pack()
        self._icon_lbl = tk.Label(rw, text="?", font=("Georgia", 36, "bold"),
                                  fg=TXT_MUT, bg=BG_CARD)
        self._icon_lbl.place(relx=0.5, rely=0.5, anchor="center", in_=rw)

        self._verdict = tk.Label(card, text="Awaiting Input",
                                 font=("Georgia", 18, "bold"),
                                 fg=TXT_MUT, bg=BG_CARD, wraplength=270)
        self._verdict.grid(row=2, column=0, pady=(4, 2))

        self._pct = tk.Label(card, text="— %", font=("Courier New", 26, "bold"),
                             fg=TXT_MUT, bg=BG_CARD)
        self._pct.grid(row=3, column=0, pady=(2, 0))

        tk.Label(card, text="Fraud Probability", font=("Consolas", 8),
                 fg=TXT_SEC, bg=BG_CARD).grid(row=4, column=0, pady=(6, 2))

        bar_bg = tk.Frame(card, bg=BG_INPUT, height=12, width=self.BAR_W)
        bar_bg.grid(row=5, column=0, pady=(0, 4))
        bar_bg.grid_propagate(False)
        self._bar = tk.Frame(bar_bg, bg=TXT_MUT, height=12)
        self._bar.place(x=0, y=0, width=0, height=12)

        det = tk.Frame(card, bg=BG_INPUT, padx=12, pady=10)
        det.grid(row=6, column=0, sticky="ew", padx=14, pady=(12, 4))
        det.columnconfigure(0, weight=1)
        self._detail = tk.Label(det,
                                text="Submit 30 transaction features\nto run fraud analysis.",
                                font=("Consolas", 9), fg=TXT_SEC, bg=BG_INPUT,
                                justify="center", wraplength=250)
        self._detail.pack()

        mf = tk.Frame(card, bg=BG_CARD)
        mf.grid(row=7, column=0, sticky="ew", padx=14, pady=(10, 4))
        for c in range(3):
            mf.columnconfigure(c, weight=1)
        self._s_pred = self._stat(mf, "Prediction", 0)
        self._s_conf = self._stat(mf, "Confidence",  1)
        self._s_risk = self._stat(mf, "Risk Level",  2)

        tk.Label(card, text="Session History",
                 font=("Consolas", 8), fg=TXT_MUT, bg=BG_CARD
                 ).grid(row=8, column=0, sticky="w", padx=16, pady=(12, 2))

        hw = tk.Frame(card, bg=BG_INPUT)
        hw.grid(row=9, column=0, sticky="ew", padx=14, pady=(0, 14))
        hw.columnconfigure(0, weight=1)
        self._hist = tk.Text(hw, height=5, bg=BG_INPUT, fg=TXT_SEC,
                             font=("Consolas", 8), relief="flat", bd=0,
                             state="disabled", cursor="arrow", insertwidth=0)
        self._hist.pack(fill="x", padx=8, pady=6)

    # ── Footer ────────────────────────────────────────────────
    def _build_footer(self):
        foot = tk.Frame(self.root, bg=BG_CARD, height=28)
        foot.grid(row=2, column=0, sticky="ew")
        foot.grid_propagate(False)
        tk.Label(foot, font=("Consolas", 7), fg=TXT_MUT, bg=BG_CARD,
                 text=("XGBoost  ·  284,807 transactions  ·  "
                       "ROC-AUC 0.9835  ·  SMOTE over-sampling")
                 ).pack(side="left", padx=16, pady=6)
        tk.Label(foot, text="CreditGuard AI v1.0",
                 font=("Consolas", 7), fg=TXT_MUT, bg=BG_CARD
                 ).pack(side="right", padx=16)

    # ── Widget factories ──────────────────────────────────────
    def _btn(self, parent, text, fg, bg, cmd):
        b = tk.Button(parent, text=text, font=("Consolas", 10, "bold"),
                      fg=fg, bg=bg, activeforeground=fg, activebackground=BORDER,
                      relief="flat", bd=0, cursor="hand2",
                      padx=10, pady=8, command=cmd)
        b.bind("<Enter>", lambda _: b.config(bg=BORDER))
        b.bind("<Leave>", lambda _: b.config(bg=bg))
        return b

    def _stat(self, parent, label, col):
        f = tk.Frame(parent, bg=BG_INPUT, padx=8, pady=6)
        f.grid(row=0, column=col, sticky="ew", padx=3)
        tk.Label(f, text=label, font=("Consolas", 7), fg=TXT_MUT, bg=BG_INPUT).pack()
        v = tk.Label(f, text="—", font=("Consolas", 11, "bold"), fg=TXT_SEC, bg=BG_INPUT)
        v.pack()
        return v

    def _draw_shield(self, canvas, cx, cy, r):
        pts = []
        for deg in range(-60, 121, 8):
            rad = math.radians(deg)
            pts += [cx + r*math.sin(rad), cy - r*math.cos(rad)]
        pts += [cx, cy + r*1.35,
                cx - r*math.sin(math.radians(60)),
                cy - r*math.cos(math.radians(60))]
        canvas.create_polygon(pts, fill=ACCENT2, outline=ACCENT, width=2.5)
        canvas.create_text(cx, cy, text="✓", font=("Arial", 15, "bold"), fill=TXT_PRI)

    # ── Model ─────────────────────────────────────────────────
    def _load_model_async(self):
        def _w():
            try:
                m = joblib.load(resource_path("fraud_model.pkl"))
                s = joblib.load(resource_path("amount_scaler.pkl"))
                self.root.after(0, lambda: self._model_ok(m, s))
            except Exception as exc:
                self.root.after(0, lambda: self._model_err(str(exc)))
        threading.Thread(target=_w, daemon=True).start()

    def _model_ok(self, m, s):
        self.model, self.scaler = m, s
        self._status.config(text="✅  Model ready  (XGBoost)", fg=SUCCESS)

    def _model_err(self, msg):
        self._status.config(text="❌  Model load failed", fg=DANGER)
        messagebox.showerror("Model Load Error",
                             f"Could not load fraud_model.pkl / amount_scaler.pkl.\n\n"
                             f"Error: {msg}\n\n"
                             "Place both .pkl files in the same directory as this script.")

    # ── Predict ───────────────────────────────────────────────
    def _on_predict(self):
        if self.model is None:
            messagebox.showwarning("Model Not Ready",
                                   "Model is still loading — please wait a moment.")
            return
        raw = []
        for feat in FEATURES:
            val = self.entries[feat].get()
            if not val:
                messagebox.showerror("Missing Input",
                                     f"Field '{feat}' is empty.\nAll 30 values are required.")
                self.entries[feat].focus()
                return
            try:
                raw.append(float(val))
            except ValueError:
                messagebox.showerror("Invalid Input",
                                     f"'{val}' in '{feat}' is not a valid number.")
                self.entries[feat].focus()
                return

        # raw = [Time, V1…V28, Amount]
        # model expects [Time, V1…V28, Scaled_Amount]
        amount    = raw[-1]
        scaled    = float(self.scaler.transform([[amount]])[0][0])
        inp       = np.array([raw[:-1] + [scaled]], dtype=np.float64)
        pred      = int(self.model.predict(inp)[0])
        prob      = float(self.model.predict_proba(inp)[0][1])
        self._show_result(pred, prob, amount)

    def _show_result(self, pred, prob, amount):
        pct = prob * 100
        if pred == 1:
            color, icon = DANGER, "⚠"
            verdict = "FRAUD DETECTED"
            detail  = (f"Transaction flagged as FRAUDULENT\n"
                       f"Amount: ${amount:,.2f}\n"
                       f"Fraud probability: {pct:.2f}%\n"
                       f"⚠  Immediate action recommended")
            risk, risk_c = "HIGH", DANGER
            conf = f"{pct:.1f}%"
        else:
            color, icon = SUCCESS, "✓"
            verdict = "NORMAL TRANSACTION"
            detail  = (f"Transaction appears LEGITIMATE\n"
                       f"Amount: ${amount:,.2f}\n"
                       f"Fraud probability: {pct:.2f}%\n"
                       f"✅  No action required")
            risk  = "MEDIUM" if pct >= 10 else "LOW"
            risk_c = WARNING if risk == "MEDIUM" else SUCCESS
            conf  = f"{(1-prob)*100:.1f}%"

        self._verdict.config(text=verdict, fg=color)
        self._icon_lbl.config(text=icon, fg=color)
        self._pct.config(text=f"{pct:.2f}%", fg=color)
        self._detail.config(text=detail, fg=TXT_SEC)
        self._s_pred.config(text="FRAUD" if pred == 1 else "NORMAL", fg=color)
        self._s_conf.config(text=conf, fg=color)
        self._s_risk.config(text=risk, fg=risk_c)
        self._bar.config(bg=color)
        self._bar.place(width=int(self.BAR_W * prob))
        self._pulse.set_color(color)
        self._pulse.stop()
        self._pulse.start()

        tag  = "FRAUD" if pred == 1 else "OK   "
        line = f"[{tag}]  ${amount:8,.2f}   {pct:6.2f}%\n"
        self._hist.config(state="normal")
        self._hist.insert("1.0", line)
        self._hist.config(state="disabled")

    def _on_clear(self):
        for fe in self.entries.values():
            fe.clear()
        self._verdict.config(text="Awaiting Input", fg=TXT_MUT)
        self._icon_lbl.config(text="?", fg=TXT_MUT)
        self._pct.config(text="— %", fg=TXT_MUT)
        self._detail.config(
            text="Submit 30 transaction features\nto run fraud analysis.", fg=TXT_SEC)
        self._s_pred.config(text="—", fg=TXT_SEC)
        self._s_conf.config(text="—", fg=TXT_SEC)
        self._s_risk.config(text="—", fg=TXT_SEC)
        self._bar.place(width=0)
        self._pulse.stop()

    def _load_sample(self, sample):
        self._on_clear()
        for feat, val in sample.items():
            if feat in self.entries:
                self.entries[feat].set(val)


# ─────────────────────────── MAIN ─────────────────────────────
def main():
    root = tk.Tk()
    app  = FraudDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW",
                  lambda: (app._pulse.stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
