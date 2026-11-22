"""Draw the corrected system-architecture diagram (PNG + PDF) and write an editable draw.io file."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from xml.sax.saxutils import escape

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out_dir.mkdir(parents=True, exist_ok=True)

C = {"data": "#dbe9f6", "dl": "#e3f1df", "rl": "#fdebd3", "eval": "#ece3f3", "label": "#fbe0e0"}
EDGE = {"data": "#3b78b0", "dl": "#3f8f3f", "rl": "#c77a12", "eval": "#7653a1", "label": "#b33a3a"}

# id: (x, y, w, h, kind, title, body)
B = {
    "data":   (0.3, 8.0, 3.4, 1.25, "data", "ESA-ADB Mission 1", "Real telemetry + expert annotations\n12 target channels, 2000–2013"),
    "prep":   (0.3, 6.2, 3.4, 1.35, "data", "Preprocessing", "90 s common grid\nInterpolate gaps ≤ 4.5 min only\nLong gaps left missing"),
    "split":  (0.3, 4.4, 3.4, 1.35, "data", "Event-level split", "168 events → 117 / 25 / 26\nLabel-clean segments\n→ train / val / test"),
    "ae":     (4.3, 7.9, 3.4, 1.35, "dl", "LSTM autoencoder", "128-sample (3.2 h) windows\nTrained on label-clean\ntrain-split windows"),
    "score":  (4.3, 6.2, 3.4, 1.35, "dl", "Anomaly scoring", "Overall + 12 channel-wise\nreconstruction errors\n(z-score, ±10σ clip)"),
    "det":    (4.3, 4.4, 3.4, 1.35, "dl", "Detection", "Window error ≥ threshold\n(threshold fixed on validation,\nfrozen for test)"),
    "case":   (4.3, 2.6, 3.4, 1.35, "rl", "Investigation case", "Highest-error detected window\nof a confirmed event"),
    "state":  (8.3, 8.0, 3.6, 1.25, "rl", "DQN state (25)", "log1p(12 channel errors)\n12 status values + budget left"),
    "agent":  (8.3, 6.2, 3.6, 1.35, "rl", "DQN agent", "Q-network 25→64→64→12\nReplay buffer, target network\nε-greedy, invalid actions masked"),
    "action": (8.3, 4.4, 3.6, 1.35, "rl", "Action", "Investigate 1 of 12 channels\nBudget −1 per action\nEpisode ends after 5"),
    "reward": (8.3, 2.6, 3.6, 1.35, "rl", "Reward", "+1 annotated channel\n−1 not annotated"),
    "labels": (12.5, 2.6, 3.4, 1.35, "label", "Ground-truth annotations", "Used only for reward,\ncase association & evaluation\n(never in the initial state)"),
    "base":   (12.5, 8.0, 3.4, 1.25, "eval", "Baselines", "Random (×1000), Fixed priority,\nLSTM-greedy, Oracle bound"),
    "eval":   (12.5, 6.2, 3.4, 1.35, "eval", "Evaluation (test events)", "DQN greedy policy, 5 seeds\n(best validation checkpoint)\nmean ± std, Wilcoxon tests"),
    "metrics":(12.5, 4.4, 3.4, 1.35, "eval", "Metrics", "Avg. reward • Hits • Precision@5\nRecall@5 • Unnecessary investigations"),
}
E = [("data", "prep", ""), ("prep", "split", ""), ("split", "ae", ""), ("ae", "score", ""),
     ("score", "det", ""), ("det", "case", ""), ("case", "state", "channel errors"), ("state", "agent", ""),
     ("agent", "action", ""), ("action", "reward", ""), ("labels", "reward", ""), ("reward", "agent", "(s, a, r, s′)\n+ status update"),
     ("agent", "eval", ""), ("base", "eval", ""), ("eval", "metrics", ""),
     ("labels", "metrics", "")]

def anchor(b, side):
    x, y, w, h = B[b][:4]
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]

def sides(a, b):
    ax, ay, aw, ah = B[a][:4]; bx, by, bw, bh = B[b][:4]
    if abs(ax - bx) < 0.5:
        return ("b", "t") if ay > by else ("t", "b")
    return ("r", "l") if ax < bx else ("l", "r")

fig, ax = plt.subplots(figsize=(17, 10.5)); ax.set_xlim(0, 16.2); ax.set_ylim(1.6, 10.3); ax.axis("off")
ax.text(8.1, 10.0, "System architecture: LSTM-AE-assisted DQN for budget-constrained satellite telemetry investigation",
        ha="center", fontsize=15, weight="bold")
for col, (x, t) in enumerate([(2.0, "Data preparation"), (6.0, "Deep learning: anomaly scoring"), (10.1, "Reinforcement learning"), (14.2, "Evaluation")]):
    ax.text(x, 9.55, t, ha="center", fontsize=11, color="#444", style="italic")
for k, (x, y, w, h, kind, title, body) in B.items():
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=C[kind], ec=EDGE[kind], lw=1.6))
    ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + (h - 0.4) / 2, body, ha="center", va="center", fontsize=8.6, linespacing=1.35)
special = {("reward", "agent"): ("r", "r", "arc3,rad=0.26"),
           ("labels", "metrics"): ("t", "b", "arc3,rad=0"), ("split", "ae"): ("r", "l", "arc3,rad=-0.25")}
for a, b, lab in E:
    if (a, b) in special:
        sa, sb, cs = special[(a, b)]
    else:
        (sa, sb), cs = sides(a, b), "arc3,rad=0"
    p, q = anchor(a, sa), anchor(b, sb)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#555", connectionstyle=cs))
    if lab and (a, b) == ("reward", "agent"):
        ax.text(12.2, (p[1] + q[1]) / 2, "(s, a, r, s′) + status update", rotation=90, fontsize=7.6, color="#333",
                ha="center", va="center", bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.4))
    elif lab:
        ax.text((p[0] + q[0]) / 2 + (0.12 if (a, b) == ("reward", "agent") else 0.08), (p[1] + q[1]) / 2, lab,
                fontsize=7.8, color="#333", ha="left", va="center", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
ax.text(8.1, 1.85, "Annotations never enter the initial state; a channel's association becomes visible only after it is investigated "
        "(status 0.5 / 1). The study evaluates channel prioritisation, not proven causal root causes.",
        ha="center", fontsize=9, color="#7a2020")
fig.savefig(out_dir / "Satellite_Telemetry_RL_System_Architecture_corrected.png", dpi=220, bbox_inches="tight")
fig.savefig(out_dir / "Satellite_Telemetry_RL_System_Architecture_corrected.pdf", bbox_inches="tight")

# editable draw.io version (same boxes and arrows)
S = 60
cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
for k, (x, y, w, h, kind, title, body) in B.items():
    val = escape(f"<b>{title}</b><br>" + body.replace("\n", "<br>"), {'"': "&quot;"})
    cells.append(f'<mxCell id="{k}" value="{val}" style="rounded=1;whiteSpace=wrap;html=1;fillColor={C[kind]};strokeColor={EDGE[kind]};fontSize=12;" vertex="1" parent="1">'
                 f'<mxGeometry x="{x*S:.0f}" y="{(10.3-y-h)*S:.0f}" width="{w*S:.0f}" height="{h*S:.0f}" as="geometry"/></mxCell>')
for i, (a, b, lab) in enumerate(E):
    cells.append(f'<mxCell id="e{i}" value="{escape(lab.replace(chr(10), " "))}" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;fontSize=10;" edge="1" parent="1" source="{a}" target="{b}"><mxGeometry relative="1" as="geometry"/></mxCell>')
xml = ('<mxfile host="app.diagrams.net"><diagram name="Corrected architecture"><mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" '
       'page="1" pageWidth="1100" pageHeight="700"><root>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
(out_dir / "Satellite_Telemetry_RL_System_Architecture_corrected.drawio").write_text(xml)
print("diagram written to", out_dir)
