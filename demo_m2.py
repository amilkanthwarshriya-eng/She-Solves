"""Standalone M2 demo:  python demo_m2.py"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from app.detection.rule_engine import analyze_text, get_rules_config  # noqa: E402
from app.nlp.inference import classify_text  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):   # emoji / rupee sign in Windows PowerShell
    sys.stdout.reconfigure(encoding="utf-8")

DARKSHOP_PRODUCT = """
Premium Wireless Headphones
₹4,999
₹799
84% OFF
🔥 ONLY 2 LEFT!
Offer ends in 08:32
BUY NOW
"""

DARKSHOP_CHECKOUT = """
Headphones ₹799
Add ₹50 donation
No, I don't want to save money.
Delivery ₹99
Platform fee ₹49
Handling fee ₹20
TOTAL ₹967
"""


def show(title: str, text: str, page: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)
    findings = analyze_text(text, page=page)
    for i, f in enumerate(findings, 1):
        print(f"[{i}] {f['rule_id']}  {f['name']}   ({f['pattern']})")
        print(f"    Severity       : {f['severity']}   (fixed in rules config)")
        print(f"    Confidence     : {f['confidence']:.2f}   (rule-based; DeBERTa not added yet)")
        print(f"    Status         : {f['status']}")
        print(f"    Evidence text  : {f['evidence']['text']}")
        print(f"    Explanation    : {f['explanation']}")
        print(f"    Recommendation : {f['recommendation']}\n")
    print(f"Total findings: {len(findings)}\n")
    print("JSON:")
    print(json.dumps(findings, indent=2, ensure_ascii=False))
    print()


if __name__ == "__main__":
    print("Rules configured (DP01-DP05):")
    for r in get_rules_config():
        who = "M2 detects" if r["m2_detects"] else f"{r['detector_owner']} detects"
        print(f"  {r['rule_id']}  {r['name']:<20} {r['method']:<15} {r['severity']:<7} [{who}]")
    print()
    show("DarkShop - PRODUCT page", DARKSHOP_PRODUCT, "product")
    show("DarkShop - CHECKOUT page", DARKSHOP_CHECKOUT, "checkout")
    print("Model-level output (text -> pattern + confidence):")
    for t in ("Only 2 rooms left! Hurry!", "No, I don't want to save money.", "No thanks."):
        print(f"  {t!r:42} -> {classify_text(t)}")
