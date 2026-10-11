import re

with open("app/app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, l in enumerate(lines, 1):
    if "px." in l or "go." in l or "st.plotly_chart" in l:
        print(f"Line {i}: {l.strip()}")
