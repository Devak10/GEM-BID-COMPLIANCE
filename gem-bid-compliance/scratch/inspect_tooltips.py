import re

with open("app/app.py", "r", encoding="utf-8") as f:
    text = f.read()

tooltips = re.findall(r'help=[\'"]([^\'"]+)[\'"]', text)
print(f"Total help tooltips found: {len(tooltips)}")
for t in tooltips:
    print(f" - {t}")
