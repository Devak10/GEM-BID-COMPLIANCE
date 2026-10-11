import re

with open("app/app.py", "r", encoding="utf-8") as f:
    text = f.read()

classes = set(re.findall(r'class=["\']([^"\']+)["\']', text))
print("Classes found in app.py:")
for c in sorted(classes):
    print(f" - {c}")

# Find any st.markdown with border or card
print("\nCard/border search:")
for i, line in enumerate(text.splitlines(), 1):
    if "border:" in line or "border-radius" in line or "card" in line.lower():
        if i > 370: # outside the style block
            print(f"Line {i}: {line[:100]}")
