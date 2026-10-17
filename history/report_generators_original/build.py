"""tables -> macros -> substitute -> pdflatex x2. Tables are substituted, not
\\input, because \\input inside a tabular breaks booktabs alignment."""
import pathlib, re, subprocess, sys
src=pathlib.Path("main_template.tex").read_text()
for f in sorted(pathlib.Path(".").glob("tab_*.tex")):
    key=f.stem[4:]; src=src.replace(f"%%{key}%%", f.read_text().rstrip())
left=re.findall(r"%%[A-Z0-9]+%%",src)
if left: sys.exit(f"unsubstituted: {left}")
# inline the macros so main.tex is a single self-contained file
src=src.replace("\\input{macros.tex}", pathlib.Path("macros.tex").read_text().rstrip())
pathlib.Path("main.tex").write_text(src)
for _ in range(2):
    subprocess.run(["pdflatex","-interaction=nonstopmode","main.tex"],capture_output=True)
log=pathlib.Path("main.log").read_text(errors="ignore")
errs=[l for l in log.splitlines() if l.startswith("! ")]
und=[l for l in log.splitlines() if "undefined" in l.lower() and "Warning" in l]
print(f"errors {len(errs)}   undefined refs/cites {len(und)}")
for e in errs[:8]: print("  ",e)
for u in und[:8]: print("  ",u)
