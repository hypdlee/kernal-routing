"""
Patch analyze_capacity.py in place.

Fixes: the non-monotone verdict fires even when no width reaches
significance, so three non-significant point estimates get described as
having a peak. Equivalence is now checked first.

    python patch_analyze.py

Safe to run twice. Makes a .bak. Verifies the result compiles.
"""

import ast
import datetime
import pathlib
import shutil
import sys

TARGET = pathlib.Path("analyze_capacity.py")

OLD = """        if interior_peak and not monotone:"""

NEW = """        # An interior peak is only meaningful if there is an effect to have
        # a shape. When every width is equivalent to static, the ordering of
        # the point estimates is seed noise and must not be described as a
        # peak. Equivalence is therefore checked first.
        all_equiv = all(q[3] < 0.05 for q in pts if not math.isnan(q[3]))
        any_sig = any(q[2] < 0.05 for q in pts)

        if all_equiv and not any_sig:
            print("    -> equivalent to static at every width tested. No effect at any")
            print("       capacity, so the ordering of the point estimates is noise.")
        elif interior_peak and not monotone and any_sig:"""

OLD2 = """        elif all(q[3] < 0.05 for q in pts if not math.isnan(q[3])):
            print("    -> equivalent to static at every width tested. The null holds")
            print("       across capacity, which is the strongest form of the claim.")"""

NEW2 = """        elif all_equiv:
            print("    -> equivalent to static at every width tested, though at least")
            print("       one width also reached significance. Read the rows.")"""


def main():
    print(f"working directory: {pathlib.Path.cwd()}")
    if not TARGET.exists():
        print(f"\nFAIL: {TARGET.name} is not in this directory.")
        print("cd to the folder that holds run_stage1.py and try again.")
        return 1

    mtime = datetime.datetime.fromtimestamp(TARGET.stat().st_mtime)
    text = TARGET.read_text(encoding="utf-8")
    print(f"target: {TARGET.resolve()}")
    print(f"        {len(text)} bytes, last modified {mtime:%Y-%m-%d %H:%M}")

    if "all_equiv" in text:
        print("\nAlready patched. Nothing to do.")
        return 0

    if text.count(OLD) != 1:
        print(f"\nFAIL: expected exactly one match for the anchor line, found "
              f"{text.count(OLD)}.")
        print("The file is not the version this patch was written for.")
        return 1

    patched = text.replace(OLD, NEW, 1)
    if patched.count(OLD2) == 1:
        patched = patched.replace(OLD2, NEW2, 1)
    else:
        print("note: second (cosmetic) hunk not found; skipping it.")

    try:
        ast.parse(patched)
    except SyntaxError as e:
        print(f"\nFAIL: patched file would not compile ({e}). Nothing written.")
        return 1

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(patched, encoding="utf-8")

    print(f"\nPatched. Backup written to {backup.name}")
    print("Verifying: 'all_equiv' present:", "all_equiv" in TARGET.read_text(encoding="utf-8"))
    print("\nNow run:  python analyze_capacity.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
