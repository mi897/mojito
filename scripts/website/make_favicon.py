"""Render the PNG favicons (site/assets/favicon-32.png, apple-touch-icon.png) from site/assets/favicon.svg.

The SVG is the master and matches the logo in the page header (site.js). Needs a Chromium-family browser on PATH.
"""
import os
import shutil
import subprocess
import sys
import tempfile

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "site", "assets"))
SIZES = {"favicon-32.png": 32, "apple-touch-icon.png": 180}


def main():
    browser = next((p for p in map(shutil.which, ("google-chrome", "chromium", "chromium-browser", "chrome")) if p), None)
    if not browser:
        sys.exit("No Chromium-family browser found on PATH.")
    svg = os.path.join(ASSETS, "favicon.svg")
    with tempfile.TemporaryDirectory() as tmp:
        for name, size in SIZES.items():
            page = os.path.join(tmp, "icon.html")
            with open(page, "w") as f:
                f.write(f'<!doctype html><body style="margin:0"><img src="file://{svg}" width="{size}" height="{size}" style="display:block">')
            subprocess.run([browser, "--headless=new", "--no-sandbox", "--allow-file-access-from-files", f"--window-size={size},{size}",
                            "--hide-scrollbars", "--default-background-color=00000000", f"--screenshot={os.path.join(ASSETS, name)}",
                            f"file://{page}"], check=True, capture_output=True)
            print("wrote", os.path.join(ASSETS, name))


if __name__ == "__main__":
    main()
