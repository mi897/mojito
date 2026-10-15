"""The website's data build, and that every number bound on a page exists in the results."""
import glob
import json
import os
import re
import sys
import tempfile
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "scripts", "website"))
import build_site  # noqa: E402


def resolve(results, path):
    node = results
    for key in path.split("/" if "/" in path else "."):
        node = node[int(key)] if isinstance(node, list) else node[key]
    return node


class TestBuildSite(unittest.TestCase):
    def test_build_writes_data_and_media(self):
        with tempfile.TemporaryDirectory() as out:
            build_site.build(out)
            for name in ("results.js", "docs.js", "build.js"):
                self.assertTrue(os.path.getsize(os.path.join(out, "data", name)) > 10, name)
            with open(os.path.join(out, "data", "results.js")) as f:
                text = f.read()
            self.assertTrue(text.startswith("window.MOJITO_RESULTS="))
            data = json.loads(text[len("window.MOJITO_RESULTS="):].rstrip().rstrip(";"))
            self.assertIn("metrics", data)
            self.assertTrue(os.path.exists(os.path.join(out, "media", "leg_demo.html")))

    def test_every_bound_number_resolves_to_a_number(self):
        results = {}
        for name in build_site.RESULT_FILES:
            with open(build_site.find_result(os.path.join(ROOT, "results"), name + ".json")) as f:
                results[name] = json.load(f)
        bound = 0
        for page in glob.glob(os.path.join(ROOT, "site", "*.html")):
            with open(page) as f:
                for path in re.findall(r'data-bind="([^"]+)"', f.read()):
                    value = resolve(results, path)
                    self.assertIsInstance(value, (int, float), f"{os.path.basename(page)}: {path}")
                    bound += 1
        self.assertGreaterEqual(bound, 25)


if __name__ == "__main__":
    unittest.main()
