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
            with open(os.path.join(out, "data", "build.js")) as f:
                text = f.read()
            self.assertIn("results_provenance", json.loads(text[len("window.MOJITO_BUILD="):].rstrip().rstrip(";")))
            with open(os.path.join(out, "data", "constants.js")) as f:
                text = f.read()
            self.assertTrue(text.startswith("globalThis.MOJITO_CONSTANTS="))
            self.assertEqual(json.loads(text[len("globalThis.MOJITO_CONSTANTS="):].rstrip().rstrip(";")), build_site.site_constants())

    def test_pages_load_the_constants_before_mojito_js(self):
        # mojito.js throws without window.MOJITO_CONSTANTS, so every page that loads it must load data/constants.js first
        for page in glob.glob(os.path.join(ROOT, "site", "*.html")):
            with open(page) as f:
                text = f.read()
            if "assets/mojito.js" in text:
                self.assertIn("data/constants.js", text, page)
                self.assertLess(text.index("data/constants.js"), text.index("assets/mojito.js"), page)

    def test_provenance_is_kept_out_of_the_result(self):
        # A result written by mojito.results.write_result reads back with the same shape as before, the block apart.
        from mojito.results import write_result
        with tempfile.TemporaryDirectory() as directory:
            path = write_result(os.path.join(directory, "metrics.json"), {"learned_varied_legs": {"median_mm": 0.5}})
            numbers, record = build_site.load_result(path)
        self.assertEqual(numbers, {"learned_varied_legs": {"median_mm": 0.5}})
        self.assertEqual(resolve(numbers, "learned_varied_legs.median_mm"), 0.5)
        self.assertIn("git_commit", record)

    def test_every_bound_number_resolves_to_a_number(self):
        results, _, _ = build_site.collect_results(os.path.join(ROOT, "results"))
        self.assertTrue(set(build_site.RESULT_FILES) <= set(results))
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
