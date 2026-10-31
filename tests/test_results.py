"""mojito.results: the provenance block, paths from the repository root, and that the scripts use them.

Run with:  python -m unittest discover tests
"""
import datetime
import glob
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mojito import results  # noqa: E402

HERE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=HERE_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class TestWriteResult(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def write(self, payload, **options):
        path = os.path.join(self.directory.name, "nested", "folder", "result.json")
        self.assertEqual(results.write_result(path, payload, **options), path)
        with open(path) as f:
            return path, f.read()

    def test_payload_is_kept_and_provenance_added(self):
        payload = {"what": "a test", "numbers": [1, 2.5], "nested": {"a": True}}
        before = json.dumps(payload)
        _, text = self.write(payload)
        written = json.loads(text)
        self.assertEqual(json.dumps(payload), before, "the caller's payload must not change")
        numbers, record = results.without_provenance(written)
        self.assertEqual(numbers, payload)
        self.assertEqual(list(written)[-1], "provenance")
        for key in ("git_commit", "git_dirty", "script", "argv", "python", "numpy", "mujoco", "utc"):
            self.assertIn(key, record)

    def test_provenance_against_independent_sources(self):
        _, text = self.write({"x": 1})
        record = json.loads(text)["provenance"]
        # git, asked directly here rather than through the helper
        self.assertEqual(record["git_commit"], git("rev-parse", "HEAD"))
        status = git("status", "--porcelain", "--untracked-files=no")
        if status is not None:
            self.assertEqual(record["git_dirty"], bool(status))
        self.assertEqual(record["python"], platform.python_version())
        import numpy
        self.assertEqual(record["numpy"], numpy.__version__)
        self.assertEqual(record["argv"], sys.argv[1:])
        when = datetime.datetime.strptime(record["utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
        self.assertLess(abs((datetime.datetime.now(datetime.timezone.utc) - when).total_seconds()), 120)

    def test_an_old_provenance_block_is_replaced(self):
        _, text = self.write({"x": 1, "provenance": {"git_commit": "stale"}})
        record = json.loads(text)["provenance"]
        self.assertNotEqual(record.get("git_commit"), "stale")
        self.assertIn("utc", record)

    def test_formats(self):
        _, compact = self.write({"x": [1, 2]}, compact=True)
        self.assertNotIn(" ", compact.split('"provenance"')[0])
        _, one_line = self.write({"x": [1, 2]}, indent=None)
        self.assertEqual(one_line.count("\n"), 0)
        _, indented = self.write({"x": [1, 2]})
        self.assertIn('\n "x": [\n  1,', indented)

    def test_only_objects(self):
        with self.assertRaises(TypeError):
            results.write_result(os.path.join(self.directory.name, "list.json"), [1, 2])

    def test_without_provenance_leaves_plain_results_alone(self):
        self.assertEqual(results.without_provenance({"a": 1}), ({"a": 1}, None))


class TestPaths(unittest.TestCase):
    def test_root_is_the_repository(self):
        self.assertEqual(os.path.realpath(results.ROOT), os.path.realpath(HERE_ROOT))
        self.assertTrue(os.path.exists(results.repo_path("pyproject.toml")))
        self.assertEqual(results.results_path("gait", "x.json"), os.path.join(results.ROOT, "results", "gait", "x.json"))

    def test_paths_do_not_depend_on_the_working_directory(self):
        here = os.getcwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            try:
                self.assertTrue(os.path.isdir(results.results_path()))
            finally:
                os.chdir(here)

    def test_relative_to_root(self):
        self.assertEqual(results.relative_to_root(results.repo_path("weights", "ik_leg.npz")), os.path.join("weights", "ik_leg.npz"))
        self.assertEqual(results.relative_to_root("/somewhere/else.json"), "/somewhere/else.json")

    def test_defaults_of(self):
        def run(a, b=2, *, c=(1, 2), **rest):
            return a
        self.assertEqual(results.defaults_of(run), {"b": 2, "c": (1, 2)})


class TestScriptsUseTheHelpers(unittest.TestCase):
    """Every script writes results through write_result and imports the package without a sys.path line."""

    SCRIPTS = sorted(glob.glob(os.path.join(HERE_ROOT, "scripts", "**", "*.py"), recursive=True))

    def test_no_script_writes_json_directly(self):
        not_results = {os.path.join("scripts", "website", "build_site.py")}  # writes site/data/*.js, not results
        offenders = [os.path.relpath(p, HERE_ROOT) for p in self.SCRIPTS if re.search(r"\bjson\.dump\(", open(p).read())]
        offenders = [p for p in offenders if p not in not_results]
        self.assertEqual(offenders, [], "write results with mojito.results.write_result")

    def test_no_script_puts_the_repository_on_sys_path(self):
        pattern = re.compile(r'sys\.path\.insert\(0, os\.path\.join\(os\.path\.dirname\(__file__\)(, "\.\.")+\)\)')
        offenders = [os.path.relpath(p, HERE_ROOT) for p in self.SCRIPTS if pattern.search(open(p).read())]
        self.assertEqual(offenders, [], "the package is installed by `uv sync`; no sys.path line is needed")

    def test_no_script_defines_its_own_palette(self):
        pattern = re.compile(r'^(?:BLUE|ORANGE|AQUA|INK|GREY)\b[^=\n]*= *"#', re.M)
        files = self.SCRIPTS + [os.path.join(HERE_ROOT, "mojito", "sim", "gait_plots.py")]
        offenders = [os.path.relpath(p, HERE_ROOT) for p in files if pattern.search(open(p).read())]
        self.assertEqual(offenders, [], "import the colours from mojito.palette")


if __name__ == "__main__":
    unittest.main()
