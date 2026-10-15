"""Every page carries the Open Graph, Twitter and icon tags, and the images they name exist (preview 1200x630)."""
import glob
import os
import re
import struct
import unittest

SITE = os.path.join(os.path.dirname(__file__), "..", "..", "site")
BASE = "https://mi8.studio/mojito/"


class TestSocialPreview(unittest.TestCase):
    def test_every_page_has_absolute_preview_tags(self):
        pages = glob.glob(os.path.join(SITE, "*.html"))
        self.assertTrue(pages)
        for page in pages:
            with open(page) as f:
                head = f.read().split("</head>")[0]
            for prop in ("og:title", "og:description", "og:url", "og:image"):
                self.assertRegex(head, r'<meta property="%s" content="[^"]+">' % prop, (page, prop))
            image = re.search(r'property="og:image" content="([^"]+)"', head).group(1)
            self.assertEqual(image, BASE + "assets/og-image.png", page)
            self.assertIn('<meta name="twitter:card" content="summary_large_image">', head, page)

    def test_every_page_links_icons_that_exist(self):
        for page in glob.glob(os.path.join(SITE, "*.html")):
            with open(page) as f:
                head = f.read().split("</head>")[0]
            hrefs = re.findall(r'<link rel="(?:icon|apple-touch-icon)" href="([^"]+)"', head)
            self.assertEqual(len(hrefs), 3, page)
            for href in hrefs:
                self.assertTrue(os.path.exists(os.path.join(SITE, href)), (page, href))

    def test_image_is_1200_by_630_png(self):
        with open(os.path.join(SITE, "assets", "og-image.png"), "rb") as f:
            header = f.read(24)
        self.assertEqual(header[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", header[16:24]), (1200, 630))


if __name__ == "__main__":
    unittest.main()
