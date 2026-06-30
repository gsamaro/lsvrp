import unittest


class MainImportTestCase(unittest.TestCase):
    def test_main_imports_without_missing_helper_modules(self):
        import main  # noqa: F401


if __name__ == "__main__":
    unittest.main()
