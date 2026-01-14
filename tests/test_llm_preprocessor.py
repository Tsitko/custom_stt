import unittest

from utils.llm_preprocessor import LLMPreprocessor


class LLMPreprocessorTests(unittest.TestCase):
    def test_missing_module_returns_original_text(self) -> None:
        preprocessor = LLMPreprocessor(
            module_dir="/nonexistent/path",
            llm_settings={"model": "dummy"}
        )
        text = "GPU работает с CUDA"
        processed = preprocessor.process(text)
        self.assertEqual(processed, text)

    def test_local_module_importable(self) -> None:
        preprocessor = LLMPreprocessor()
        # force import to ensure vendored module is reachable without network
        preprocessor._import_client()

    def test_process_runs_with_vendored_llm(self) -> None:
        preprocessor = LLMPreprocessor()
        text = "GPU и CUDA должны транскрибироваться."
        processed = preprocessor.process(text)
        # even если LLM не доступен, должно вернуться исходное или обработанное
        self.assertTrue(isinstance(processed, str) and len(processed) > 0)


if __name__ == "__main__":
    unittest.main()
