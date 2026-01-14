import tempfile
from pathlib import Path
import unittest

from utils.config_loader import ConfigLoader


class ConfigLoaderTests(unittest.TestCase):
    def test_loads_defaults_when_config_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yml"
            config_path.write_text("\n", encoding="utf-8")

            config = ConfigLoader(config_path).load()
            self.assertEqual(config.sample_rate, 24000)
            self.assertEqual(config.output_dir, "outputs")
            self.assertTrue(config.use_llm)
            self.assertEqual(config.llm_module_dir, "llm")
            self.assertIsInstance(config.llm_settings, dict)
            self.assertEqual(config.stt_model, "e2e_rnnt")
            self.assertEqual(config.stt_device, "auto")
            self.assertEqual(config.orpheus_model, "papacliff/orpheus-3b-0.1-ft-ru")
            self.assertEqual(config.orpheus_device, "auto")
            self.assertEqual(config.orpheus_dtype, "float16")

    def test_missing_config_raises(self) -> None:
        loader = ConfigLoader("missing.yml")
        with self.assertRaises(FileNotFoundError):
            loader.load()

    def test_invalid_sample_rate_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yml"
            config_path.write_text("sample_rate: -1\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                ConfigLoader(config_path).load()

    def test_custom_values_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yml"
            config_path.write_text(
                (
                    "sample_rate: 48000\noutput_dir: custom\n"
                    "orpheus_model: custom/orpheus\norpheus_device: cuda\norpheus_dtype: float32\n"
                    "max_reference_seconds: 12.5\nuse_llm: false\n"
                ),
                encoding="utf-8",
            )

            config = ConfigLoader(config_path).load()
            self.assertEqual(config.sample_rate, 48000)
            self.assertEqual(config.output_dir, "custom")
            self.assertFalse(config.use_llm)
            self.assertEqual(config.llm_module_dir, "llm")
            self.assertIsInstance(config.llm_settings, dict)
            self.assertEqual(config.stt_model, "e2e_rnnt")
            self.assertEqual(config.stt_device, "auto")
            self.assertEqual(config.orpheus_model, "custom/orpheus")
            self.assertEqual(config.orpheus_device, "cuda")
            self.assertEqual(config.orpheus_dtype, "float32")
            self.assertAlmostEqual(config.max_reference_seconds, 12.5)

    def test_llm_fields_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Create LLM settings file
            llm_settings_path = Path(tmp) / "llm_settings.yml"
            llm_settings_path.write_text(
                "model: my-model\nbase_url: http://localhost:11434\n",
                encoding="utf-8",
            )

            config_path = Path(tmp) / "config.yml"
            config_path.write_text(
                (
                    "sample_rate: 22050\noutput_dir: out\n"
                    "use_llm: true\n"
                    f"llm_module_dir: /tmp\nllm_settings_path: {llm_settings_path}\n"
                    "orpheus_model: papacliff/orpheus-3b-0.1-ft-ru\n"
                ),
                encoding="utf-8",
            )

            config = ConfigLoader(config_path).load()
            self.assertTrue(config.use_llm)
            self.assertEqual(config.llm_module_dir, "/tmp")
            self.assertIsInstance(config.llm_settings, dict)
            self.assertEqual(config.llm_settings.get("model"), "my-model")
            self.assertEqual(config.llm_settings.get("base_url"), "http://localhost:11434")
            self.assertEqual(config.stt_model, "e2e_rnnt")
            self.assertEqual(config.stt_device, "auto")


if __name__ == "__main__":
    unittest.main()
