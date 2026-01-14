import tempfile
import unittest
from pathlib import Path

from utils.config_loader import ConfigLoader
from utils.file_handler import FileHandler
from utils.interfaces import IConfigLoader, IFileHandler, ITTSEngine
from utils.orpheus_engine import OrpheusTTSEngine


class IntegrationImportsTests(unittest.TestCase):
    def test_components_instantiate_via_interfaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yml"
            config_path.write_text("sample_rate: 24000\n", encoding="utf-8")

            loader: IConfigLoader = ConfigLoader(config_path)
            config = loader.load()

            tts_engine: ITTSEngine = OrpheusTTSEngine(
                model_name=config.orpheus_model,
                codec_name=config.orpheus_codec,
                device=config.orpheus_device,
                dtype=config.orpheus_dtype,
                sample_rate=config.sample_rate,
            )
            file_handler: IFileHandler = FileHandler(target_rate=config.sample_rate)

            self.assertIsInstance(loader, IConfigLoader)
            self.assertIsInstance(tts_engine, ITTSEngine)
            self.assertIsInstance(file_handler, IFileHandler)


if __name__ == "__main__":
    unittest.main()
