import unittest


class PublicExportTests(unittest.TestCase):
    def test_encoding_model_is_exported_from_models_and_package(self):
        from psm_final import EncodingModel as PackageEncodingModel
        from psm_final.models import EncodingModel as ModelsEncodingModel

        self.assertIs(PackageEncodingModel, ModelsEncodingModel)


if __name__ == "__main__":
    unittest.main()
