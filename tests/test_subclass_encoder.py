import unittest
import numpy as np
from classification.subclass_encoder import PTBXLSubclassLabelEncoder, PTBXL_SUBCLASSES

class TestPTBXLSubclassLabelEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = PTBXLSubclassLabelEncoder()

    def test_encode_decode(self):
        # Multi-label test
        targets = ["NORM", "IMI"]
        vector = self.encoder.encode(targets)
        self.assertEqual(vector.shape, (len(PTBXL_SUBCLASSES),))
        self.assertEqual(vector[PTBXL_SUBCLASSES.index("NORM")], 1.0)
        self.assertEqual(vector[PTBXL_SUBCLASSES.index("IMI")], 1.0)
        self.assertEqual(vector[PTBXL_SUBCLASSES.index("AMI")], 0.0)

        decoded = self.encoder.decode(vector, threshold=0.5)
        self.assertIn("NORM", decoded)
        self.assertIn("IMI", decoded)
        self.assertNotIn("AMI", decoded)

    def test_ignore_unknown_subclass(self):
        targets = ["NORM", "UNKNOWN_SUBCLASS"]
        vector = self.encoder.encode(targets)
        decoded = self.encoder.decode(vector, threshold=0.5)
        self.assertIn("NORM", decoded)
        self.assertNotIn("UNKNOWN_SUBCLASS", decoded)

if __name__ == "__main__":
    unittest.main()
