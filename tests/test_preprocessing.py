import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from IA.POC2PROD.StackOverflow import clean_text, load_dataset, keep_primary_tag

class TestPreprocessing(unittest.TestCase):

    def test_clean_text(self):
        """Test Unitaire pur : vérifie que le nettoyage fonctionne"""
        raw_text = "Hello   World! 123"
        # Ta fonction attendu : minuscules, pas de chiffres, lemmatisation
        expected = "hello world"
        # Note: adapte 'expected' selon ta vraie logique de clean_text (ex: si tu gardes les chiffres ou pas)
        self.assertEqual(clean_text(raw_text).strip(), expected)

    @patch("StackOverflow.pd.read_csv")
    def test_load_dataset_mocked(self, mock_read_csv):
        """
        Test avec Mock : on fait croire à load_dataset que le CSV
        ne contient que 2 lignes pour vérifier qu'il ne plante pas.
        """
        # 1. On crée le faux DataFrame que read_csv va renvoyer
        mock_df = pd.DataFrame({
            "post_id": [1, 2],
            "title": ["Test Python", "Test Java"],
            "tag_name": ["python", "java"],
            "tag_id": [10, 20],
            "tag_position": [0, 0]
        })
        mock_read_csv.return_value = mock_df

        # 2. On appelle ta vraie fonction
        df = load_dataset("chemin_bidon.csv")

        # 3. Vérifications
        self.assertEqual(len(df), 2)
        self.assertTrue("post_id" in df.columns)
        # Vérifie que pd.read_csv a bien été appelé une fois
        mock_read_csv.assert_called_once()