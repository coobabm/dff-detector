import unittest
from unittest.mock import patch
import torch
from PIL import Image
import app

class Fixed:
    def __init__(self, p): self.p = p
    def __call__(self, x): return torch.logit(torch.tensor([[self.p]]))

class AllModelsTest(unittest.TestCase):
    def test_all_models_continue_after_failure(self):
        specs = [('low', 'b0', 'low'), ('broken', 'b0', 'broken'), ('high', 'b4', 'high')]
        def load(name):
            if name == 'broken': raise ValueError('bad checkpoint')
            return Fixed(.1)
        with patch.object(app, 'model_specs', return_value=specs), patch.object(app, 'load', side_effect=load), patch.object(app, 'load_b4', return_value=(Fixed(.9), .7)):
            rows = app.score_all([Image.new('RGB', (256,256))], [Image.new('RGB',(380,380))], 'zh')
        self.assertEqual(len(rows), 3)
        self.assertTrue(any(r[0].startswith("broken") and r[1] is None for r in rows))
        high = next(r for r in rows if r[0] == "high")
        self.assertGreater(high[1], high[2])
        result = app.all_report(rows, 'zh')
        self.assertIn('意见不一致', result)
        self.assertIn('检测不完整', result)
        self.assertNotIn('真实照片', result)

    def test_low_scores_are_not_authenticity(self):
        result = app.all_report([('a', .1, .5, 1)], 'zh')
        self.assertIn('不能证明真实', result)
        self.assertNotIn('真实照片', result)

    def test_no_face_preserves_models(self):
        with patch.object(app, 'model_specs', return_value=[('a','b0','a'),('b','b4','b')]), patch.object(app, 'load') as load:
            rows = app.score_all([], [], 'zh')
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r[1] is None for r in rows))
        load.assert_not_called()

    def test_all_languages(self):
        for lang in app.LANGS.values():
            result = app.all_report([('a', .9, .5, 1), ('b', .1, .5, 1)], lang)
            self.assertIn('a</td>', result)
            self.assertIn('b</td>', result)

    def test_registry_includes_versions(self):
        names = [s[0] for s in app.model_specs()]
        self.assertIn('runs/fm_ens' if app.ensemble() else 'runs/' + app.image_model(), names)
        self.assertIn('runs_aivid/v4', names)
        self.assertIn('runs_swap/b4', names)

if __name__ == '__main__': unittest.main()
