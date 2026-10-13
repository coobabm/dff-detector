import unittest
from unittest.mock import patch
import torch
from PIL import Image
import app

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.conv=torch.nn.Conv2d(3,2,1)
    def forward_features(self,x): return self.conv(x)
    def forward_head(self,x): return x.mean((1,2,3)).unsqueeze(1)
    def forward(self,x): return self.forward_head(self.forward_features(x))

class HeatmapTests(unittest.TestCase):
    def test_cam_does_not_accumulate_parameter_grads(self):
        model=Tiny().to(app.DEVICE).eval(); im=Image.new('RGB',(256,256),'white')
        with torch.no_grad():
            expected=torch.sigmoid(model(app.tf(im).unsqueeze(0).to(app.DEVICE))).item()
            p,overlay=app.grad_cam(model,im,app.tf)
        self.assertAlmostEqual(expected,p,places=6)
        self.assertEqual(overlay.size,(224,224))
        self.assertTrue(all(v.grad is None for v in model.parameters()))
        self.assertFalse(torch.isnan(torch.tensor(p)))

    def test_cnn_with_blocks_is_not_vit(self):
        class CNN(Tiny):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.Sequential(self.conv, torch.nn.Identity())
            def forward_features(self, x): return self.blocks(x)
        model = CNN().to(app.DEVICE).eval()
        score, im = app.grad_cam(model, Image.new('RGB', (256,256), 'white'), app.tf)
        self.assertEqual(im.size, (224,224))
        self.assertTrue(0 <= score <= 1)

    def test_all_models_gallery(self):
        im=Image.new('RGB',(256,256),'white'); cams=[]; errors=[]
        with patch.object(app,'model_specs',return_value=[('test','b0','test')]), patch.object(app,'load',return_value=Tiny().to(app.DEVICE).eval()):
            rows=app.score_all([im],[], 'zh', heatmaps=cams, cam_errors=errors, representatives={'b0':im}, frame_label='image')
        self.assertEqual(len(cams),1); self.assertEqual(errors,[])
        self.assertIn('test',cams[0][1]); self.assertIn('0.500',cams[0][1])
        self.assertEqual(len(rows),1)

    def test_heatmap_failure_preserves_prediction(self):
        im=Image.new('RGB',(256,256),'white'); cams=[]; errors=[]
        with patch.object(app,'model_specs',return_value=[('test','b0','test')]), patch.object(app,'load',return_value=Tiny().to(app.DEVICE).eval()), patch.object(app,'grad_cam',side_effect=RuntimeError('test failure')):
            rows=app.score_all([im],[], 'zh', heatmaps=cams, cam_errors=errors, representatives={'b0':im})
        self.assertIsNotNone(rows[0][1]); self.assertEqual(cams,[]); self.assertEqual(errors,['test'])

    def test_empty_auto_hides_gallery(self):
        self.assertFalse(app.analyze_any_ui(None)[1]['visible'])

if __name__=='__main__': unittest.main()
