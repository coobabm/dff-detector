import unittest
import app

class TrainingHintTests(unittest.TestCase):
    def test_highest_score_not_first_model(self):
        result=app.all_report([('runs/t2i',.92,.5,1),('runs_swap/b4',.98,.27,1),('runs/all',.1,.5,1)],'zh')
        self.assertIn('疑似 AI 换脸',result)
        self.assertIn('最高分模型',result)
        self.assertIn('模型意见不一致',result)
    def test_strict_boundary(self):
        for p in [.89,.9]:
            self.assertNotIn('疑似 AI 换脸',app.all_report([('runs_swap/b4',p,.27,1)],'zh'))
        self.assertIn('疑似 AI 换脸',app.all_report([('runs_swap/b4',.90001,.27,1)],'zh'))
    def test_mixed_training_not_specific_generator(self):
        result=app.all_report([('runs/all_image',.99,.5,1)],'zh')
        self.assertIn('混合类型',result)
        self.assertIn('不能据此确定具体生成工具',result)
    def test_tie_and_failure(self):
        result=app.all_report([('runs/t2i',.99,.5,1),('runs_swap/b4',.99,.27,1),('broken',None,None,0)],'zh')
        self.assertIn('文生图',result);self.assertIn('换脸',result);self.assertIn('检测不完整',result)
    def test_languages(self):
        for lang,phrase in [('en','Suspected'),('ko','의심')]:
            self.assertIn(phrase,app.all_report([('runs_swap/b4',.99,.27,1)],lang))

if __name__=='__main__': unittest.main()
