import threading
import time
import unittest
from unittest.mock import patch
import app

class ParallelTests(unittest.TestCase):
    def test_bfree_disabled(self):
        self.assertNotIn('bfree', [s[2] for s in app.model_specs('image')])

    def test_three_workers_finish_order_and_no_heatmap_wait(self):
        barrier=threading.Barrier(3)
        specs=[('slow','b0','slow'),('fast','b0','fast'),('medium','b0','medium')]
        release=threading.Event()
        def fake(loose,tight,lang,**kw):
            name=kw['specs'][0][0];barrier.wait(timeout=3)
            time.sleep({'fast':.01,'medium':.07,'slow':.12}[name])
            yield [(name,.6,.5,1)]
            release.wait(timeout=3)
            kw['heatmaps'].append(('image',name))
        with patch.object(app,'_iter_scores_single',side_effect=fake):
            maps=[];steps=app.iter_scores([],[],'zh',specs=specs,heatmaps=maps)
            try:
                first=next(steps)
                self.assertEqual(first[0][0],'fast');self.assertFalse(release.is_set())
            finally: release.set()
            results=list(steps)
        self.assertEqual([r[0] for r in results[-1]],['fast','medium','slow'])
        self.assertEqual(len(maps),3)

    def test_failure_does_not_block_other_models(self):
        def fake(a,b,lang,**kw):
            name=kw['specs'][0][0]
            if name=='broken': raise RuntimeError('test')
            yield [(name,.4,.5,1)]
        with patch.object(app,'_iter_scores_single',side_effect=fake):
            rows=list(app.iter_scores([],[],'zh',specs=[('broken','b0','x'),('ok','b0','y')]))[-1]
        self.assertEqual(len(rows),2)
        self.assertEqual(sum(r[1] is None for r in rows),1)

    def test_no_artificial_pacing(self):
        self.assertEqual(app.PACE,0)

if __name__=='__main__':unittest.main()
