# Data lists

Real photos are not redistributed here (licences / privacy). These lists name exactly
which images were used, so `fetch_real.py` / `prepare.py` can be checked or re-run.

| File | Source | Entry format |
|---|---|---|
| `dff_manifest.csv` | [DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace) | path, source, label (0 real / 1 fake), split |
| `real_ffhq.txt` | [bitmind/ffhq-256](https://huggingface.co/datasets/bitmind/ffhq-256) `train-00000-of-00016.parquet` | row index |
| `real_pexels.txt` | [gaunernst/pexels-portrait](https://huggingface.co/datasets/gaunernst/pexels-portrait) | Pexels photo id |
| `real_ppr10k.txt` | [JarvisArt/MMArt-PPR10k](https://huggingface.co/datasets/JarvisArt/MMArt-PPR10k) | sample folder (`global/<id>/before.jpg`) |
| `real_flickr30k.txt` | [nlphuji/flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k) | image filename stem |
| `real_easyportrait.txt` | [gofixyourself/EasyPortrait](https://huggingface.co/datasets/gofixyourself/EasyPortrait) `data/test.zip` | image filename stem |

The OpenAI-generated faces are in the GitHub Release (`openai_faces_*.zip`).
