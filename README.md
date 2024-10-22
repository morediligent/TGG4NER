# TGG4NER

## 1. Environments

```
- python (3.8.12)
- cuda (11.8)
```
## 2. Dependencies

```
- numpy (1.21.4)
- torch (1.10.0)
- gensim (4.1.2)
- transformers (4.13.0)
- pandas (1.3.4)
- scikit-learn (1.0.1)
- prettytable (2.4.0)
- networkx (3.0)
- pytorch-transformers (1.2.0)
- torch-geometric(2.5.3)
- tqdm (4.66.4)
```

## 3. Dataset

- [OntoNotes 4.0](https://catalog.ldc.upenn.edu/LDC2011T03)
- [ACE 2005](https://catalog.ldc.upenn.edu/LDC2006T06)
- [GENIA](http://www.geniaproject.org/genia-corpus)
- [CADEC](https://pubmed.ncbi.nlm.nih.gov/25817970/)
- [Resume]
- [Weibo]

## 5. Training

```bash
>> python main_w2ner_origin.py --config ./config/example.json
# save the best model in the path of './model_ner/ner_model.pt'
>> python main_n_ration.py --config ./config/example.json
# save the mask model in the path of './model_mask/mask_model.pt'
>> python main_EM_ner.py --config ./config/example.json
# save the iterately training model in the path of './model_EM/EM_model.pt'
