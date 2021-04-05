# Computer Vision and Language Project
Visual Question Answering(VQA) received in-creasing  attention  in  multi-discipline  Artifi-cial  Intelligence.   Given  an  image  and  ques-tions  in  natural  language. VQA  model  rea-soning  over  visual  cues  of  image  and  com-mon sense knowledge to reply to a correct an-swer.(Singh et al., 2020) VQA task usually re-quires a great amount of knowledge, and trans-fer learning would be a potential solution for it.

This project will demonstrate how can transferlearning models perform in VQA tasks. I fine-tuned various pre-trained models on the down-stream dataset.  The research question in thisreport  is  toinvestigate the performance ofthe different pre-trained models on down-stream datasets.  Additionally, exploring thefuture work based on the result and error analysis.

# Data Preprocessing(utils)

You need to load data from VQA dataset(https://visualqa.org), resize image, and make vocabulary dictionary.

## Train a VQA model 

You need to choose the pre-trained models. The default VQA model is built by VGG16 and BERT.

```bash
python train.py --pretrained_model vgg16 --bert yes
```

## Citation
@misc{wu2016visual,
      title={Visual Question Answering: A Survey of Methods and Datasets}, 
      author={Qi Wu and Damien Teney and Peng Wang and Chunhua Shen and Anthony Dick and Anton van den Hengel},
      year={2016},
      eprint={1607.05910},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@InProceedings{VQA, 
        author = {Stanislaw Antol and Aishwarya Agrawal and Jiasen Lu and Margaret Mitchell and Dhruv Batra and C. Lawrence Zitnick and Devi Parikh}, 
        title = {{VQA}: {V}isual {Q}uestion {A}nswering}, 
        booktitle = {International Conference on Computer Vision (ICCV)}, 
        year = {2015}, 
}
