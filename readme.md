# Chinese Sentence Matching

Final accuracy: 95.1%.  

The model with the highest accuracy used `Roberta-chinese-pair-large` as pretrained model. Since there were quite a few medical related sentences, seperating medical sentences with non-medical sentences for model training improved the model performance. Tuning hyperparameters were very important. Additionally, this code used stacking method (a model ensemble method), which incorporated the 15 prediction results from 3 models.   

For more details, refer to `spec.md`.  