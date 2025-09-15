---
base_model: distilbert-base-uncased
license: apache-2.0
language:
- en
metrics:
- f1
- accuracy
- precision
- recall
pipeline_tag: text-classification
tags:
- Transformers
- ' PyTorch'
- safety
- innapropriate
- distilbert
- DiffGuard
datasets:
- eliasalbouzidi/NSFW-Safe-Dataset
model-index:
- name: NSFW-Safe-Dataset
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: NSFW-Safe-Dataset
      type: .
    metrics:
    - name: F1
      type: f1
      value: 0.974
    - name: Accuracy
      type: accuracy
      value: 0.98
---

# Model Card

[![Open on Hugging Face](https://img.shields.io/badge/Open%20on-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/eliasalbouzidi/distilbert-nsfw-text-classifier)

This model is designed to categorize text into two classes: "safe", or "nsfw" (not safe for work), which makes it suitable for content moderation and filtering applications.

The model was trained using a dataset containing 190,000 labeled text samples, distributed among the two classes of "safe" and "nsfw".

The model is based on the Distilbert-base model. 

In terms of performance, the model has achieved a score of 0.974 for F1 (40K exemples).

To improve the performance of the model, it is necessary to preprocess the input text. You can refer to the preprocess function in the app.py file in the following space: <https://huggingface.co/spaces/eliasalbouzidi/distilbert-nsfw-text-classifier>.
### Model Description

The model can be used directly to classify text into one of the two classes. It takes in a string of text as input and outputs a probability distribution over the two classes. The class with the highest probability is selected as the predicted class.


- **Developed by:** Elias Al Bouzidi, Massine El Khader, Abdellah Oumida, Mohammed Sbaihi, Eliott Binard
- **Model type:** 60M
- **Language (NLP):** English
- **License:** apache-2.0

## Technical Paper:

A more detailed technical overview of the model and the dataset can be found [here](https://arxiv.org/pdf/2412.00064).

### Uses

The model can be integrated into larger systems for content moderation or filtering.
### Training Data
The training data for finetuning the text classification model consists of a large corpus of text labeled with one of the two classes: "safe" and "nsfw". The dataset contains a total of 190,000 examples, which are distributed as follows:

117,000 examples labeled as "safe"

63,000 examples labeled as "nsfw"

It was assembled by scraping data from the web and utilizing existing open-source datasets. A significant portion of the dataset consists of descriptions for images and scenes. The primary objective was to prevent diffusers from generating NSFW content but it can be used for other moderation purposes.

You can access the dataset : https://huggingface.co/datasets/eliasalbouzidi/NSFW-Safe-Dataset
### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 600
- num_epochs: 3
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step  | Validation Loss | Accuracy | F1     | Fbeta 1.6 | False positive rate | False negative rate | Precision | Recall |
|:-------------:|:------:|:-----:|:---------------:|:--------:|:------:|:---------:|:-------------------:|:-------------------:|:---------:|:------:|
| 0.3367        | 0.0998 | 586   | 0.1227          | 0.9586   | 0.9448 | 0.9447    | 0.0331              | 0.0554              | 0.9450    | 0.9446 |
| 0.0998        | 0.1997 | 1172  | 0.0919          | 0.9705   | 0.9606 | 0.9595    | 0.0221              | 0.0419              | 0.9631    | 0.9581 |
| 0.0896        | 0.2995 | 1758  | 0.0900          | 0.9730   | 0.9638 | 0.9600    | 0.0163              | 0.0448              | 0.9724    | 0.9552 |
| 0.087         | 0.3994 | 2344  | 0.0820          | 0.9743   | 0.9657 | 0.9646    | 0.0191              | 0.0367              | 0.9681    | 0.9633 |
| 0.0806        | 0.4992 | 2930  | 0.0717          | 0.9752   | 0.9672 | 0.9713    | 0.0256              | 0.0235              | 0.9582    | 0.9765 |
| 0.0741        | 0.5991 | 3516  | 0.0741          | 0.9753   | 0.9674 | 0.9712    | 0.0251              | 0.0240              | 0.9589    | 0.9760 |
| 0.0747        | 0.6989 | 4102  | 0.0689          | 0.9773   | 0.9697 | 0.9696    | 0.0181              | 0.0305              | 0.9699    | 0.9695 |
| 0.0707        | 0.7988 | 4688  | 0.0738          | 0.9781   | 0.9706 | 0.9678    | 0.0137              | 0.0356              | 0.9769    | 0.9644 |
| 0.0644        | 0.8986 | 5274  | 0.0682          | 0.9796   | 0.9728 | 0.9708    | 0.0135              | 0.0317              | 0.9773    | 0.9683 |
| 0.0688        | 0.9985 | 5860  | 0.0658          | 0.9798   | 0.9730 | 0.9718    | 0.0144              | 0.0298              | 0.9758    | 0.9702 |
| 0.0462        | 1.0983 | 6446  | 0.0682          | 0.9800   | 0.9733 | 0.9723    | 0.0146              | 0.0290              | 0.9756    | 0.9710 |
| 0.0498        | 1.1982 | 7032  | 0.0706          | 0.9800   | 0.9733 | 0.9717    | 0.0138              | 0.0303              | 0.9768    | 0.9697 |
| 0.0484        | 1.2980 | 7618  | 0.0773          | 0.9797   | 0.9728 | 0.9696    | 0.0117              | 0.0345              | 0.9802    | 0.9655 |
| 0.0483        | 1.3979 | 8204  | 0.0676          | 0.9800   | 0.9734 | 0.9742    | 0.0172              | 0.0248              | 0.9715    | 0.9752 |
| 0.0481        | 1.4977 | 8790  | 0.0678          | 0.9798   | 0.9731 | 0.9737    | 0.0170              | 0.0255              | 0.9717    | 0.9745 |
| 0.0474        | 1.5975 | 9376  | 0.0665          | 0.9782   | 0.9713 | 0.9755    | 0.0234              | 0.0191              | 0.9618    | 0.9809 |
| 0.0432        | 1.6974 | 9962  | 0.0691          | 0.9787   | 0.9718 | 0.9748    | 0.0213              | 0.0213              | 0.9651    | 0.9787 |
| 0.0439        | 1.7972 | 10548 | 0.0683          | 0.9811   | 0.9748 | 0.9747    | 0.0150              | 0.0254              | 0.9750    | 0.9746 |
| 0.0442        | 1.8971 | 11134 | 0.0710          | 0.9809   | 0.9744 | 0.9719    | 0.0118              | 0.0313              | 0.9802    | 0.9687 |
| 0.0425        | 1.9969 | 11720 | 0.0671          | 0.9810   | 0.9747 | 0.9756    | 0.0165              | 0.0232              | 0.9726    | 0.9768 |
| 0.0299        | 2.0968 | 12306 | 0.0723          | 0.9802   | 0.9738 | 0.9758    | 0.0187              | 0.0217              | 0.9692    | 0.9783 |
| 0.0312        | 2.1966 | 12892 | 0.0790          | 0.9804   | 0.9738 | 0.9731    | 0.0146              | 0.0279              | 0.9755    | 0.9721 |
| 0.0266        | 2.2965 | 13478 | 0.0840          | 0.9815   | 0.9752 | 0.9728    | 0.0115              | 0.0302              | 0.9806    | 0.9698 |
| 0.0277        | 2.3963 | 14064 | 0.0742          | 0.9808   | 0.9746 | 0.9770    | 0.0188              | 0.0199              | 0.9690    | 0.9801 |
| 0.0294        | 2.4962 | 14650 | 0.0764          | 0.9809   | 0.9747 | 0.9765    | 0.0179              | 0.0211              | 0.9705    | 0.9789 |
| 0.0304        | 2.5960 | 15236 | 0.0795          | 0.9811   | 0.9748 | 0.9742    | 0.0142              | 0.0266              | 0.9763    | 0.9734 |
| 0.0287        | 2.6959 | 15822 | 0.0783          | 0.9814   | 0.9751 | 0.9741    | 0.0134              | 0.0272              | 0.9775    | 0.9728 |
| 0.0267        | 2.7957 | 16408 | 0.0805          | 0.9814   | 0.9751 | 0.9740    | 0.0133              | 0.0274              | 0.9777    | 0.9726 |
| 0.0318        | 2.8956 | 16994 | 0.0767          | 0.9814   | 0.9752 | 0.9756    | 0.0154              | 0.0240              | 0.9744    | 0.9760 |
| 0.0305        | 2.9954 | 17580 | 0.0779          | 0.9815   | 0.9753 | 0.9751    | 0.0146              | 0.0251              | 0.9757    | 0.9749 |

We selected the checkpoint with the highest F-beta1.6 score.

### Framework versions

- Transformers 4.40.1
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1


### Out-of-Scope Use

It should not be used for any illegal activities.

## Bias, Risks, and Limitations

The model may exhibit biases based on the training data used. It may not perform well on text that is written in languages other than English. It may also struggle with sarcasm, irony, or other forms of figurative language. The model may produce false positives or false negatives, which could lead to incorrect categorization of text.

### Recommendations


Users should be aware of the limitations and biases of the model and use it accordingly. They should also be prepared to handle false positives and false negatives. It is recommended to fine-tune the model for specific downstream tasks and to evaluate its performance on relevant datasets.



### Load model directly
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("eliasalbouzidi/distilbert-nsfw-text-classifier")

model = AutoModelForSequenceClassification.from_pretrained("eliasalbouzidi/distilbert-nsfw-text-classifier")

```
### Use a pipeline 
```python
from transformers import pipeline

pipe = pipeline("text-classification", model="eliasalbouzidi/distilbert-nsfw-text-classifier")
```

## Citation

If you find our work useful, please consider citing us!

```bibtex
@misc{khader2025diffguardtextbasedsafetychecker,
      title={DiffGuard: Text-Based Safety Checker for Diffusion Models}, 
      author={Massine El Khader and Elias Al Bouzidi and Abdellah Oumida and Mohammed Sbaihi and Eliott Binard and Jean-Philippe Poli and Wassila Ouerdane and Boussad Addad and Katarzyna Kapusta},
      year={2025},
      eprint={2412.00064},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.00064}, 
}
```


## Contact
Please reach out to eliasalbouzidi@gmail.com if you have any questions or feedback.
