# DIPROMATS_2024
This repository contains information concerning the DIPROMATS 2024 task of IberLEF 2024 competition in which we have participated as part of the course Natural Language Processing with Deep Learning.

**DIPROMATS 2024 website**: https://sites.google.com/view/dipromats2024

**IberLEF 2024 website**: https://sites.google.com/view/iberlef-2024/home

## HuggingFace repository
A HuggingFace repository was used to strore the datasets that were used as well as the fine-tuned models. 

**HuggingFace repository URL**: https://huggingface.co/UC3M-LCPM

## Datasets
This folder contains all files regarding the datasets that were used in the competition, including the original training and testing datasets that were provided and the Python notebooks that were used to format them.

- **dipromats24_t1_train_en.json**: Original training dataset with instances in English.
  
- **dipromats24_t1_train_es.json**: Original training dataset with instances in Spanish.
  
- **dipromats24_t1_test_en.json**: Original testing dataset with instances in English.
  
- **dipromats24_t1_test_es.json**: Original testing dataset with instances in Spanish.

- **dataset_formatting.ipynb**: Python notebook to format the original datasets in an appropriate format for ease of use with the HuggingFace libraries. It also combines the English and Spanish datasets in order to make a multilingual approach.

- **Data Augmentation**

  This folder contains the notebooks used for balancing and augmenting the datasets.
  - **Data_balance_EN.ipynb**: This Python notebook was used to balance the English training dataset for task1a.
    
  - **Data_balance_ES.ipynb**: This Python notebook was used to balance the Spanish training dataset for task1a.
    
  - **Data_augmentation_EN.ipynb**: This Python notebook nas used to augment the English training dataset.
    
  - **Data_augmentation_ES.ipynb**: This Python notebook nas used to augment the Spanish training dataset.
    
  - **Join_augmented_datasets.ipynb**: This Python notebook was used to join the augmented datasets.

- **Dataset study**

  This folder contains the notebooks in which we have studied the datasets for the different tasks.
  - **data_study_task1a**: In this Python notebook we study the properties of the training dataset focusing on task1a.
    
  - **data_study_task1b**: In this Python notebook we study the properties of the training dataset focusing on task1b. 

## Models
This folder includes the Python notebooks that were used to train the LLMs.

- **EN + ES**

  This folder includes the Python notebooks that were used to train the models in English and in Spanish sepparately.
  
  - **Bert-base-uncased_task1a.ipynb**: This Python notebook was used to fine-tune a BERT-base-uncased model (pre-trained over a dataset in English) with English texts and a BETO-base-uncased model (pre-trained over a dataset in Spanish) with Spanish texts for Task1a.
    
  - **Bert-base-uncased_task1b.ipynb**: This Python notebook was used to fine-tune a BERT-base-uncased model (pre-trained over a dataset in English) with English texts and a BETO-base-uncased model (pre-trained over a dataset in Spanish) with Spanish texts for Task1b.
    
  - **GPT2-base_task1a.ipynb**: This Python notebook was used to fine-tune a GPT-2-base model (pre-trained over a dataset in English) with English texts and a GPT-2-base model (pre-trained over a dataset in Spanish) with Spanish texts for Task1a.
    
  - **GPT2-base_task1b.ipynb**: This Python notebook was used to fine-tune a GPT-2-base model (pre-trained over a dataset in English) with English texts and a GPT-2-base model (pre-trained over a dataset in Spanish) with Spanish texts for Task1b.
    
  - **Roberta-large_task1a.ipynb**: This Python notebook was used to fine-tune a RoBERTa-large model (pre-trained over a dataset in English) with English texts and a RoBERTa-large model (pre-trained over a dataset in Spanish) with Spanish texts for Task1a.
    
  - **Roberta-large_task1b.ipynb**: This Python notebook was used to fine-tune a RoBERTa-large model (pre-trained over a dataset in English) with English texts and a RoBERTa-large model (pre-trained over a dataset in Spanish) with Spanish texts for Task1b.
    
  - **Robertuito-sentiment-analysis_XLNet-base-cased_task1a.ipynb**: This Python notebook was used to fine-tune a XLNet-base-cased model (pre-trained over a dataset in English) with English texts and a RoBERTuito-sentiment-analysis model (pre-trained over a dataset in Spanish) with Spanish texts for Task1a.
    
  - **Robertuito-sentiment-analysis_XLNet-base-case**: This Python notebook was used to fine-tune a XLNet-base-cased model (pre-trained over a dataset in English) with English texts and a RoBERTuito-sentiment-analysis model (pre-trained over a dataset in Spanish) with Spanish texts for Task1b.
    
- **Multilinüe**

   This folder includes the Python notebooks that were used to train the multilingual models (English and Spanish).
  
  - **task1_RoBERTa_base_en_es_emoji.ipynb**: This Python notebook was used to fine-tune a RoBERTa-base model pre-trained over a dataset with texts in Enslish and in Spanish containing emojis for task1a.
    
  - **twitter-XLM-roBERTa-base-sentiment-analysis_task1a_ML.ipynb**: This Python notebook was used to fine-tune a XLM-roBERTa-base model pre-trained over a dataset with texts in Enslish and in Spanish for task1a.
    
  - **twitter-XLM-roBERTa-base-sentiment-analysis_task1b_ML.ipynb**: This Python notebook was used to fine-tune a XLM-roBERTa-base model pre-trained over a dataset with texts in Enslish and in Spanish for task1b.
    
- **Tests**

  This folder includes the Python notebooks that were used to evaluate the models and their results.
  
  - **Testing_task1a.ipynb**: This Python notebook was used to test the fine-tuned models for task1a.
    
  - **Testing_task1b.ipynb**: This Python notebook was used to test the fine-tuned models for task1b.
    
  - **final_results.xlsx**: This Excel file contains the metrics of the fine-tuned models which were used as criteria to select the models for each of the rounds. The file also contains which models were selected for which round and the threshold configuration of the models for task1b.
    
## Runs
This folder includes all the runs that were sent to the DIPROMATS 2024 competition.

- **UC3M-LCPM-DIPROMATS2024-R1.json**. For Task1a predictions in English we have used a RoBERTa-large model fine-tuned using an augmented version of the English train dataset. For predictions in Spanish for Task1a we have used a sentiment analysis version of the Robertuito model fine-tuned with the original Spanish train dataset. For Task1b in English we have used XLNet-base model fine-tuned with a balanced version of the English train dataset. For Task1b in Spanish we have used a RoBERTa-large model fine-tuned with the original Spanish train dataset.
  
- **UC3M-LCPM-DIPROMATS2024-R2.json**. For Task1a predictions in English we have used a XLNet-Base-cased model fine-tuned using an augmented version of the English train dataset. For predictions in Spanish for Task1a we have used a sentiment analysis version of the Robertuito model fine-tuned using an augmented version of the Spanish train dataset. For Task1b we have used a Twitter-trained version of the XLM-RoBERTa multilingual model fine-tuned with a balanced version of the English and Spanish train datasets.
  
- **UC3M-LCPM-DIPROMATS2024-R3.json**. For Task1a predictions in English we have used a Bert-base-uncased model fine-tuned using an augmented version of the English train dataset. For predictions in Spanish for Task1a we have used the Roberta-Large model fine-tuned with an augmented version of the Spanish train dataset. For Task1b in English we have used a RoBERTa-large model fine-tuned with a balanced version of the English train dataset. For Task1b in Spanish we have used a Beto-base-uncased model fine-tuned with the original Spanish train dataset.
  
- **UC3M-LCPM-DIPROMATS2024-R4.json**. For Task1a predictions in English we have used a RoBERTa-large model fine-tuned using an augmented version of the English train dataset. For predictions in Spanish for Task1a we have used a sentiment analysis version of the Robertuito model fine-tuned with the original Spanish train dataset. For Task1b in English we have used a Bert-base-uncased model fine-tuned with a balanced version of the English train dataset. For Task1b in Spanish we have used a sentiment analysis version of the Robertuito model fine-tuned with a balanced version of the Spanish train dataset.
  
- **UC3M-LCPM-DIPROMATS2024-R5.json**. For Task1a we have used a Twitter-trained version of the XLM-RoBERTa multilingual model fine-tuned with the original English and Spanish train datasets. For Task1b we have used a Twitter-trained version of the XLM-RoBERTa multilingual model fine-tuned with a balanced version of the English and Spanish train datasets.
