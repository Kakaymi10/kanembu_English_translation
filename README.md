# Project Name: Translation Model with Limited Dataset: English-Kanembu

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Creation and Preprocessing](#dataset-creation-and-preprocessing)
3. [Model Architecture and Design Choices](#model-architecture-and-design-choices)
4. [Training Process and Hyperparameters](#training-process-and-hyperparameters)
5. [Evaluation Metrics and Results](#evaluation-metrics-and-results)
6. [Insights and Potential Improvements](#insights-and-potential-improvements)

---

## Introduction
This project focuses on building a simple translation model using a very small dataset of around 10 sentences. The model is tasked with translating short sentences from English to another language. Given the limited amount of data, the focus is on understanding the performance of the model, exploring its limitations, and identifying areas for potential improvements.

---

## Dataset Creation and Preprocessing
The dataset is small, consisting of just 10 sentences. These sentences were manually gathered to serve as input-output pairs for the translation task. The dataset contains simple sentences such as:
- **English:** "Abakar is looking for banana"
- **Translation:** "Abakar banana mâi"

### Preprocessing Steps:
1. **Tokenization:** Each sentence was tokenized into words.
2. **Padding:** Since sentence lengths vary, padding was applied to ensure that all inputs are of equal length.
3. **Vocabulary:** A limited vocabulary was created from the dataset, including both the source and target languages.
4. **Encoding:** Each word in the sentences was mapped to an integer for feeding into the neural network.

---

## Model Architecture and Design Choices
Given the simplicity of the task and the small dataset, the model architecture chosen was a basic sequence-to-sequence (Seq2Seq) model. 

- **Encoder:** The encoder takes the input English sentence and converts it into a hidden state representation using a series of LSTM layers.
- **Decoder:** The decoder uses the hidden state from the encoder to generate the translated sentence, also using LSTM layers.
- **Embedding Layer:** Both the encoder and decoder have embedding layers to represent words as dense vectors.
- **Attention Mechanism (Optional):** Although not implemented in this iteration, an attention mechanism could be useful in improving translation accuracy by allowing the model to focus on different parts of the input sentence.
  
---

## Training Process and Hyperparameters
The training process was relatively straightforward due to the small dataset. The model was trained for a few epochs, with a small learning rate to prevent overfitting.

### Hyperparameters:
- **Optimizer:** Adam optimizer with a learning rate of 0.001.
- **Batch Size:** 1 (due to the small dataset).
- **Epochs:** 100.
- **Loss Function:** Categorical cross-entropy was used for the translation task.
- **Dropout:** To prevent overfitting, dropout layers were used in both the encoder and decoder.

---

## Evaluation Metrics and Results
Given the limited size of the dataset, the model's performance was assessed using the following metrics:

1. **Accuracy:**
   ```
   Accuracy: 91.96%
   Loss: 0.6629
   ```
   Despite the small dataset, the model achieved a decent accuracy on the training set, but this is likely due to overfitting, as the model learned to memorize the few available sentences.

2. **BLEU Score:** 
   The BLEU (Bilingual Evaluation Understudy) score is a standard metric for evaluating translation quality by comparing the predicted translation with the reference translation.
   - **BLEU Score:** The model performed very poorly with BLEU scores close to zero, indicating that the translations were not accurate.

   Sample outputs:
   ```
   English Sentence: Abakar is looking for banana
   Predicted Translation: token abakar banana mâi
   Reference Translation: Abakar banana mâi
   BLEU Score: 9.53e-155

   English Sentence: Did Fatime go to the market?
   Predicted Translation: token fatime kasuw ro tîdi a
   Reference Translation: Fatime kasuw-ro tîdi a?
   BLEU Score: 1.16e-231
   ```

   The model struggles to generate valid translations, likely due to the extremely small training dataset.

---

## Insights and Potential Improvements
### Key Insights:
- **Limited Data Impact:** The major challenge in this project is the small dataset size. With only 10 sentences, the model is unable to generalize well and instead memorizes the training data.
- **BLEU Score Failures:** The near-zero BLEU scores highlight the inadequacy of the model's translations. This indicates the need for a much larger and more diverse dataset to improve translation accuracy.
- **Overfitting:** The model achieves high accuracy during training but fails to perform well on new inputs, suggesting that it has overfit the limited data.

### Potential Improvements:
1. **Data Augmentation:** Increasing the dataset size by gathering more sentences or using data augmentation techniques (e.g., paraphrasing) could help the model generalize better.
2. **Pre-trained Embeddings:** Incorporating pre-trained word embeddings (e.g., Word2Vec or GloVe) could improve the model's ability to handle unseen words and sentences.
3. **Attention Mechanism:** Adding an attention mechanism could improve translation accuracy by allowing the model to focus on relevant parts of the input sentence.
4. **Hyperparameter Tuning:** Experimenting with different hyperparameters, such as batch size and learning rate, could further refine the training process.
5. **Use Smoothing in BLEU Score:** The warnings about the BLEU score suggest using a smoothing function to handle cases where n-gram overlaps are scarce.

---

By addressing these limitations and implementing improvements, the model could potentially deliver better translation results.
