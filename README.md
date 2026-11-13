# SMS Spam Detection

Machine-learning classifier that labels SMS messages as **spam** or **ham**, tuned to catch spam while rarely blocking real messages.

**Data:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (downloaded automatically on first run).

## Approach
1. EDA: class imbalance, message-length patterns, top words
2. Cleaning: duplicates removed before splitting; URLs, currency and numbers kept as placeholder tokens; stop words removed (negations kept)
3. TF-IDF (unigrams + bigrams) inside a scikit-learn `Pipeline` to avoid leakage
4. Naive Bayes vs Logistic Regression vs Decision Tree, 5-fold stratified CV on spam-class F1
5. Decision threshold tuned on a validation split with a precision floor (false positives are costly)
6. One final evaluation on the untouched test set, error analysis, saved model

## Results
Run the notebook and paste your final test-set numbers here (accuracy, spam precision, recall, F1).

## Run it
```bash
pip install pandas numpy matplotlib seaborn scikit-learn requests joblib nltk wordcloud
jupyter notebook sms_spam_detection.ipynb   # Kernel > Restart & Run All
```

Outputs: `spam_dataset.csv`, `spam_model.joblib`, and charts in `figures/`.

## Limitations
Small, dated, English-only dataset; bag-of-words ignores context. Modern spam and other languages need new data.
