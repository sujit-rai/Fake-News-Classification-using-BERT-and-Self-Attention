# Fake-News-Classification-using-BERT-and-Self-Attention
Fake news classification (Binary and Six-way) on LIAR-PLUS as the benchmark dataset

"preprocess.py" produces a dataset consisting only sentences and justification

"preprocess_m.py" produces a dataset consisting of sentences + justification + emolex + metadata

"LSTM.ipynb" contains code for LSTM classifier

"BERT.ipynb" contains code for Bert classifier

"self_attention.ipynb" contains code for classifier with self attention

The dataset have been split into 2 files, "train.tsv" contains labels, sentences and justification. "train2.tsv" contains metadata. similarly for "val" and "test"

# Recommended mode of execution
Please use [this](https://drive.google.com/drive/folders/1-Oo5oYOhwctwi0KzhsN7BrXo_COhojTZ?usp=sharing) link to access Google Drive folder containing the Jupyter notebooks for Google Colab and all the relevant dataset files.
