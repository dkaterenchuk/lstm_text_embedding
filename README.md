# lstm_text_embedding
## Keras LSTM model for learning sentence representations.

This project implements LSTM autoencoder model architecture to train embedding of short texts. 
This can be used an embedding model or as a part of larger project that relies on text representations.

### Assumptions: 
* Word Embeddings: 
this code is based on pretrained word embedding model. The models can be download online or trained 
on a given corpus. 
* Data:
This project used Wiki dump data. A sample data file can be found in "data/wiki/" folder.

The data is freelly available here: https://www.wikidata.org/wiki/Wikidata:Database_download

NOTE: the Wiki dumps are in xml format and in order to extract text I recommend to use wikiextractor
(https://github.com/attardi/wikiextractor) This project adds "--json" param to have the data in json 
format.

* Local path definitions:
some project organization is defined in "definitions.py" file, such as word embeddings and the data.
It needs to be modified to fit the local system organizations. 

### Usage:
To train word embedding on your dataset run the following command:

```bash
python train_word_embedding.py <data_path> <output_model> [word_embedding_path]
```

Args:

    data_path - path to wiki data
    output_model - path to save trained model
    word_embedding_path - path to pretrained word embedding
    
    
### LSTM training:

```bash
python train_lstm_embedding.py <data_dir> <output_file>
```

Args:

    data_dir - is the wiki data processed with "wikiextractor"
    ouput_file - is trained model
    
    