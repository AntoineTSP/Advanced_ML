# Advanced_ML
This will contain all the file for the ENSAE course : Advanced ML

# t-SNE custom implementation
We coded our own implementation of the t-SNE method, following the Scikit-Learn ways. 
To create an instance and fit it onto data: 
```python
from TSNE_code.TSNE_utils import TSNE

custom_tsne = TSNE(n_components=2, perplexity=15, 
                   adaptive_learning_rate=True, patience=50, 
                   n_iter=1000, early_exaggeration=4)

custom_embedding = custom_tsne.fit_transform(X, verbose=3)
```

# Documentation
The t-SNE functions are all explained through docstrings. We also leveraged theSphinx library to create a Numpy-like HTML documentation, making it more easily readable. <br>

To view this documentation, navigate to the `build/html/index.html` file or click [here](docs/build/html/index.html).