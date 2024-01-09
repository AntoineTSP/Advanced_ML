# Advanced ML : Course Project
_Observing the semantics of textual data through embedding coupled with t-SNE_

## t-SNE custom implementation
We implemented our own t-SNE method, following the _Scikit-Learn_ ways. 
To create an instance and fit it onto data: 
```python
from TSNE_code.TSNE_utils import TSNE

custom_tsne = TSNE(n_components=2, perplexity=15, 
                   adaptive_learning_rate=True, patience=50, 
                   n_iter=1000, early_exaggeration=4)

custom_embedding = custom_tsne.fit_transform(X, verbose=3)
```

## Documentation
The t-SNE functions are all explained through docstrings. We also leveraged the _Sphinx_ library to create a _numpy_-like HTML documentation, making it more easily readable. <br>

To view this documentation, navigate to the `build/html/index.html` file or click [here](build/html/index.html).

## Reference paper

 [Visualizing Data using t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html),
Laurens van der Maaten, Geoffrey Hinton; 9(86):2579−2605, 2008.
