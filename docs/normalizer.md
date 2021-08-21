## normalizer
```python
_normalizer_ = {'minmax','normalize','scale',None}
```
- all Sklearn models (and those that use an Sklearn API) have a normalizer argument that can be optimized in `tune()`
- `'minmax'`: MinMaxScaler from sklearn.preprocessing
- `'normalizer'`: Normalizer from sklearn.preprocessing
- `'scale'`: StandardScaler from sklearn.preprocessing