# WIP

Packages required (you can install them via 'pip install'):

'
ntlk
transformers
sentencepiece
numpy
torch
requests
'

Install them with:
```
pip install -r requirements.txt
```

Now simply run the code:
```
python3 main.py
```

The code is far from complete, but, by far it analyzes:
1. Post sentiment, number of posts
2. Average comments sentiment for a particular post, number of comments for this post
3. Average reactions sentiment per post (which can be correlated to comments sentiment)
4. Aggregated channel sentiment

Important: all the sentiments can be represented as diagrams in front-end

The main issue encountered now is huge overhead -- thanks to slow sentiment analysis, but this model is by far the best for slavic languages like Ukrainian and russian. Also the code will be further optimized with parallelization, batching etc. 