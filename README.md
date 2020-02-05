# personalization-and-rl-literature-review
This is a series literature review of personalization and recommendation systems. Though there is no clear boundary between personalization and recommendation, it is commonly to recognize the search with query involved as personalization like search engine, e-commerce search, while to treat those without query as recommendation like news and videos recommendation.

# Personalization

## Embedding
- **Search Personalization with Embeddings**, 2016 (https://arxiv.org/pdf/1612.03597.pdf)
   - Previous search personalization highly depends on user profile (like human generated ontology), and this paper proposes a novel embedding method to track the user's topical interest.
   - Each user is represented by two matrices **W<sub>u,1</sub>** and **W<sub>u,2</sub>** to represent the relationship between user and query/document, and an additional vector *v<sub>u</sub>* to represent the user topical interests. On the other side, each query/document is represented by a vector *v<sub>q</sub>* and *v<sub>d</sub>* respectively which is pre-determined using the LDA topic model (http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf).
   - The goal is selecting a *scoring function* like 
   <p align="center">
   <img src="https://render.githubusercontent.com/render/math?math=f(q, u, d) = ||W_{u,1}v_{q} %2B v_{u}  %2B W_{u,2}v_{d}||_{l_2}">
   </p>
   such that the value *f(q, u, d)* of relevant triple *(q, u, d)* is smaller than that of irrelevant triple *f(q', u, d')*.
   - Next we train the model by minimize the total marginal value for the same user <img src="https://render.githubusercontent.com/render/math?math=\sum{max(0, \gamma %2B f(q, u, d) - f(q', u, d'))}"> where sum is taken for all relevant triples *(q, u, d)* and all irrelevant triples *(q', u, d')*.

# Recommendation

