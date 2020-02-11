# personalization-and-rl-literature-review
This is a series literature review of personalization and recommendation systems. Though there is no clear boundary between personalization and recommendation, it is commonly to recognize the search with query involved as personalization like search engine, e-commerce search, while to treat those without query as recommendation like news and videos recommendation.

# Personalization

## Embedding
- **Search Personalization with Embeddings**, 2016 (https://arxiv.org/pdf/1612.03597.pdf)
   - Previous search personalization highly depends on user profile (like human generated ontology), and this paper proposes a novel embedding method to track the user's topical interest.
   - Each user is represented by two matrices **W<sub>u,1</sub>** and **W<sub>u,2</sub>** to represent the relationship between user and query/document, and an additional vector *v<sub>u</sub>* to represent the user topical interests. On the other side, each query/document is represented by a vector *v<sub>q</sub>* and *v<sub>d</sub>* respectively which is pre-determined using the LDA topic model (http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf).
   - The goal is selecting a *scoring function* like <p align="center"><img src="https://render.githubusercontent.com/render/math?math=f(q, u, d) = ||W_{u,1}v_{q} %2B v_{u}  %2B W_{u,2}v_{d}||_{l_2}"></p> such that the value *f(q, u, d)* of relevant triple *(q, u, d)* is smaller than that of irrelevant triple *f(q', u, d')*. 
   - Next we train the model by minimize the total marginal value for the same user <p align="center"><img src="https://render.githubusercontent.com/render/math?math=\sum{max(0, \gamma %2B f(q, u, d) - f(q', u, d'))}"></p> where sum is taken for all relevant triples *(q, u, d)* and all irrelevant triples *(q', u, d')*.

- **Personalized Ranking in eCommerce Search**, 2019 (https://arxiv.org/pdf/1905.00052.pdf)
   - This paper uses a combination of latent features learned from item co-clicks in historic sessions and content-based features that use **item title** and **price**.
   - The first step is to learn item embeddings using in-session clicks history. Each phase is represented by "id1 id2 id3" which is clicked items in the same session. Phases are filtered for at least two items and each item appears in at least 16 phases. Then item embedding is learned using skip-gram model similar as *word2vec* (with window size 5 and dimension 32).
   - Four main features will be added to the rerank step(LambdaMART) for the in-session personalization:
      - **cos_distance_ave**: Average cosine distance of the item to be ranked from the previously clicked items in the embedding space
      - **cos_distance_last**: cosine distance to the last clicked item
      - **price_ratio_mean**: ratio of the price of the current item to the average price of previously clicked items
      - **title_jaccard_sim**: Jaccard similarity of the title of the current item to the last clicked item
      
- **Real-time Personalization using Embeddings for Search Ranking at Airbnb**, 2018 (https://astro.temple.edu/~tua95067/kdd2018.pdf)
   - Two types of embeddings will be learned and served as rerank features: short-term (in-session) and long-term
   - In-session listing embeddings: In each session, user clicked listing *s = (l1, l2, ..., lM)* will be treated as a context and listing embedding will be learned using the skip-gram model as in *word2vec*. However, the following parts are different:
      - Each context can be break down into two categories based on whether the last clicks is converted to a booking: If yes, then the booked listing will also be treated as one neighbor of the central listing
      - ![figure 1]()

# Recommendation

