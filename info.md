preprocesamiento:
eliminamos las columnas innecesarias, con mucha falta de informacion o literalmente trampa(reviews de usuarios y metacritic)
```
'imdb_title_id', 'date_published', 'reviews_from_users', 'reviews_from_critics', 'original_title' 'worlwide_gross_income', 'budget', 'usa_gross_income', 'metascore'
```

luego se hizo un analisis de las columnas restantes para medir su "complejidad" para ver como manejarlas, se propone que las mas
simples se transformen en one hot encoding, como generos de la pelicula, y las mas complejas como desc y titulo se hagan x embedding
```
df = pd.read_csv('data/IMDb_movies_cleaned.csv')
Dataset shape: (82186, 18)
Total rows: 82,186

year
──────────────────────────────────────────
Unique values: 156
Uniqueness ratio: 0.19%
Data type: object
Top 10 most common values:
  - 2017: 3,015 (3.67%)
  - 2018: 2,990 (3.64%)
  - 2016: 2,786 (3.39%)
  - 2015: 2,472 (3.01%)
  - 2012: 2,376 (2.89%)
  - 2011: 2,290 (2.79%)
  - 2009: 2,182 (2.65%)
  - 2010: 2,169 (2.64%)
  - 2019: 2,058 (2.50%)
  - 2008: 2,041 (2.48%)

duration
──────────────────────────────────────────
Unique values: 262
Uniqueness ratio: 0.32%
Data type: int64
Top 10 most common values:
  - 90: 4,920 (5.99%)
  - 95: 3,071 (3.74%)
  - 100: 2,965 (3.61%)
  - 92: 2,333 (2.84%)
  - 93: 2,318 (2.82%)
  - 85: 2,216 (2.70%)
  - 88: 2,158 (2.63%)
  - 94: 2,110 (2.57%)
  - 96: 2,097 (2.55%)
  - 97: 2,056 (2.50%)
Range: [41, 808]
Mean: 100.18, Median: 96.00

votes
──────────────────────────────────────────
Unique values: 14,908
Uniqueness ratio: 18.14%
Data type: int64
Top 10 most common values:
  - 101: 285 (0.35%)
  - 105: 274 (0.33%)
  - 100: 274 (0.33%)
  - 102: 269 (0.33%)
  - 112: 264 (0.32%)
  - 111: 260 (0.32%)
  - 106: 260 (0.32%)
  - 109: 254 (0.31%)
  - 107: 252 (0.31%)
  - 117: 250 (0.30%)
Range: [99, 2278845]
Mean: 9882.35, Median: 516.00

country
──────────────────────────────────────────
Unique values: 4,837
Uniqueness ratio: 5.89%
Data type: object
Top 10 most common values:
  - USA: 28,223 (34.34%)
  - India: 5,371 (6.54%)
  - UK: 4,059 (4.94%)
  - Japan: 2,919 (3.55%)
  - France: 2,912 (3.54%)
  - Italy: 2,276 (2.77%)
  - Canada: 1,766 (2.15%)
  - Germany: 1,253 (1.52%)
  - Hong Kong: 1,166 (1.42%)
  - Spain: 1,138 (1.38%)

language
──────────────────────────────────────────
Unique values: 4,313
Uniqueness ratio: 5.25%
Data type: object
Top 10 most common values:
  - English: 35,541 (43.24%)
  - French: 3,721 (4.53%)
  - Japanese: 2,671 (3.25%)
  - Spanish: 2,663 (3.24%)
  - Italian: 2,555 (3.11%)
  - Hindi: 1,962 (2.39%)
  - German: 1,588 (1.93%)
  - Russian: 1,168 (1.42%)
  - English, Spanish: 1,103 (1.34%)
  - Turkish: 1,083 (1.32%)

production_company
──────────────────────────────────────────
Unique values: 31,104
Uniqueness ratio: 37.85%
Data type: object
Top 10 most common values:
  - Unknown: 3,802 (4.63%)
  - Metro-Goldwyn-Mayer (MGM): 1,281 (1.56%)
  - Warner Bros.: 1,148 (1.40%)
  - Columbia Pictures: 909 (1.11%)
  - Paramount Pictures: 901 (1.10%)
  - Twentieth Century Fox: 863 (1.05%)
  - Universal Pictures: 729 (0.89%)
  - RKO Radio Pictures: 534 (0.65%)
  - Universal International Pictures (UI): 272 (0.33%)
  - Mosfilm: 263 (0.32%)



director
──────────────────────────────────────────
Unique values: 33,472
Uniqueness ratio: 40.73%
Data type: object
Top 10 most common values:
  - Jesús Franco: 85 (0.10%)
  - Michael Curtiz: 85 (0.10%)
  - Lesley Selander: 78 (0.09%)
  - Lloyd Bacon: 73 (0.09%)
  - William Beaudine: 70 (0.09%)
  - Richard Thorpe: 68 (0.08%)
  - John Ford: 66 (0.08%)
  - Gordon Douglas: 64 (0.08%)
  - Raoul Walsh: 61 (0.07%)
  - Mervyn LeRoy: 59 (0.07%)
→ Total unique individual director: 33,443

writer
──────────────────────────────────────────
Unique values: 65,400
Uniqueness ratio: 79.58%
Data type: object
Top 10 most common values:
  - Jing Wong: 76 (0.09%)
  - Kuang Ni: 44 (0.05%)
  - Woody Allen: 40 (0.05%)
  - Leonardo Benvenuti, Piero De Bernardi: 34 (0.04%)
  - Cheh Chang, Kuang Ni: 31 (0.04%)
  - Giannis Dalianidis: 29 (0.04%)
  - Carlo Vanzina, Enrico Vanzina: 28 (0.03%)
  - Ingmar Bergman: 27 (0.03%)
  - William Shakespeare: 25 (0.03%)
  - Agenore Incrocci, Furio Scarpelli: 25 (0.03%)
→ Total unique individual writer: 60,507

actors
──────────────────────────────────────────
Unique values: 82,149
Uniqueness ratio: 99.95%
Data type: object
Top 10 most common values:
  - Nobuyo Ôyama, Noriko Ohara, Michiko Nomura, Kaneta...: 8 (0.01%)
  - Sergey A.: 6 (0.01%)
  - Richard Pryor: 3 (0.00%)
  - Ian McKellen, Martin Freeman, Richard Armitage, Ke...: 3 (0.00%)
  - H.B. Halicki, Marion Busia, Jerry Daugirda, James ...: 2 (0.00%)
  - Tomoki Hirose, Yûki Hiyori, Rin Ishikawa, Itsuki S...: 2 (0.00%)
  - Mai Fuchigami, Ai Kayano, Mami Ozaki, Ikumi Nakaga...: 2 (0.00%)
  - Thomas Freeley, Maria Petrano, Jacob Whiteshed: 2 (0.00%)
  - Hiroshi Kamiya, Takahiro Sakurai, Maaya Sakamoto, ...: 2 (0.00%)
  - Toni Servillo, Elena Sofia Ricci, Riccardo Scamarc...: 2 (0.00%)
→ Total unique individual actors: 406,289



genre
──────────────────────────────────────────
Unique values: 1,240
Uniqueness ratio: 1.51%
Data type: object
Top 10 most common values:
  - Drama: 11,849 (14.42%)
  - Comedy: 6,982 (8.50%)
  - Comedy, Drama: 3,826 (4.66%)
  - Drama, Romance: 3,316 (4.03%)
  - Comedy, Romance: 2,355 (2.87%)
  - Comedy, Drama, Romance: 2,230 (2.71%)
  - Horror: 2,192 (2.67%)
  - Drama, Thriller: 1,323 (1.61%)
  - Crime, Drama: 1,313 (1.60%)
  - Action, Crime, Drama: 1,278 (1.56%)
→ Total unique individual genres: 25
→ Individual genres: Action, Adult, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, Family, Fantasy, Film-Noir, History, Horror, Music, Musical, Mystery, News, Reality-TV, Romance
  ... and 5 more


title
──────────────────────────────────────────
Unique values: 78,595
Uniqueness ratio: 95.63%
Data type: object
Top 10 most common values:
  - Anna: 9 (0.01%)
  - Darling: 8 (0.01%)
  - Wanted: 7 (0.01%)
  - Maya: 7 (0.01%)
  - Vendetta: 7 (0.01%)
  - Hero: 7 (0.01%)
  - I miserabili: 7 (0.01%)
  - Solo: 7 (0.01%)
  - Aurora: 7 (0.01%)
  - Lucky: 6 (0.01%)
Average text length: 17.0 characters

description
──────────────────────────────────────────
Unique values: 82,061
Uniqueness ratio: 99.85%
Data type: object
Top 10 most common values:
  - The story of: 15 (0.02%)
  - Mail: 6 (0.01%)
  - In this sequel to: 5 (0.01%)
  - The true story of: 5 (0.01%)
  - Based on: 5 (0.01%)
  - During World War II, a teenage Jewish girl named A...: 4 (0.00%)
  - Tom Sawyer and his pal Huckleberry Finn have great...: 4 (0.00%)
  - Emil goes to Berlin to see his grandmother with a ...: 4 (0.00%)
  - Desperate measures are taken by a man who tries to...: 4 (0.00%)
  - Based on the true story of: 3 (0.00%)
Average text length: 160.0 characters

avg_vote
──────────────────────────────────────────
Unique values: 88
Uniqueness ratio: 0.11%
Data type: float64
Top 10 most common values:
  - 6.4: 3,274 (3.98%)
  - 6.2: 3,221 (3.92%)
  - 6.5: 3,216 (3.91%)
  - 6.3: 3,176 (3.86%)
  - 6.6: 3,087 (3.76%)
  - 6.1: 3,017 (3.67%)
  - 6.7: 2,971 (3.61%)
  - 6.8: 2,952 (3.59%)
  - 6.0: 2,723 (3.31%)
  - 7.0: 2,654 (3.23%)
Range: [1.0, 9.8]
Mean: 5.91, Median: 6.10


Complexity Classification:
LOW complexity (< 100 unique):    1 columns
MEDIUM complexity (100-1000):     3 columns
HIGH complexity (1000-10000):     4 columns
VERY HIGH complexity (>10000):    9 columns

```

# transforms
finalmente, se transformaron las columnas:
una propuesta por Emilio, fue transformar a los actores, directores y escritores con su avg rating, lo cual me parece una buena sugerencia, por lo que para lograrlo se partio el dataset en train y test, y usando exclusivamente informacion de train, se transformaron esas columnas para train y test.

# votes
para la columna de 'votes', que me parece importante, le aplique una transformada log(1+x) para evitar log0 porque se cargaba mucho hacia la derecha la distribucion:
```
- Median: 516 votes
- Mean: 9,882 votes
- Max: 2,278,845 votes
```

# genre
para generos Multi-Hot Encoding:

"The Dark Knight" → "Action, Crime, Drama"

binary columns:
```
Action | Comedy | Crime | Drama | Horror | Romance | ...
  1    |   0    |   1   |   1   |   0    |    0    | ...
```
"Superbad" → "Comedy, Romance"
```
Action | Comedy | Crime | Drama | Horror | Romance | ...
  0    |   1    |   0   |   0   |   0    |    1    | ...
```

# country / language / production_company
el problema tambien fue para pais, ya que existen 4,837 valores unicos, y un one hot encoding generaria la misma cantidad de columnas, lo cual no es muy ideal, ya que incluso algunos paises solo aparecen unas cuantas veces
se tomo un enfoque similar al de los actores, directores y escritores:
se remplaza el valor de la categoria con su promedio de target, por ejemplo: \
```
pelicula 1: USA → avg_vote = 7.2
pelicula 2: USA → avg_vote = 6.8
pelicula 3: USA → avg_vote = 7.0
pelicula 4: Mexico → avg_vote = 6.5
pelicula 5: Mexico → avg_vote = 6.7

Target encoding:
USA → (7.2 + 6.8 + 7.0) / 3 = 7.0
Mexico → (6.5 + 6.7) / 2 = 6.6

Now in features:
pelicula 6 from USA → country_encoded = 7.0
pelicula 7 from Mexico → country_encoded = 6.6
```
pero esto podria ser problematico, si por ejemplo tomamos Luxemburgo, que aparece una sola vez y su rating fue 8.5, ahora todas las peliculas que aparezcan de Luxemburgo tendran 8.5, lo cual no seria congruente con la media global.

se aplico entonces una tecnica llamada suavizado bayesiano:
```python
smoothing = 1.0 / (1.0 + np.exp(-(count - min_samples) / min_samples))
smoothed = global_mean * (1 - smoothing) + category_mean * smoothing
```
la cual asigna un valor de "confianza" en el valor calculado con la media de ese pais, y otra "confianza" a la media global, y las mezcla, por lo que resulta en un encoding mas natural. reduciendo el overfitting a paises raros, y mantiene informacion en los paises mas representados
```
Global mean: 5.91
USA (28,223 movies, mean 6.2): smoothed ≈ 6.2 (trust USA mean)
Luxembourg (1 movie, mean 8.5): smoothed ≈ 6.1 (mostly use global mean)
```

# actor/director/writer
la idea es calcular una metrica de calidad basada en trabajos previos, pero el problema es que estas columna incluye a los involucrados en una sola linea, asi que primero lo separamos y calculamos individualmente su desempe;o, unicamente usando los datos de train:

```python
for idx, row in train_df.iterrows():
    people = [p.strip() for p in str(row[column]).split(',')]
    rating = row[target]
    for person in people:
        person_ratings[person].append(rating)

def encode_row(row):
    people = [p.strip() for p in str(row[column]).split(',')]
    ratings = [person_mean.get(p, global_mean) for p in people]
    return np.mean(ratings)
```


**Ejemplo:**
```
Christopher Nolan's movies in TRAINING set:
- Inception: 8.8
- Interstellar: 8.6
- The Dark Knight: 9.0
- Memento: 8.4

Christopher Nolan's rating = (8.8 + 8.6 + 9.0 + 8.4) / 4 = 8.7
```

**Example movie in test set:**
```
Movie: "Tenet"
Director: Christopher Nolan
→ director_rating = 8.7

Actors: "John David Washington, Robert Pattinson, Elizabeth Debicki"
- John David Washington: 6.2 (from training movies)
- Robert Pattinson: 6.8 (from training movies)
- Elizabeth Debicki: 6.5 (from training movies)
→ actor_rating = (6.2 + 6.8 + 6.5) / 3 = 6.5
```

adicionalmente, se genero una columna con la cantidad de apariciones de una figura:
```
person_counts = {person: len(ratings) for person, ratings in person_ratings.items()}
```

# desc, title
se uso magia negra de embeddings:
```python
model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
train_embeddings = model.encode(train_descriptions, batch_size=64)
```

**Pre-trained model**: `all-mpnet-base-v2`
  - Trained on billions of sentences
  - Learned what words/phrases mean
  - Can encode ANY text into a 768-dimensional vector


al final, nos quedaron las siguientes categorias:
```
feature_cols = ['year', 'duration', 'votes_log',
                'country_encoded', 'language_encoded', 'production_company_encoded',
                'director_rating', 'director_count',
                'writer_rating', 'writer_count',
                'actor_rating', 'actor_count']

train_features = pd.concat([
    train_df[feature_cols].reset_index(drop=True),
    train_genre_encoded.reset_index(drop=True),
    train_embed_df.reset_index(drop=True)
], axis=1)

test_features = pd.concat([
    test_df[feature_cols].reset_index(drop=True),
    test_genre_encoded.reset_index(drop=True),
    test_embed_df.reset_index(drop=True)
], axis=1)```
