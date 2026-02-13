# Program 1: Exploring Pre-Trained Word Vectors Using GloVe

## 1. Aim

To explore pre-trained word embeddings using the GloVe model, analyze semantic similarity between words, and perform vector arithmetic to understand word relationships.

---

## 2. Libraries Used

- **gensim** – to load pre-trained word embedding models  
- **numpy** – for numerical computations  
- **numpy.linalg.norm** – for calculating vector magnitude  

---

## 3. Model Description

The program uses the pre-trained model:

- **Model Name:** `glove-wiki-gigaword-100`
- **Vector Size:** 100 dimensions
- **Training Data:** Wikipedia + Gigaword corpus

In this model, each word is represented as a 100-dimensional dense vector.

Example representation:

"king" → [v1, v2, v3, ..., v100]
"queen" → [v1, v2, v3, ..., v100]

These vectors encode semantic relationships based on word co-occurrence statistics.

---

## 4. Methodology

### Step 1: Load the Pre-Trained Model

```python
model = api.load("glove-wiki-gigaword-100")
This loads a vocabulary of approximately 400,000 words, each mapped to a 100-dimensional vector.

Step 2: Finding Similar Words
model.most_similar("king", topn=5)

Similarity between words is computed using cosine similarity:

cosine similarity=A⋅B∣∣A∣∣ ∣∣B∣∣
cosine similarity=
∣∣A∣∣∣∣B∣∣
A⋅B

Value close to 1 → highly similar

Value close to 0 → unrelated

Value close to -1 → opposite direction

Example output:
queen
prince
monarch


This indicates that semantically related words are located near each other in vector space.
Step 3: Vector Arithmetic

The following operation is performed:

model.most_similar(
    positive=["king", "woman"],
    negative=["man"]
)

This computes:
v=vking−vman+vwoman
v=vking​−vman​+vwoman​

Interpretation:

    "king" represents male royalty

    Subtracting "man" removes the male component

    Adding "woman" introduces the female component

Result:

queen

This demonstrates that semantic relationships are encoded as linear transformations in vector space.
Step 4: Manual Vector Calculation

Vectors are extracted individually:

king_vec = model["king"]
man_vec = model["man"]
woman_vec = model["woman"]

New vector is computed: