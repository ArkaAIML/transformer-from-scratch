# Positional Encoding

Self-attention sees tokens as a set of vectors. By itself, it does not know whether a token came first, second, or tenth.

Sinusoidal positional encoding solves this by adding a deterministic position-dependent vector to each token embedding.

## Why add instead of concatenate?

## Why sine and cosine?

## Why multiple frequencies?

## Why the base 10000?

## Shape Walkthrough

Input token embeddings:

```txt
(batch_size, seq_len, d_model)