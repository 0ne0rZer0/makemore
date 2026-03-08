makemore creates more of things that you give it. Hence makemore.

Example, it can create names that sound like a given list of names as input.

Character level language model would basically predict next character. 

### Current Language NNs:
- Bigram
- Bag of words
- MLP (Bengio 2003)
- RNN (Sutskever 2011)
- GRU (Kyunghyun 2014)
- Transformer (Vaswani 2017)

---
### What does a name tell us?

isabella,

i comes first, s comes second, then so on, and at the end a comes at last. 
This is just an example, but each name reinforces an order of the character

## Bigram

What character follows from the current character is the strategy, it is very weak language model but it's a good starting point.

Bigram is a collection of two characters (in order), and a frequency associated with it.
You add a token for start and end to also recognize starting and ending letters

```
b = {}

for w in words[:10]:

    # Add a special token start and end tokens

    chs = ['<S>'] + list(w) + ['<E>']

    for ch1, ch2 in zip(chs, chs[1:]):

        bigram = (ch1,ch2)

        b[bigram] = b.get(bigram, 0) + 1

        print(bigram)
        
```


![[Pasted image 20260309010133.png]]![[Pasted image 20260309011100.png]]