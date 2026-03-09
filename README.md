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


### Diagram

The above diagram has all things needed to sample and train our bigram.


### Torch utils

We need to get probablity of all frequencies here, to do that we use torch.

- [[Torch generator]]
- [[Torch multinomial]]



So suppose you run multinomial over a probablity distribution, and each index has a probablity of let's say 30%, 20%, 50%

so when you run the multinomial it will create a sample dataset of size given (num_samples) with indexes occuring approximately close to the probablities

Example: $[.3, 0.2, 0.5]$ distribution of size 6 can result in $[2,0,2,2,1,0]$  where 2 approximately results in 50% of the time, 0 30% of the time, and 1 20% of the time.
You can keep it consistently returning same results by using a generator (as explained previously). This doesn't mean that if you call multinomial again it will result in same output, it means if you call it the outputs will change as expected (1st call on multinomial will always be the same when the generator is used for 1st time, 2nd result on the same generator will also be the same). It means that the first time you pick it with a fresh generator of same seed, it will return the same output,

g init to seed 4, $[xi, xj, xk]$ are the outputs when multinomial is called i,j,k times. If you reinitialize g with seed 4, you will again get $[xi..]$ for the first time, then $[xj]$ where $x$ is a sample list.


## Bigram 

You start from ix 0 which is the "." character, the starting point. You sample from row 0 in N using multinomial so it's weighted by probability but not always picking the most likely one, gives you variety.

Say you get 'd' as your first sample. Now ix becomes the index of d. You go to row d, which makes sense because you want to know what character comes next after d. Row d has all the pair frequencies of d followed by every other letter (da, de, di...).

Say 'e' has high probability so you sample 'e'. Now ix becomes the index of e. You go to row e, which again makes sense because now you need to know what comes after e. Row e has all pair frequencies of e followed by every letter.

You keep doing this, each time you sample a character you move to its row because that row tells you what's likely to come next after that character.

You stop when ix == 0, meaning the model sampled "." which is the end token, so the name is done.

It produces name-like output because each row in N has real pair frequencies from the dataset. It's not always the same output because multinomial samples with randomness, high probability pairs get picked more often but not always.


```python
g = torch.Generator().manual_seed(2147483647)
for i in range(30):
    out = []
    ix = 0
    while True:
        p = N[ix].float()
        p = p / p.sum()
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
```


### Optimization

Sum and float again and again is not needed, just prep before hand the entire matrix and do that in parallel.

Following is a float representation of the 27x27 2dimension matrix
`` P = N.float() ``

Now to convert each row to it's probabilistic distribution you need to dive by it's rows sum, now you want to that for each row. This can be done parallely using [[Torch sum]] 

You want to add each row and present it as an array using
`P.sum(1, keepdim=True)` where 1 means adding over a single row  (col by col) and keep dim returns the dimension as `[27,1]` as it keeps the dimension of the row (and reduces only the columns). If you choose to make keepdim as False then you get in this case `[27]` array  and reducing the 1 col to a single array of 1 dimension. 

Dimensions 
`P = [27,27] sum of P's rows = [27,1]`

Now can you use this array to make P divide each row's element by it's sum of the row? You would need to understand [[Torch broadcasting rules]]. Which agrees to our situation where one dimension is equal (0th dimension 27 for both) and the second dimension is 1 which also works. 

