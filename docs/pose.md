[Type of positional embeddings](https://github.com/JonasGeiping/cramming/blob/main/cramming/architectures/embeddings.py):
- Learnable
- Sinusoidal
- Rotary
- ALiBi

(( cramming page 6 ))

Embedding: We implement scaled sinusoidal positional embeddings as described in Hua et al. (2022), __finding incremental benefits over learned or unscaled sinusoidal embeddings__. We see no improvements from decoupling the input and output embeddings (Chung et al., 2020). The suggestion from Lan et al. (2019) to factorize the input embedding provides no gains in our setting. We include a layer normalization at the end of the embedding block.

# What Language Model to Train if You Have One Million GPU Hours?
https://openreview.net/pdf?id=rI7BL3fHIZq

- Finding 1. Diverse cross-domain pretraining data combining web crawls with curated highquality sources significantly improves zeroshot generalization over pretraining datasets constructed from Common Crawl only.

- Finding 2. ALiBi positionnal embeddings significantly outperforms other embeddings for zero-shot generalization.

- Finding 3. Adding layer normalization after the embedding layer incurs a significant penalty on zero-shot generalization.

- - -

One aspect of the Transformer architecture that has attracted recent significant interest is the way position information is captured within the model. Positional embeddings are important because without positional embeddings, Transformers cannot order tokens against one another.

__Background__ The Transformer paper Vaswani et al. (2017) proposed two options: static sinusoidal position embeddings and learned position embeddings (i.e., the position of each token is associated with a learned embedding vector). __Learned position embeddings are popular in large language models, and are used for GPT-3__. Su et al. (2021) later proposed the rotary position embedding method, where the query and key representations inside the self-attention mechanism is modified such that the attention computation captures relative distances between keys and queries. Recently, Press et al. (2022) proposed a position method which does not use embeddings, and instead directly attenuates the attention scores based on how far away the keys and queries are.

__Results__ We compare learned, rotary, and ALiBi position embeddings, and include a baseline without position embeddings. Our results are presented in Table 2. Although __learned positional embeddings outperforms rotary embeddings__, __ALiBi yield significant better results than all other alternatives__. We also confirm the discovery of Biderman (2021), that the __baseline without explicit position information shows competitive performance__. While bidirectional models require positional embeddings to determine the location of tokens, we find autoregressive models can simply leverage the causal attention masking.

![](files/pose-00.png)
https://paperswithcode.com/method/alibi

https://youtu.be/-Kgxv64aG3o explain Sinusoidal and compare it to ALiBi

Implementation https://nn.labml.ai/transformers/alibi/index.html

- - -

ALiBi author video https://www.youtube.com/watch?v=Pp61ShI9VGc

Trong b??i b??o g???c v??? tfm, h??? n??i r???ng tfm may exactly stick with longer than the ones encountered during training. The first thing that we didn't have by is we actually tested this thing so this game was made in 2017 but nobody actually looked into it in the paper. And the first thing we wonna do is we're going to figure out if this thing is correct. Can tfm actually do inference on sequences that are longers than what they saw during training? So here is how we are going to test it: we train a LM length of 1024, trained on WikiText103 a dataset of 100m tokens, 247M params.

![](files/pose-01.jpg)

Th???c nghi???m n??y tr??ng c?? v??? ngu ng???c v?? ch??ng ta train tr??n c??c chu???i c?? ????? d??i 1024, v?? sau ???? ki???m th??? tr??n c??c chu???i c?? ????? d??i l???n h??n 1024. B??i b??o g???c v??? tfm n??u 2 c??ch ????? encode positions, m???t l?? learnable c??ch n??y kh??ng ??p d???ng ???????c v?? 1 khi ???????c hu???n luy???n pos encode l?? c??? ?????nh kh??ng m??? r???ng ???????c. C??ch th??? 2 l?? d??ng sinusoidial encoding, c??ch n??y c?? th??? m??? r???ng v?? h??? hy v???ng r???ng n?? s??? ho???t ?????ng t???t khi m??? r???ng ????? d??i chu???i. Ch??a ai ki???m ch???ng gi??? thi???t n??y v?? ch??ng t??i ???? l??m ??i???u ????, th???c t??? ch??? ra r???ng n?? ho???t ?????ng kh??ng t???t, perplexity c???a ML t??ng l??n r???t nhanh (???????ng m??u v??ng).

=> K???t lu???n #1: original tfm LMs kh??ng th??? ngo???i suy!

V???Y T???I SAO NGO???I SUY L???I QUAN TR???NG? C?? r???t nhi???u l?? do nh??ng ?????u ti??n l?? cho d?? b???n hu???n luy???n tr??n chu???i ng???n nh??ng khi infer b???n ch???y ???????c tr??n chu???i d??i n?? s??? ti???t ki???m cho ch??ng ta r???t nhi???u t??nh to??n, con ng?????i kh??ng ???????c hu???n luy???n tr??n nh???ng chu???i d??i nh??ng v??n hi???u ???????c s??ch v?? nh???ng ??o???n v??n d??i, v?? th??? n?? s??? ti???t ki???m cho ch??ng ta r???t nhi???u t??nh to??n khi hu???n luy???n. Ch??ng ta ???? m???t kh??? n??ng ngo???i suy ???? khi r???i xa RNNs!

Ch??ng ta c?? kh??? n??ng ???? khi d??ng RNNs, ch??ng ta th?????ng hu???n luy???n tr??n chu???i 100-200 tokens nh??ng v??? l?? thuy???t RNNs c?? th??? x??? l?? chu???i c?? ????? d??i v?? t???n khi infer nh??ng tfm kh??ng l??m ???????c. ????ng l?? RNNs qu??n h???t m???i th??? nh??ng n?? kh??ng explode, v?? th??? RNN th???c s??? l??m t???t h??n. You'd probably get dimishing returns in terms of like longer lengths would probably stop and the accidentally ... ????ng r???i, ch??ng kh??ng hi???u g?? d??i h??n few hundred tokens away but they don't explode.

OK, vi???c ti???p theo ch??ng t??i th??? nghi???m l?? just change position of bank that we use, and so the first thing that we try is rotary position embeddings, which are like super complicated mapping but none of that mathematics, what they really do is they instead of adding position embeddings to where the representations they dot product position of embeddings and word representation. The second thing we try is T5, m???t m?? h??nh n???i ti???ng, v?? nhi???u ng?????i kh??ng bi???t r???ng khi gi???i thi???u T5 h??? ???? gi???i thi???u m???t ki???u pos embedding m???i, h??? n??i r???t ??t v??? n?? trong b??i b??o, nh??ng n?? l?? m???t c??ch m???i v?? duy nh???t ????? model the positionality and khi ch??ng t??i th??? c??ch ????, n?? th???c s??? c???i thi???n hi???u n??ng m???c d?? m?? h??nh c???a ch??ng t??i kh??ng li??n quan g?? t???i T5, ch??ng t??i ch??? s??? d???ng pos embedding gi???ng nh?? T5 m?? th??i. Ch??ng t??i hu???n luy???n v???i 1k tokens v???y m?? ch??ng t??i c?? th??? m??? r???ng hi???u qu??? t???i 2k tokens, nh??ng sau ???? hi???u n??ng b???t ?????u gi???m ??i.

=> K???t lu???n #2: ngo???i suy l?? c?? th??? ????n gi???n b???ng c??ch ?????i c??ch encode position!

????? c?? th??? ngo???i suy, ch??ng t??i t?????ng r???ng ph???i thay ?????i c?? ch??? attn, thay ?????i pos embedding method, v?? r???t nhi???u th??? n???a, nh??ng c?? v??? nh?? l?? ch??? c???n thay ?????i position embedding method has lots of power and can really impact performance.

![](files/pose-02.jpg)
And here is our boy, ALiBi.

=> K???t lu???n #3: ALiBi is awesome :D

ALiBi th???c s??? r???t d??? ????? th???c thi, ch??? m???t b?????c ????n gi???n:
- step 0: comment out the sinusoidal embeddings
- step 1: add bias to attention scores

![](files/pose-03.jpg)
V?? LM l?? m?? h??nh nh??n qu???n n??n v???i chu???i t??? ??? tr??n v?? t??? cu???i c??ng d??ng l??m query v?? nh???ng t??? tr?????c n?? l?? keys.x

![](files/pose-04.jpg)
Ch??ng t??i ????n gi???n tr??? ??i bias v??o attn scores, t??? c??ng ??? xa query c??ng b??? tr??? nhi???u h??n. N???u ch??? l??m nh?? v???y n?? s??? kh??ng ho???t ?????ng t???t v?? c??c con s??? n??y s??? t??ng qu?? nhanh v?? th??? ch??ng t??i normalize ch??ng b???ng c??ch nh??n v???i m???t s??? m n???m trong kho???ng t??? 0 t???i 1. V?? ??i???u th?? v??? ??? ????y l?? ????y l?? ph???n c???t l??i nh???t c???a ALiBi, ch??ng ta th???c hi???n multi-head attn v?? v?? th??? b???t c??? c??i g?? b???n attend b???n l??m n?? v???i a tension of 8 or 16, or 32 diff heads, b???n th???c s??? attend t???i chu???i ?????u v??o 32 l???n kh??c nhau, v?? b???i v?? ch??ng ta c?? h??? s??? chu???n h??a m, ch??ng ta c?? th??? do something really cool here. N???u ch??ng ta s??? d???ng 16 heads, ch??ng ta c?? th??? s??? d???ng 16 m kh??c nhau. And what that does is now we have this distance bias, which grow in different ways. _So if m is really close to 1 if it's really big that means that this function is going to increase really really quickly and so this spike is going to become big very very quickly and so those heads are going to look at very very very few tokens_. I've seen this empirically, I can look at what model is looking at and I can see the head that have high m value they only look at like 5, 6 ho???c 10 tokens. V?? ng?????c l???i v???i m g???n v???i 0 th?? attn s??? nh??n v??o context r???t d??i nh?? l?? 1000-2000 tokens away. V?? v?? th??? khi th???c thi Alibi t??i th???y s??? c???n thi???t ph???i c?? nhi???u gi?? tr??? m. And this inductive bias will forces heads to behave differently. ??i???u n??y r???t kh??c v???i attn truy???n th???ng, m???c d?? ???????c kh???i t???o v?? update kh??c nhau nh??ng v??? c?? b???n ????? d??i quan s??t c???a ch??ng l?? gi???ng nhau.

![](files/pose-05.jpg)
L??u ?? l?? m kh??ng ???????c hu???n luy???n, m?? l?? ???????c thi???t l???p b???ng tay. T???i sao l???i v???y? V?? khi thi???t l???p ????n gi???n nh?? tr??n ch??ng ho???t ?????ng r???t t???t v???i nhi???u t??c v??? NLP, th???m ch?? ???nh. C??n khi ???????c hu???n luy???n ch??ng l??m vi???c train model ch???m h??n v?? ????i khi g???p tr???c tr???c.

![](files/pose-06.jpg)

## Tri???n khai
![](files/pose-07.jpg)

ALiBi r???t hi???u qu???:
- Kh??ng c???n pos embd
- ALiBi ch???y nhanh nh?? sinusoidal
- Kh??ng c???n th??m parmas nh?? T5
- ????i khi ALiBi c???n th??m 100MB b??? nh??? 

## Interesting discussions https://youtu.be/Pp61ShI9VGc?t=1270

![](files/pose-02.jpg)
Quay tr??? l???i v?? d??? v???i Sinusoidal v?? xem t???i sao n?? l???i explode. T??i tin r???ng tfm lm hu???n luy???n pos embedding over fit to specific position. T??i gi??? thi???t r???ng, tfm ngh?? r???ng dog at five and dog at six are two different but similar concepts, which are similar but different to dog seven which is similar but different to dog at eight. Khi ch??ng ta ????a cho tfm position embedding, ch??ng ta ch??? mu???n n?? hi???u kho???ng c??ch t????ng ?????i gi???a c??c t???, nh??ng cu???i c??ng n?? l???i overfiting to specific positions and __the only reason that GPT-3 works is because their training datasets are so massive, they've just seen everywhere at basically every single position and that's why we kind of feel like they can kind of generalize the stuff but they don't__ If you give them position of writing I really believe that they over fit to specific position rings and I have bunch of experiments where that's really make me strongly believe this thing.

M???t trong nh???ng b???ng ch???ng l?? khi hu???n luy???n m?? h??nh v???i 250m params, n?? c?? th??? ngo???i suy th??m 50 tokens, ngh??a l?? hu???n luy???n v???i 1024 v?? n?? works ok v???i 1025, 1026, v?? ch??? explode t??? 1070. M???t kh??c khi ta hu???n luy???n 1.3B model, n?? exploded right away, ngh??a l?? ta hu???n luy???n n?? v???i 1024 v?? cho n?? 1025 tokens and right away the perplexity went through the roof. And we know that larger model have more capacity to overfit more and that's one of piece of evidence that makes you think that tfm overfit the specific position of things. ...

![](files/pose-08.jpg)
M???t b???ng ch???ng n???a l?? ch??ng t??i ch??? hu???n luy???n ALiBi v???i L = 512 r???i cho n?? ngo???i suy ra v???i chu???i c?? ????? d??i t??? 513 t???i 3072. Ch??ng t??i so s??nh hi???u su???t c???a c??ch ???? v???i vi???c hu???n luy???n Sinusoidal cho c??c ????? d??i kh??c nhau. V?? th???t ng???c nhi??u ALiBi kh??ng ???????c hu???n luy???n t???i L > 512 ho???t ?????ng t???t h??n Sinusoidal ???? ???????c hu???n luy???n t???i L.

![](files/pose-09.jpg)
V?? khi ALiBi ???????c hu???n luy???n v???i L l???n h??n, n?? c??ng ho???t ?????ng t???t h??n :D


![](files/pose-10.jpg)

H??y th??? nghi???m v???i model l???n h??n
![](files/pose-11.jpg)
K???t qu??? l?? ALiBi lu??n t???t h??n!

- - -

https://ofir.io/The-Use-Case-for-Relative-Position-Embeddings

We???re in 2022 but many of our most popular causal language models (LMs), including GPT-3, still use absolute positional embeddings. I believe we should stop using those and move to relative positional embeddings such as ALiBi. Deepmind???s Gopher and BigScience???s BLOOM already use relative positioning methods, and I???ve heard that multiple upcoming models also will, and so hopefully this post will help in encouraging the remanining holdouts to follow suit.

https://arxiv.org/abs/2210.12574
The Curious Case of Absolute Position Embeddings
