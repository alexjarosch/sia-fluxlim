# sia-fluxlim

A python implementation of a MUSCL flux limiter to a shallow ice approximation flow code based on finite differences.

For an overview of possible flux limiter functions (phi) to the MUSCL scheme, see [Wikipedia - flux limiter](https://en.wikipedia.org/wiki/Flux_limiter).
I have implemented now the Sweby (1984) limiter, which uses a parameter called 'beta'. For 1 <= beta <= 2 the limiter is second order TVD and its end members are equal to the limiter discussed in the paper below. If beta = 1, the Sweby limiter is equal to Roe's minmod limiter and if beta = 2 one gets Roe's superbee limiter. 

For more details such as the scientific argument as to why this is a good idea, please consult the accompanying publication by **A. H. Jarosch**, C. G. Schoof, and F. S. Anslow, "[Restoring mass conservation to shallow ice flow models over complex terrain](http://www.the-cryosphere.net/7/229/2013/tc-7-229-2013.html),” *The Cryosphere*, Vol. 7, Iss. 1, pp. 229-240, 2013.
