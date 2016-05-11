---
layout: post
title:  "Derivative of softmax under Cross-Entropy Error function"
date:   2016-05-11 20:29:58 +0800
categories: machine-learning
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

>This derivation process used to stuck me for a while, so I plan to share this to others who may also encounter this problem.

First, a quick overview of how *softmax* work and what is *cross-entropy* error function:

## Softmax

Here is the equation how we compute \\(y_i\\), which is the probability class \\(i\\) occured, given its penalty \\(z_i\\):

$$y_i(z_i) = \frac{e^{z_i}}{\sum_{k=1}^K e^{z_k}}$$

where \\(K\\) is the number of all possible classes, \\(i\\) is the class we want.

## Cross Entorpy

The equation below compute the cross entropy \\(C\\) over *softmax* function:

$$C = -\sum_{k=1}^K t_k \log y_k$$

where \\(K\\) is the number of all possible classes, \\(t_k\\) and \\(y_k\\) are the target and the *softmax* output of class \\(k\\) respectively.

## Derivation

Now we want to compute the derivative of \\(C\\) with respect to \\(z_i\\), where \\(z_i\\) is the penalty of a particular class \\(i\\). We can expand the derivative like this(simply because \\(t_k\\) does not depended on \\(z_i\\), and the summation notation \\(\sum\\) means independent relation among its terms):

$$\frac{\partial C}{\partial z_i} = -\sum_{k=1}^K t_k \frac{\partial \log y_k}{\partial z_i}$$

By applying **Chain Rule**, we can simplify the above equation:

$$\frac{\partial C}{\partial z_i} = -\sum_{k=1}^K t_k \frac{\partial \log y_k}{\partial y_k} \frac{\partial y_k}{\partial z_i} = -\sum_{k=1}^K \frac{t_k}{y_k} \frac{\partial y_k}{\partial z_i}$$

Assume that we already knew \\(\frac{\partial y_k}{\partial z_i}\\)(more details will be presented later) is following:

$$\frac{\partial y_k}{\partial z_i} = y_k (1\{k = i\} - y_i)$$

where \\(1\\{k = i\\}\\) equals to \\(1\\) if \\(k = i\\) and \\(0\\) otherwise.

Plug that into the previous equation, we get:

$$\frac{\partial C}{\partial z_i} = -\sum_{k=1}^K \frac{t_k}{y_k} y_k (1\{k = i\} - y_i) = -\sum_{k=1}^K t_k (1\{k = i\} - y_i)$$

Note that the subscript \\(i\\) of \\(y_i\\) is constant and \\(1\\{k = i\\}\\) only have non-zero value if \\(k = i\\), finally we can get:

$$\frac{\partial C}{\partial z_i} = -\sum_{k=1}^K t_k (1\{k = i\} - y_i) = -\sum_{k=1}^K t_k 1\{k = i\} + \sum_{k=1}^K y_i = -t_i + y_i = y_i - t_i$$

It is almost done. If you are not interested in how to get \\(\frac{\partial y_k}{\partial z_i}\\) or you don't have enough time, now you can leave this page(because the remaining part is boring).

## Appendix

Now I describe how to compute \\(\frac{\partial y_k}{\partial z_i}\\).

Before you start, forgetting the original meanings of notations we used before may helps to avoiding confusion(like notations:\\(k\\) and \\(i\\)).

*Softmax* recap:

$$y_k = \frac{e^{z_k}}{\sum_{j=1}^J e^{z_j}}$$

Beacause \\(k\\) may takes all its possible values, so we manually divide its values into two sets:

1. \\(k = i\\)

	By applying **Quotient Rule**:

	$$\frac{\partial y_{k=i}}{\partial z_i} = \frac{e^{z_{k=i}} \sum e^{z_j} - e^{z_{k=i}}e^{z_i}}{(\sum e^{z_j})^2} = \frac{e^{z_{k=i}}}{\sum e^{z_j}} \frac{(\sum e^{z_j}) - e^{z_i}}{\sum e^{z_j}} = y_{k=i} (1 - \frac{e^{z_i}}{\sum e^{z_j}}) = y_{k=i}(1 - y_i)$$
	
	where \\(\sum\\) here represent \\(\sum_{j=1}^J\\).
	
2. \\(k \not= i\\)

	Applying **Quotient Rule** again, but notice that the numerator does not depended on \\(z_i\\):
	
	$$ {\partial y_{k \not= i} \over \partial z_i} = {0\sum e^{z_j} - e^{z_{k \not= i}}e^{z_i} \over (\sum {e^{z_j}})^2} =  {-e^{z_{k \not= i}}e^{z_i} \over (\sum {e^{z_j}})^2} = - \frac{e^{z_{k \not= i}}}{\sum {e^{z_j}}} \frac{e^{z_i}}{\sum {e^{z_j}}} = - y_{k \not= i} y_i $$
	
	here \\(\sum\\) also denote \\(\sum_{j=1}^J\\).
	

By carefully watching the \\(=\\) and \\(\not=\\) notations, we can wrap this two cases up into this much more compact form:

$$\frac{\partial y_k}{\partial z_i} = y_k (1\{k = i\} - y_i)$$

where \\(1\\{k = i\\}\\) equals to \\(1\\) if \\(k = i\\) and \\(0\\) otherwise.

You can verify this equation by hand.

Right now all derivations are done.