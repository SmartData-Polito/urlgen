# URLGEN -- Towards Automatic URL Generation Using GANs

*This is a work in progress. Further updates will increase the number of parameters available to control de model.*

URLs play an essential role on the Internet, allowing the access to Web resources. Automatically generating URLs is helpful in various tasks, such as for application debugging, API testing, and blocklist creation for security applications. Current testing suites deeply embed specialists' domain knowledge to generate suitable URLs, resulting in an ad-hoc solution for each given application. These tools thus require heavy manual intervention, with the expensive coding of rules that are hard to create and maintain.

In this work, we introduce URLGEN, a system that uses Generative Adversarial Networks (GANs) to tackle the automatic URL generation problem. URLGEN is designed for web API testing, and generates URL samples for an application without any system expertise. 

It leverages Long Short-Term Memory (LSTM) and Convolutional Neural Network (CNN) architectures, augmented by an embedding layer that simplifies the URL learning and generation process. We show that URLGEN learns to generate new valid URL from samples of real URLs, without requiring any domain knowledge and following a pure data-driven approach. We compare the GAN architecture of URLGEN against other possible design options and show that the LSTM architecture can better capture the correlation among characters in URLs, outperforming previously proposed solutions.
%URLGEN includes an embedding layer that maps characters into a vector space. This step speeds up the GAN convergence, while improving both the system discriminative and generative abilities. 

We show that the URLGEN approach can be extended to other scenarios, which we illustrate with two use cases, i.e., cybersquatting domain prediction and URL classification.

## Usage

The script.py has to positional parameter. The first is the file containing samples from the target API and the second the number of new samples to generate:

Example

``` bash
python script.py example-dataset.txt 100
```


