# MCLN
## Multimodal Counterfactual Learning Network for Multimedia-based Recommendation

<p align="left">
    <img src='https://img.shields.io/badge/key word-Recommender Systems-green.svg' alt="Build Status">
    <img src='https://img.shields.io/badge/key word-Multimodal User Preference-green.svg' alt="Build Status">
    <img src='https://img.shields.io/badge/key word-Counterfactual Learning-green.svg' alt="Build Status">
    <img src='https://img.shields.io/badge/key word-Spurious Correlation-green.svg' alt="Build Status">
</p>

Multimedia-based recommendation (MMRec) utilizes multimodal content (images, textual descriptions, etc.) as auxiliary information on historical interactions to determine user preferences. Most MMRec approaches predict user interests by exploiting a large amount of multimodal contents of user-interacted items, ignoring the potential effect of multimodal content of user-uninteracted items. As a matter of fact, there is a small portion of user preference-irrelevant features in the multimodal content of user-interacted items, which may be a kind of spurious correlation with user preferences, thereby degrading the recommendation performance. In this work, we argue that the multimodal content of user-uninteracted items can be further exploited to identify and eliminate the user preference-irrelevant portion inside user-interacted multimodal content, for example by counterfactual inference of causal theory. Going beyond multimodal user preference modeling only using interacted items, we propose a novel model called Multimodal Counterfactual Learning Network (MCLN), in which user-uninteracted items’ multimodal content is additionally exploited to further purify the representation of user preference-relevant multimodal content that better matches the user’s interests, yielding state-of-the-art performance. Extensive experiments are conducted to validate the effectiveness and rationality of MCLN.

We provide tensorflow implementation for MCLN.

### Before running the codes, please download the [datasets](https://www.aliyundrive.com/s/V1RPArCZQYt) and copy them to the Data directory.

## prerequisites

- Tensorflow 1.10.0
- Python 3.6
- NVIDIA GPU + CUDA + CuDNN
