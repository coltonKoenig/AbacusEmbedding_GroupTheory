# Algebraic Abacus Embeddings for Symbolic Reasoning in Language Models
## Colton H. Koenig

## 1. Abstract
Recent work on Abacus embeddings has shown promise in improving language models' performance on numerical tasks by encoding numbers through structured, interpretable vector representations. This project advances that line of research by incorporating algebraic structure such as associativity, commutativity, and distributivity into Abacus embeddings. Drawing inspiration from group theory and Lie algebras, we construct numerical embedding spaces where operations like addition and multiplication correspond to algebraic transformations rather than vector additions. To implement this, we can represent numbers as structured tensors or group elements, where arithmetic relations are preserved operators. This project aims to investigate two core questions, as follows: Can these embeddings compress numerical content more efficiently than standard tokenization such as word2vec without degrading performance? What are the geometric properties of the Abacus embedding space, and how do they compare to learned embeddings? We can visualize the embedding space and compare effectiveness between Abacus embedding methods and other commonly used embedding methods. Preliminary results suggest that Abacus embeddings can enhance the efficiency and interpretability of mathematical reasoning models, potentially improving symbolic and proof-based reasoning.


## 2. Background
Traditional embedding methods in language models treat numerical inputs as unstructured tokens, resulting in representations that fail to capture the underlying mathematical relationships between values. This limitation get in the way of generalization in tasks requiring numerical or symbolic reasoning, as standard embeddings do not preserve many standard arithmetic properties.

Abacus embeddings provide a structured alternative by encoding numbers as digit-wise representations in fixed bases, preserving their composition and allowing more consistent processing by language models.This project extends the Abacus framework by embedding algebraic properties such as commutativity, associativity, and distributivity into the vector space. By benchmarking against symbolic reasoning tasks and analyzing the embedding geometry through dimensionality reduction techniques, we aim to assess whether algebraically structured Abacus embeddings improve generalization of a models’ ability to properly handle algebraic expressions.


## 3. Hypothesis and Scientific Method

Hypothesis
Embedding numerical quantities with explicitly encoded algebraic structure such that operations preserve mathematical identities like associativity, commutativity, and distributivity will improve a language model’s ability to generalize in symbolic reasoning tasks and theorem proving, relative to unstructured token or learned embeddings.

Testing Strategy
Firstly, to implement the embedding design, structured embeddings of numbers will be created where operations are algebraic transformations. This will in part be inspired by group theory ideas (e.g., embeddings lie in a group, addition/multiplication are homomorphisms), Lie algebras (e.g., map operations to vector fields), and explicitly coding of standard algebraic properties such as associativity, commutativity, and distributivity.

A mathematical dataset will also be needed for training and testing. I will start with a dataset of relatively basic algebraic equations and theorems. This may include TPTP Theorem proving, miniF2f, or ProofWriter datasets, and manually-generated expressions using SymPy to train and test for specific goals.

Next, in the way of benchmarks, we will compare loss and efficiency to token-based embeddings (e.g., standard embeddings, word2vec), learned embeddings without specific structure, pretrained symbolic embedding models (e.g., math2vec, logic2vec), and potentially random embeddings as a sanity check. Symbolic arithmetic benchmarks may include generalizing associativity, commutativity, and distributivity outside of training data or of longer expressions. This can be extended to multi-step algebraic reasoning, for example, expression simplification. Comparisons can also be made to inductive theorem proving using datasets provided by ProofWriter or miniF2F, for example. Ablation may also be a viable strategy- this could include removing a group property (e.g., commutativity) to observe performance degradation.

Metrics
Accuracy: On symbolic equation evaluation
Generalization: To unseen combinations of numeric values and symbolic forms
Sample efficiency: Number of examples required to reach target accuracy
Ablation sensitivity: Performance when structural properties (e.g., commutativity) are disabled

Visualization of Embedding Space
To test whether the algebraic structure is learned and reflected in the geometry of the embedding space, we can use dimensionality reduction techniques such as UMAP or PCA to express embeddings of numbers visually. We can use vector arithmetic in embedding space to verify, for example, checking if equal-valued expressions are grouped together (e.g., 2 + 2, 1 + 3). We can also check if embeddings are smoothly distributed or if certain operations lead to discontinuities- we can compare this to other embedding methods mentioned above.


## 4. Outcomes
We expect models that use algebraic Abacus embeddings will require fewer examples to learn symbolic arithmetic rules and show better generalization to novel numeric expressions and proofs. The Abacus embedding model should also exhibit algebraic structure in the embedding space and preserve algebraic properties. We should be able to verify the effectiveness of the Abacus embedding model with the benchmarks and metrics as described above, as well as visual verification of the embedding space.

If these benchmarks and metrics are not met, we’d question if the human-based interpretation of these algebraic concepts are the most efficient interpretation for a language model. We’d question if group theory based properties are sufficient to capture the algebraic structure of certain expressions and equations- the ideas encoded may be too rigid. The model may require auxiliary objectives for better learning, such as autoencoding or algebraic regularization to enforce structure.

## 5. Minimal Viable Example
As an easy to obtain proof of concept, we can implement two types of embeddings for digits 0-20, say. To start, one of these embeddings can be a random embedding, but the other will be a basic implementation of Abacus embeddings. This basic implementation can be base-10 digit-slot encoding, where each digit in a number is given its own learned embedding, and the final number vector is the sum. We can then choose to encode a simple algebraic property, such as commutativity (e.g., f(a) + f(b) = f(a+b)). We can repeat this with random vectors of similar dimension and evaluate differences numerically, such as loss evaluation.


## 6. Timeline
Week 1: 
Developing benchmark testing by implementing structured and random embeddings for numbers followed by evaluation of basic arithmetic properties.<br>
Implementation of a basic Abacus embedding in base-10 digit-slot encoding.<br>
Further review of existing literature.<br>
Week 2:<br>
Define algebraic operations in Abacus embedding model (e.g., associativity, commutativity, distributivity).<br>
Encode with Lie group-style operators (e.g., matrix representations of operators)<br>
Develop a dataset of symbolic arithmetic problems<br>
Week 3:<br>
Use or adapt ProofWriter, miniF2F, or a custom symbolic algebra dataset.<br>
Generalization to longer expressions or expressions not found in the dataset<br>
Visualize ablation effects (e.g., removing commutativity)<br>
Week 4:<br>
Visualizations of embeddings<br>
Evaluating efficiency by plotting how many samples are needed for sufficient learning.<br>
Evaluating accuracy compared to other embedding methods<br>
Presentation formulation<br>


## 7. Progress Update (4/16)
Firstly, I worked through the GitHub repository created by mcleish7 from [1] to further understand the implementation of learned digit-wise Abacus embeddings in practice. Coding-wise, I’ve done minor implementation and testing of foundational components of Abacus digit-wise embedding. Seen in main.ipynb, the Abacus embedding function takes into account fixed positional weighting, encoding explicit place value into an input integer. This ensures, for example, that 12 and 21 are embedded as different vectors. Minor testing has also been done with sinusoidal and one-hot positional embedding, resulting in very high loss. Evaluation functions have been implemented for commutativity, associativity, and distributivity where loss is the difference between, for example, embedded(a) + embedded(b) and embedded(a+b). The loss of the Abacus embedding fluctuates greatly based on the type of positional embedding used, but can be constructed in a way that results in lower loss of than the random embedding method. Furthermore, both Abacus and random embeddings have been visualized using PCA, confirming that the structured embeddings display a more coherent spatial distribution. As stated, please see main.ipynb for implementation progress.<br>
From this minor implementation, the next step is to construct a usable and robust method for digit-wise encoding, which can be done after further experimentation/comparison and literature review. Afterwards, I can implement learned weights inspired by [1], and incorporate group theory structure into the training process. I have found a training set of algebraic expressions created by [1] which will be useful in the training process. Reflecting on the original proposal, I may have been overzealous- I’m thinking about adjusting my goals to focus primarily on the properties of commutativity, associativity, and distributivity and their respective evaluations. I am still apprehensive on how exactly I am to construct the layers that ensure the group theory properties listed before.<br>
My classmate Levi Sprung has expressed interest in joining my project, which I am open to. If he does decide to join, I believe we can achieve a further-developed goal than what I had personally. This progress update has been completed under the assumption this project remains individual. Note, section 8 has been updated with recent LLM usage. VS Code Copilot was used in line completion and, subsequently, minor idea generation.<br>


## 8. LLM Usage
Chat-GPT model 4o was used for idea generation, assistance writing the abstract, background, and hypothesis sections, and generating a structure for the rest of the proposal. The link to the transcript is as follows:
[https://chatgpt.com/share/67e59c69-d608-800b-8e0f-88e767ae49c6]

Chat-GPT model 4o was also used to gather resources. The link to this transcript is as follows:
[https://chatgpt.com/share/67e5b2cf-8cec-800b-8860-17f075ab1547]

Chat-GPT model 4o was used to generate code to plot PCA results for Progress Update (4/16):
[https://chatgpt.com/share/680032bc-cc24-800b-90f2-66ccfb40c415]


## 9. Bibliography

[1] McLeish, Sean. Transformers Can Do Arithmetic with the Right Embeddings, 27 May 2024, arxiv.org/html/2405.17399v1.

[2] AI Papers Academy. YouTube, YouTube, 31 May 2024, www.youtube.com/watch?v=ENykjLzRpoI.

[3] Wallace, Eric, et al. “Do NLP Models Know Numbers? Probing Numeracy in Embeddings.” arXiv.Org, 18 Sept. 2019, arxiv.org/abs/1909.07940. 

[4] Elmo. “Abacus Embeddings: How to Make LLMS Infallible Mathematicians?!” Medium, AI Advances, 29 May 2024, ai.gopubby.com/abacus-embeddings-how-tomake-llms-infallible-mathematicians-2c586a35517b.

[5] “Word Embeddings in NLP.” GeeksforGeeks, 5 Jan. 2024, www.geeksforgeeks.org/word-embeddings-in-nlp/?utm_source=chatgpt.com.

[6] Tomar, Ankur. “A Math-First Explanation of Word2vec.” Medium, Analytics Vidhya, 30 July 2019, medium.com/analytics-vidhya/maths-behind-word2vec-explained-38d74f32726b.
