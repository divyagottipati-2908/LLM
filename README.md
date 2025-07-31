The rapid advancement of Large Language Models (LLMs) has significantly improved natural language 
understanding and generation. However, these models often generate responses based on statistical 
patterns rather than grounded knowledge, which can lead to inaccuracies or hallucinations. This project 
presents an enhanced approach that integrates Knowledge Graphs into a Retrieval-Augmented Generation 
(RAG) architecture to improve the reliability, factuality, and contextual awareness of LLM outputs. 
In this framework, the Knowledge Graph serves as an external, structured data source that stores 
interconnected facts and relationships relevant to the domain. During query processing, the retriever 
identifies the most relevant subgraphs or entities, which are then fed into the LLM as contextual input. 
This method allows the model to generate responses that are not only linguistically fluent but also aligned 
with verifiable facts. 
By combining the expressive capabilities of LLMs with the precision of knowledge graphs, the system 
supports intelligent, explainable, and trustworthy interactions. The proposed solution is highly adaptable 
and can be applied in domains like customer support, healthcare, legal research, and academic assistance, 
where accuracy and reasoning transparency are essential.

To evaluate the performance improvements brought by Retrieval-Augmented Generation (RAG) over 
traditional LLMs, a series of experiments were conducted across multiple tasks, including open-domain 
question answering, factual summarization, and knowledge-intensive dialogue. The setup involved using 
a pre-trained sequence-to-sequence language model (such as BART or T5) as the generator and a dense 
passage retriever (DPR) trained on question-answering data as the retriever module. The system was 
benchmarked using publicly available datasets such as Natural Questions (NQ), TriviaQA, and a curated 
factual QA dataset to measure knowledge retrieval accuracy and contextual grounding of generated 
responses. 
The first phase of the experiment compared the answer accuracy and relevance of the RAG-enhanced 
model against a standalone LLM model without external retrieval. On the Natural Questions dataset, the 
baseline LLM achieved an average top-1 accuracy of approximately 47%, whereas the RAG-enhanced 
model reached 61%. This 14% improvement highlights the impact of having up-to-date and query
relevant passages available during inference. Furthermore, qualitative inspection of the responses 
revealed that RAG-based outputs included more factually grounded information, whereas the base LLM 
tended to produce plausible-sounding but incorrect answers, a common issue associated with 
hallucination in language models. 
