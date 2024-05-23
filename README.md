# GENAI-LLAMA3
Interactive AI-Powered Blog Article Generator Using Streamlit and LLAMA

**Code Overview:**
The application defines a function getLLAMAresponse that interacts with the LLAMA model, processes user inputs, and generates the desired text. The Streamlit interface collects the necessary inputs and displays the generated article.

The BitsAndBytesConfig class, is configured with load_in_4bit=True and bnb_4bit_compute_dtype=torch.bfloat16, and is used for quantization in neural network models. Quantization is a technique used to reduce the memory footprint and computational complexity of models by representing weights and activations with fewer bits. However, reducing the precision of weights and activations can potentially degrade the model's performance, particularly in tasks where high precision is crucial, such as in natural language understanding or computer vision tasks. In summary, while using 4-bit quantization and bfloat16 computation may lead to a slight reduction in performance, the significant memory savings make it a valuable technique, especially in resource-constrained environments.

<img width="1429" alt="image" src="https://github.com/miramnair/genai-llama3/assets/128325004/65f47adc-98f2-4aa3-b6bb-e2aa4b016248">

