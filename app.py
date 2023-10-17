import gradio as gr
from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain

config = {'max_new_tokens': 100, 'temperature': 0}
llm = CTransformers(model='TheBloke/Mistral-7B-Instruct-v0.1-GGUF', model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", config=config)

template = """<s>[INST] You are a helpful, respectful and honest assistant. Answer exactly in few words from the context
Answer the question below from context below :
{context}
{question} [/INST] </s>
"""
prompt = PromptTemplate(template=template, input_variables=["question","context"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

def question_answer(context: str, question: str):
  print(context, question)
  response = llm_chain.run({"question":question, "context":context})
  print(response)
  return response

theme = gr.themes.Default(
    primary_hue="indigo",
    secondary_hue="pink",
    neutral_hue="slate",
)

with gr.Blocks(theme=theme) as interface:
  context = gr.Textbox(lines=5, placeholder="On August 10 said that its arm JSW Neo Energy has agreed to buy a portfolio of 1753 mega watt renewable energy generation capacity from Mytrah Energy India Pvt Ltd for Rs 10,530 crore.", label="Context")
  question = gr.Textbox(placeholder="What company is buyer and seller here", label="Question")
  answer = gr.Textbox(placeholder="Answer will be here", label="Answer")
  ask_button = gr.Button("Ask (this might take a minute since it's using CPU)")
  ask_button.click(fn=question_answer, inputs=[context, question], outputs=answer)

interface.launch(debug=True)