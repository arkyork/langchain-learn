from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline,BitsAndBytesConfig

# HuggingFaceのモデルを利用するため
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# エージェントの作成
from langchain.agents import create_agent


# hugging faceのモデルを利用

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

llm = HuggingFacePipeline.from_model_id(
    model_id = model_id,
    task = "text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
    model_kwargs={"quantization_config": bnb_config},
    
    )
chat_model = ChatHuggingFace(llm=llm)

# 以下　エージェントの作成

agent = create_agent(model = chat_model, system_prompt="あなたは数学者です。", tools = [])

response = agent.invoke({"messages":[{
    "role":"user",
    "content":"RLHFって何？日本語で答えて"
}]})

print(response)