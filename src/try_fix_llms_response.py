from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

chat = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key, temperature=0.7)

memory = ConversationBufferMemory()

# Crea la catena conversazionale
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True  
)

# Primo turno
output1 = conversation.predict(input="Quali sono gli effetti del cambiamento climatico sugli oceani?")
print("\nAssistant:", output1)

# Secondo turno (follow-up)
output2 = conversation.predict(input="E per le barriere coralline?")
print("\nAssistant:", output2)