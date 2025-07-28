from typing import Dict, List, Optional
from pydantic import BaseModel, Field, root_validator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory


# 🔍 Required symptom keys for validation
REQUIRED_SYMPTOMS = ["fever", "cough", "fatigue", "pain", "duration", "location"]


class SymptomInfo(BaseModel):
    fever: Optional[str] = Field(None, description="Fever details (e.g. 'mild', 'high')")
    cough: Optional[str] = Field(None, description="Cough description")
    fatigue: Optional[str] = Field(None, description="Fatigue level or comment")
    pain: Optional[str] = Field(None, description="Type or location of pain")
    duration: Optional[str] = Field(None, description="How long symptoms have lasted")
    location: Optional[str] = Field(None, description="Body part affected")

    @root_validator
    def validate_required_symptoms(cls, values):
        missing = [sym for sym in REQUIRED_SYMPTOMS if not values.get(sym)]
        if missing:
            raise ValueError(f"Missing required symptoms: {', '.join(missing)}")
        return values


class SymptomCollectorAgent:
    def __init__(self, llm: BaseChatModel, memory: Optional[ConversationBufferMemory] = None):
        self.llm = llm
        self.memory = memory or ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        self.parser = PydanticOutputParser(pydantic_object=SymptomInfo)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a medical assistant. Your job is to collect symptoms from a patient. "
                "Ask follow-up questions until all required information is collected: "
                f"{', '.join(REQUIRED_SYMPTOMS)}. Output your final answer as a JSON matching this schema:\n"
                "{format_instructions}"
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]).partial(format_instructions=self.parser.get_format_instructions())

        self.chain = self.prompt | self.llm | self.parser

    def run(self, state: Dict) -> Dict:
        messages = state.get("messages", [])
        for msg in messages:
            self.memory.chat_memory.add_message(msg)

        # Attempt to extract structured symptoms from conversation
        try:
            output = self.chain.invoke({"input": "Please summarize the symptoms collected so far."})
            return {
                "messages": messages,
                "symptoms": output.dict(),
                "agent_status": "complete"
            }
        except Exception as e:
            clarification_prompt = f"Some required symptoms are missing. {str(e)} Please ask the user for clarification."
            ai_response = self.llm.invoke([AIMessage(content=clarification_prompt)])
            self.memory.chat_memory.add_message(ai_response)
            return {
                "messages": [*messages, ai_response],
                "symptoms": {},
                "agent_status": "incomplete"
            }
