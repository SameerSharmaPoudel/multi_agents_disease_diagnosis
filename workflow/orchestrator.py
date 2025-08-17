from graph_builder import GraphBuilder

class DiagnosisOrchestrator:
    def __init__(self, model_provider="groq"):
        self.graph = GraphBuilder(model_provider=model_provider)()

    def run(self, user_input: str):
        # Start execution of the LangGraph pipeline
        result = self.graph.invoke({"messages": [user_input]})
        return result
    
# orchestrator = DiagnosisOrchestrator(model_provider="groq")
# response = orchestrator.run("I have chest pain, cough, and fever for 2 days")
# print(response)