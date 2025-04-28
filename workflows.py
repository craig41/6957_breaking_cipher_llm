from utils import *

def _translate_basic():
    system_message = Message(SYSTEM, "You are a translator that converts encoded or foreign text into plain English. When given input text, translate it accurately to English." )
    def workflow(messages: Conversation, instance):
        text = instance["text"]
        text = "Translate this text to English: " + text

        messages.append(system_message)
        translation = messages.query(text)
        return translation
    return workflow

_workflows = {
    "translate": _translate_basic,
}

def load_workflow(workflow_name, *args, **kwargs):
    if workflow_name not in _workflows:
        raise ValueError(f"Workflow {workflow_name} not found!")
    return _workflows[workflow_name](*args, **kwargs)