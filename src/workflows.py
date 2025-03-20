from utils import *

def _translate_basic():
    system_message = Message(SYSTEM, "I will give you text which you will need to translate to ENGLISH. Please return ONLY the given text translated to ENGLISH, with NO additional output." )
    def workflow(messages: Conversation, instance):
        text = instance["text"]

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