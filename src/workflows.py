from utils import *

def _translate_basic():
    system_message = Message(SYSTEM, "I will give you text which you will need to translate to English." )
    def workflow(messages: Conversation, instance):
        text = instance["translated"]

        messages.append(system_message)
        user_message = Message(USER, text)
        translation = messages.query(user_message)
        return translation
    return workflow

_workflows = {
    "translate": _translate_basic,
}

def load_workflow(workflow_name, *args, **kwargs):
    if workflow_name not in _workflows:
        raise ValueError(f"Workflow {workflow_name} not found!")
    return _workflows[workflow_name](*args, **kwargs)