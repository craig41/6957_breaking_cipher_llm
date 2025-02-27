from datasets import load_dataset

def read_flores():
    dset = load_dataset("SEACrowd/flores200", trust_remote_code=True)
    return dset['test']

USER = "user"
ASSISTANT = "assistant"
SYSTEM = "system"
class Message(dict):
    def __init__(self, role, content):
        assert role in {USER, ASSISTANT, SYSTEM}
        self['role'] = role ; self['content'] = content
class Conversation(list):
    def __init__(
                    self, 
                    query = (lambda x : [warn("This conversation has not been attached to a model!"), ''][-1]), 
                    messages=[], 
                    branches=[], 
                    parent=None,
                    force_branch_at_query=False,
                ):
        self.reset(query, messages, branches, parent, force_branch_at_query)
    def reset(
                    self, 
                    query = (lambda x : [warn("This conversation has not been attached to a model!"), ''][-1]), 
                    messages=[], 
                    branches=[], 
                    parent=None,
                    force_branch_at_query=False,
                ):
        super().__init__(messages)
        self._query = query
        self.branches:list[Conversation] = branches
        self.parent = parent
        self.force_branch_at_query = force_branch_at_query

    def append(self, message):
        if not isinstance(message, Message):
            raise TypeError("Conversation should only be used to contain Message instances!")
        super().append(message)


    def query(self, user_message:str, postprocess=None, n=None, *args, **kwargs):
        return_result = list()
        for conversation in self.iter_leafs():
            postprocess = postprocess if postprocess is not None else (lambda response:response)
            conversation.append(Message(USER, user_message))
            #print(); print(conversation.messages); print()
            if n is None:
                model_message = postprocess(conversation._query(conversation.messages, *args, **kwargs))
                conversation.append(Message(ASSISTANT, model_message))
                return_result.append(model_message)
            else:
                model_messages = conversation._query(conversation.messages, n=n, *args, **kwargs)
                model_messages = [postprocess(model_message) for model_message in model_messages]
                branches = [conversation.branch() for _ in range(n)]
                for b, r in zip(branches, model_messages):
                    b.append(Message(ASSISTANT, r))
                return_result.append(model_messages)
        return return_result if len(return_result) != 1 else return_result[0]
    def branch(self, model=None, *starting_message_args, **starting_message_kwargs):
        new_branch = Conversation(query = (self._query if model is None else model), messages=[], branches=[], parent=self)
        self.branches.append(new_branch)
        if starting_message_args or starting_message_kwargs:
            new_branch.query(*starting_message_args, **starting_message_kwargs)
        return new_branch
    
    def iter_leafs(self):
        if len(self.branches) == 0:
            yield self
        else:
            for branch in self.branches:
                yield from branch.iter_leafs()

    def __add__(self, other):
        return Conversation(query = self._query, messages = list(self)+list(other), branches=[], parent=None)

    @property
    def messages(self):
        return (Conversation(query = self._query, messages=[], branches=[], parent=None) if self.parent is None else self.parent.messages) + self
        
    
    def __str__(self):
        def str_recursive(messages:Conversation, indent=0):
            chunks = []
            current_level = '\n'.join   (
                                            " "*indent+f"{message['role']}: {message['content']}"
                                            for message in messages
                                        )
            chunks.append(current_level)
            if len(messages.branches) > 0:
                for branch in messages.branches:
                    chunks.extend(str_recursive(branch, indent+1))
            else:
                chunks.append(' '*indent + "(end)")
            
            return chunks
        
        parent_message = '' if self.parent is None else "(current branch only)\n"
        return parent_message + '\n'.join([turn for turn in str_recursive(self) if len(turn) > 0]) + '\n'
    def __repr__(self):
        return self.__str__()

    def copy(self):
        return Conversation(query = self._query, messages = self, branches = self.branches.copy(), parent = self.parent)
    
    def interactive(self):
        conversation = self
        while True:
            user_text = input()
            if user_text == r"\exit":
                break
            elif user_text == r"\messages":
                print(conversation.messages)
            elif user_text == r"\print":
                print(self)
            elif user_text == r"\printall":
                print_conversation = conversation
                while print_conversation.parent is not None:
                    print_conversation = print_conversation.parent
                print(print_conversation)
            elif user_text.startswith(r"\branch"):
                next_branch = conversation
                branch_args = user_text.split(' ')[1:]
                if branch_args == []:
                    next_branch = next_branch.branch()
                elif len(branch_args) == 1:
                    for branch_step in branch_args[0].split('/'):
                        if branch_step == '..':
                            if next_branch.parent is not None:
                                next_branch = next_branch.parent
                            else:
                                warn("invalid branch directions! (parent branch not found)")
                        elif branch_step == 'new':
                            next_branch = next_branch.branch()
                        elif branch_step.isnumeric():
                            branch_number = int(branch_step)
                            if 0 <= branch_number < len(next_branch.branches):
                                next_branch = next_branch.branches[branch_number]
                            else:
                                warn(f"invalid branch directions! (child branch {branch_number} not found)")
                        else:
                            warn("invalid branch directions!")

                else:
                    warn(r"\branch takes only one argument in Conversation interactive mode!")
                conversation = next_branch
            else:
                if self.force_branch_at_query and len(self) > 0:
                    conversation = conversation.branch()
                print(conversation.query(user_message=user_text))


    
class Pipeline:
    def __init__(self,  model, workflow):
        self.model = model
        self.workflow = workflow

    def __call__(self, instance):
        conversation = Conversation(self.model, messages=[], branches=[])
        return self.workflow(conversation, instance), conversation
    


def run(data, pipeline, gold_key = (lambda instance: None), report_generator = (lambda instances, golds, outputs, conversations : "Test Complete"), per_instance_callback=None):
        instances = data
        golds = []
        outputs = []
        conversations = []
        for i, instance in enumerate((instances)):
            print(f"Processing instance {i}")
            gold = gold_key(instance)
            golds.append(gold)

            output, conversation = pipeline(instance)
            outputs.append(output)
            conversations.append(conversation)
            # # # # # 
            if per_instance_callback is not None:
                per_instance_callback(instance, gold, output, conversation)
            

        return report_generator(instances, golds, outputs, conversations)

def flatten_list(L):
    def list_flatten_iterator(L):
        if not isinstance(L, list):
            yield L
        else:
            for i in L:
                yield from list_flatten_iterator(i)
    return list(list_flatten_iterator(L))
