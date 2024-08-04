"""OpenAI wrapper"""

# License

# Copyright 2024 Marc Weinberg https://marc-w.com

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import logging
from typing_extensions import override

from openai import AssistantEventHandler, OpenAI


logging.basicConfig()
LOG_NAME = "openai_mgr"
logger = logging.getLogger(LOG_NAME)
logger.setLevel(logging.INFO)


client = OpenAI()


def attribute_puller(_path=(), _obj=None) -> object|list|dict|str|None:
    """
    Safely get nested attributes in an object using an 
    iterable of string values to define a path

    :param tuple|list _path: path to object, defaults to ()
    :param object _obj: Object to consume, defaults to None
    :return object|list|dict|str|None: Object data at the path, or None
    """
    if not _path or not _obj:
        return None
    ret = None
    for attr in _path:
        if ret:
            ret = getattr(ret, attr, None)
        else:
            ret = getattr(_obj, attr, None)
        if not ret:
            logger.error("Attribute not found: %s on object %s", attr, str(_obj))
            break
    return ret


class EventHandler(AssistantEventHandler):
    """
    Event handler originally sourced from https://openai.com/ forums.

    Used to control the stream and return the response form OpenAI.

    Added code to retrieve the data.

    :param class AssistantEventHandler: Iterator
    """

    def __init__(self, parse_response=None):
        super().__init__()
        self.parse_response=parse_response

    @override
    def on_text_created(self, text) -> None:
       logger.info("\nassistant > on_text_created > %s",str(text))

    @override
    def on_tool_call_created(self, tool_call):
       logger.info("\nassistant > on_tool_call_created > %s \n", str(tool_call.type))

    @override
    def on_message_done(self, message) -> None:
        message_content = message.content[0].text
        annotations = message_content.annotations
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(
                annotation.text, f"[{index}]"
        )

        self.parse_response = message_content.value


class GPT:
    """Main GPT"""

    def __init__(self, *args, **kwargs):  # pylint: disable=W0613
        """
        init
        """
        self.assistant = None
        self.message_file = None
        self.thread = None


    def create_assistant(self, name=None, instructions=None, **kwargs) -> object|None:
        """
        Returns assistant, creates if needed.

        :param string name: _description_, defaults to None
        :param string instructions: _description_, defaults to None
        :raises RuntimeError: Don't mess up.
        :return object|None: Returns assistant object, else None
        """
        # use assistant if available
        if not name and self.assistant:
            logger.info("Using assistant found in object")
            return self.assistant

        if not name:
            raise RuntimeError("Missing name for assistant creation in GPT")

        # Assigns None if not found.
        # Looking for existing assistant for the auth matching name
        if asst:= self.find_assistant(name):
            logger.info("Using existing assistant found on service")
            self.assistant = asst
            return self.assistant
        # Set arguments for the assistant
        tools = kwargs.get("tools", [{"type": "file_search"}])  # file stuff

        _model = kwargs.get("model", "gpt-3.5-turbo")  # control the model

        self.assistant = client.beta.assistants.create(
            name=name,
            model=_model,
            tools=tools,
            instructions=instructions
        )
        logger.info("Created new object, added to object")

        return self.assistant

    def find_assistant(self, id_or_name=None) -> object|None:
        """
        Looks for an existing assistant by name or id value.

        :param str id_or_name: ID or pretty name of the created assistant, defaults to None
        
        ex:
            id: asst_7KIgsjSjeuhsguBwI7UIXy7
            name "Pretty Name"

        :return object|None: <class 'openai.types.beta.assistant.Assistant'>
        """
        if not id_or_name:
            return None
        assistants =  client.beta.assistants.list()
        if not assistants:
            return None

        for asst in assistants:
            # keys via __dict__ from OpenAI object
            #     id='NOT_SET'
            #     created_at=999
            #     description=None
            #     instructions=None
            #     metadata={}
            #     model='gpt-3.5-turbo'
            #     name='NOT_SET'
            #     tools=[FileSearchTool(type='file_search', file_search=None)]
            #     response_format='auto'
            #     temperature=1.0
            #     tool_resources=ToolResources(code_interpreter=None
            #     file_search=ToolResourcesFileSearch(vector_store_ids=[]))top_p=1.0
            if asst.id == id_or_name:
                return client.beta.assistants.retrieve(assistant_id=asst.id)
            if asst.name == id_or_name:
                return client.beta.assistants.retrieve(assistant_id=asst.id)
        return None

    def attach_file(self, _path):
        """
        Open a local file and add to the OpenAI client

        :param _type_ _path: _description_
        :return None: No return
        """
        self.message_file = client.files.create(
            # to-do: look to change to `with` usage
            file = open(_path, "rb"), purpose="assistants"
        )

    def create_thread(self, content=None, **kwargs):
        """
        Create thread to associate the message and data for OpenAI to process.

        :param _type_ content: _description_, defaults to None
        :raises RuntimeError: DOn't mess up.
        """
        role = kwargs.get("role", "user")
        if not content:
            raise RuntimeError("No content provided for create_thread [hqt673]")

        # Create a thread and attach the file to the message
        self.thread = client.beta.threads.create(
            messages=[
                {
                "role": role,  
                "content": content,
                # Attach the file to the message.
                "attachments": [
                    {
                        "file_id": self.message_file.id, 
                        # TO-DO: Can this next line be removed?
                        "tools": [{"type": "file_search"}] }
                ],
                }
            ]
        )

    def get_response(self):
        """
        Get the final response form OpenAI, assembled with the aid of
        the EventHandler callback.

        :raises RuntimeError: Don't mess up
        :raises RuntimeError: Don't mess up
        :raises RuntimeError: Don't mess up
        :return ???: Stream response
        """

        # Validate ready
        if not self.thread:
            raise RuntimeError("get_response did not receive thread")
        if not self.assistant:
            raise RuntimeError("get_response did not receive assistant")

        parse_response = None

        with client.beta.threads.runs.stream( # pylint: disable=E1129
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            event_handler=EventHandler(parse_response=parse_response),
        ) as stream:
            stream.until_done()

            completion_tokens = (
                '_AssistantEventHandler__current_run', 'usage', 'completion_tokens')
            prompt_tokens = ('_AssistantEventHandler__current_run', 'usage', 'prompt_tokens')
            total_tokens = ('_AssistantEventHandler__current_run', 'usage', 'total_tokens')

            logger.info(
                "completion_tokens: %s", str(attribute_puller(completion_tokens, stream)))
            logger.info(
                "prompt_tokens: %s", str(attribute_puller(prompt_tokens, stream)))
            logger.info(
                "total_tokens: %s", str(attribute_puller(total_tokens, stream)))

            return stream.parse_response

def send_pdf_to_openai(
        assistant_name=None,
        assistant_instructions=None,
        thread_content=None,
        file_path=None,
        gpt_obj=None,
    ):
    """
    Wrapper to send a file, instructions, and assistant info to parse a file.

    :param string assistant_name: Simple name of the assistant, defaults to None
    :param string assistant_instructions: instruction for setting
        up the assistant (you are a thing...), defaults to None
    :param string thread_content: Description of the work for the assistant
        to be performed, defaults to None
    :param string file_path: Location of the file on the local system, defaults to None
    :param object gpt_client: Location of the file on the local system, defaults to None
    :return string: Complete generated message
    """
    if not gpt_obj:
        gpt_obj = gpt = GPT()

    gpt_obj.create_assistant(
        name=assistant_name,
        instructions=assistant_instructions,
    )
    gpt_obj.attach_file(file_path)
    gpt_obj.create_thread(
        content=thread_content,
    )
    res = gpt.get_response()
    return res
