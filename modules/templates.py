from pydantic import BaseModel, Field, create_model
from typing import List
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate


SYSTEM_PROMPT = """
You are FlowGPT, an assistant that can generate documentation for nodes based on provided context information:
You extract data and return it in JSON format, according to provided JSON schema.
Use your own knowledge to give more insights into high level technical concepts.
"""

USER_PROMPT = """
Context information is below.
---------------------
{context_str}
---------------------
Given the context information, answer the query.
Query: {query_str}
Answer:
"""

TEXT_QA_PROMPTS = ChatPromptTemplate(
    message_templates=[
        ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM),
        ChatMessage(content=USER_PROMPT, role=MessageRole.USER),
    ]
)

PROMPT_DOCUMENTATION = """
Provide a docstring for a method named FUNCTION of the node in 1-2 sentences, focusing solely on the node's purpose
and high-level functionality. Your description should abstractly convey the node's capabilities and aims without referencing
specific methods by name, detailing inputs or outputs, or discussing nested functions.
Emphasize the node's overall contribution in a conceptual manner, aiming for a thematic overview rather than a
procedural breakdown. Ensure the narrative centers on the node's functionality, steering clear of method-specific language.
"""

PROMPT_INPUT_PARAMETER = """
For the parameter, provide not only a description of its function but also its importance in the overall operation (1-2 sentences).
Include explanations on how it affects the node's execution and results.
Don't mention parameter min, max and default values and parameter type.
For collapsed parameters like "input_blocks.i.j.transformer_blocks.k.attn2.to_q" additionally explain the range of each index.
"""

PROMPT_OUTPUT_PARAMETER = "Detail parameter, including a description of its function and significance (1-2 sentences)."

PROMPT_INPUT_TYPES = """
Elaborate on each type from the INPUT_TYPES of the node. The types names are specified in the list: "{input_parameters}".
Ensure to accurately list all names exactly as they appear, without altering spellings or resorting to generalizations.
It's crucial that each type name is presented and elaborated on precisely as provided.
"""

PROMPT_OUTPUT_TYPES = """
Elaborate on each type from the RETURN_TYPES of the node. The types names are specified in the list: "{output_parameters}".
Ensure to accurately list all names exactly as they appear, without altering spellings or resorting to generalizations.
It's crucial that each type name is presented and elaborated on precisely as provided.
Additionally, if analysis of the node's code reveals that it returns a dictionary in the form {{"ui": ...}},
you must add a new parameter named ui (without any variations) to the list.
Describe the ui parameter in the same manner as the other parameters.
"""

PROMPT_INFRA_TYPE = """Respond in one word: GPU if GPU is recommended to run the node, CPU otherwise.
Typically, use GPU for nodes involving pytorch/tensorflow, and CPU for nodes accessing models via API or not using models."""

PROMPT_PYTHON_DTYPE = """
Identify the precise Python data type of the parameter,
including both standard types (e.g., int, str) and types from libraries (e.g., torch.Tensor, torch.nn.Module).
For collections, detail the types of contained elements (e.g., Dict[str, Tuple[int, torch.Tensor]] instead of just dict).
Avoid vague descriptors like object or Any, seeking specificity.
Refrain from using ComfyUI-specific references (e.g., comfy.sd.vae).
"""


class NodeInputParameter(BaseModel):
    """Data model for a node input parameter"""

    name: str | None = Field(description="Parameter name", default=None)
    documentation: str | None = Field(description=PROMPT_INPUT_PARAMETER, default=None)
    python_dtype: str | None = Field(description=PROMPT_PYTHON_DTYPE, default=None)


class NodeOutputParameter(BaseModel):
    """Data model for a node output parameter"""

    name: str | None = Field(description="Parameter name", default=None)
    documentation: str | None = Field(description=PROMPT_OUTPUT_PARAMETER, default=None)
    python_dtype: str | None = Field(description=PROMPT_PYTHON_DTYPE, default=None)


class Node(BaseModel):
    """Data model for a node"""

    name: str | None = Field(description="Node name", default=None)
    documentation: str | None = Field(description=PROMPT_DOCUMENTATION, default=None)
    input_types: List[NodeInputParameter] | None = Field(description=PROMPT_INPUT_TYPES, default=None)
    output_types: List[NodeOutputParameter] | None = Field(description=PROMPT_OUTPUT_TYPES, default=None)
    infra_type: str | None = Field(description=PROMPT_INFRA_TYPE, default=None)

    @classmethod
    def format_field_description(cls, field_name: str, **kwargs) -> None:
        field = cls.__fields__[field_name]
        field.field_info.description = field.field_info.description.format(**kwargs)

    @classmethod
    def restore_field_description(cls, field_name: str) -> None:
        original_template = globals().get("PROMPT_" + field_name.upper())
        cls.__fields__[field_name].field_info.description = original_template


PROMPT_EVAL = """
Your task is to evaluate the response's quality.
Assign a score and offer short feedback based on the criteria outlined in the provided JSON schema.
Each question is worth 5 points.
It's important to remember that the model generating the response had a deeper understanding of the subject.
Do not attempt to correct areas outside your expertise; adhere strictly to the questions.
When addressing each question from the provided JSON schema, focus solely on that specific question without referencing
other questions.

Response:

{response}
"""

eval_questions = {
    "node_name": "Is the node name strictly equal to {node_name}?",
    "node_doc": """Does node documentation correspond to the prompt "{prompt_documentation}"? Responses that delve into
specific parameters, settings, or detailed methodological descriptions should be seen as not aligning with the desired criteria""",
    "input_types": """Verify "input_types" of the node match list: {input_parameters} by ensuring "name" fields correspond
to the provided list elements.
It should be exact match without any synonyms, substitutions, additional information or letters case changes.
Make sure that all names from the provided list are present in the response, identify any mismatches and omissions""",
    "output_types": """Verify "output_types" of the node match list: {output_parameters} by ensuring "name" fields correspond
to the provided list elements.
It should be exact match without any synonyms, substitutions, additional information or letters case changes.
Make sure that all names from the provided list are present in the response, identify any mismatches and omissions.
Exception: if node output_types contain parameter named "ui" anticipate its absence in the provided list,
and don't deduct points for mismatch. However, ensure that the "name" for the "ui" parameter is specifically set to "ui".
If this condition is met, do not mention the "ui" parameter in your feedback""",
    "input_types_doc": """Does each of node input_types documentation match the prompt "{prompt_input_parameter}"?""",
    "output_types_doc": """Does each of node output_types documentation match the prompt "{prompt_output_parameter}"?""",
    "python_dtype": """Does python_dtype for each entry in node input_types and output_types match the definition
"{prompt_python_dtype}"?""",
    "infra_type": """Is infra_type identified in one word: CPU or GPU?""",
}


class Question(BaseModel):
    """Data model for a question about {question_name}"""

    feedback: str | None = Field(default=None)
    score: int = Field(description="Integer score for this question, ranging for 0 to 5", default=0)


def create_eval_schema(**kwargs) -> BaseModel:
    """Dynamically creates data model for a list of evaluation questions"""
    questions = {}
    for k, v in eval_questions.items():
        question = create_model(
            f"{Question.__name__}_{''.join(x.capitalize() for x in k.split('_'))}",
            __base__=Question,
            feedback=(str | None, Field(default=None, description=v.format(**kwargs))),
        )
        question.__doc__ = Question.__doc__.format(question_name=k)
        questions[k] = question
    schema = create_model("Evaluation", **{k: (v, ...) for k, v in questions.items()})
    schema.__doc__ = "Data model for evaluation"
    return schema


PROMPT_RETRY = """
Please correct the original response below based on the specified feedback, ensuring to incorporate these corrections alongside
the original content that doesn't require changes. The goal is to produce a complete, updated response that addresses the
feedback while retaining the accurate parts of the initial statement.

Original response:
{response}

Feedback for corrections:
{feedback}
"""
