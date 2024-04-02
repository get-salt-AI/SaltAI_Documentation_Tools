import json
import traceback
from typing import Any, Dict, Tuple
from collections import Counter
from dataclasses import dataclass
from llama_index.core.base.response.schema import PydanticResponse
from llama_index.core.prompts import PromptTemplate
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.llms.llm import LLM
from llama_index.program.openai import OpenAIPydanticProgram
from . import ROOT
from .templates import (
    PROMPT_EVAL,
    PROMPT_RETRY,
    PROMPT_DOCUMENTATION,
    PROMPT_INPUT_PARAMETER,
    PROMPT_OUTPUT_PARAMETER,
    PROMPT_PYTHON_DTYPE,
    PROMPT_INFRA_TYPE,
    TEXT_QA_PROMPTS,
    eval_questions,
    create_eval_schema,
    Node,
)
from .utils import update_node_info, format_response, id_to_node, node_to_id


@dataclass
class EvaluatedResponse:
    response: PydanticResponse | None = None
    score: int = -1
    feedback: str | None = None
    passing: bool = False


class NodeQueryEngine:
    """
    Implements advanced querying for node documentation.
    Process includes retrieveing context from code database, querying llm with structured output guidance,
    self correcting via evaluation, refining query with evaluation output, obtaining node usage info from pipelines database,
    postprocessing and formatting output into the desired format.
    Querying works iteratively until num_retries is reached or response is satisfactory.
    """

    def __init__(self, indexes: VectorStoreIndex, num_retries: int, top_k: int) -> None:
        retrievers = [index.as_retriever(similarity_top_k=top_k) for index in indexes]
        retriever = QueryFusionRetriever(retrievers=retrievers, similarity_top_k=top_k, num_queries=1)
        self.engine = RetrieverQueryEngine.from_args(retriever, text_qa_template=TEXT_QA_PROMPTS)
        self.max_score = len(eval_questions) * 5
        self.evaluator = StructuredEvaluator(eval_template=PROMPT_EVAL, score_threshold=self.max_score)
        self.num_retries = num_retries

    def query(self, node_name: str, node_info: Dict[str, Any]) -> Dict[str, Any]:
        node_info = update_node_info(node_info)
        inputs_str = str([k for d in node_info["input_parameters"].values() for k in d.keys()])
        outputs_str = str(list(node_info["output_parameters"].keys()))
        Evaluation = create_eval_schema(
            prompt_documentation=PROMPT_DOCUMENTATION,
            prompt_input_parameter=PROMPT_INPUT_PARAMETER,
            prompt_output_parameter=PROMPT_OUTPUT_PARAMETER,
            prompt_python_dtype=PROMPT_PYTHON_DTYPE,
            prompt_infra_type=PROMPT_INFRA_TYPE,
            node_name=node_name,
            input_parameters=inputs_str,
            output_parameters=outputs_str,
        )
        Node.format_field_description("input_types", input_parameters=inputs_str)
        Node.format_field_description("output_types", output_parameters=outputs_str)
        self.engine._response_synthesizer._output_cls = Node
        self.evaluator.llm_program._output_cls = Evaluation
        query = f"Analyze node {node_name}."
        error = None
        best_response = EvaluatedResponse()
        for _ in range(self.num_retries):
            try:
                response = self.engine.query(query)
                evaluated_response = self.evaluator.evaluate_response(response)
            except Exception:
                error = traceback.format_exc()
                continue
            error = None
            if evaluated_response.score > best_response.score:
                best_response = evaluated_response
            if evaluated_response.passing:
                response = format_response(
                    node_name, response, node_info, f"{evaluated_response.score}/{self.max_score}"
                )
                response = self.generate_usage_information(response)
                self.reset()
                return response
            query = PROMPT_RETRY.format(response=best_response.response, feedback=best_response.feedback)
        if error is not None and best_response.response is None:
            self.reset()
            return {"_debug_info": {"_eval_score": f"0/{self.max_score}", "_has_unknown": True, "_error": error}}
        response = format_response(
            node_name, best_response.response, node_info, f"{int(best_response.score)}/{self.max_score}"
        )
        response = self.generate_usage_information(response)
        self.reset()
        return response

    def generate_usage_information(self, response: Dict[str, Any], most_common: int = 10, max_pipelines: int = 10):
        common_nodes = []
        with open(ROOT / "data" / "pipelines_db.json", "r") as file:
            json_data = json.load(file)
        for json_value in json_data:
            # Look for nodes that are frequently used with this node
            current_pipeline = json.loads(json_value["json"])
            node_id = node_to_id(response, current_pipeline)
            for _, from_id, _, to_id, _, _ in current_pipeline["links"]:
                if from_id == node_id:
                    common_nodes.append(id_to_node(to_id, current_pipeline))
        common_nodes = Counter(common_nodes)
        common_nodes = common_nodes.most_common(most_common)
        usage_doc = ""
        if isinstance(common_nodes, list):
            common_nodes = ",".join([x[0] for x in common_nodes if x[0] is not None])
        response.update({"common_nodes": common_nodes, "usage_doc": usage_doc})
        return response

    def reset(self):
        Node.restore_field_description("input_types")
        Node.restore_field_description("output_types")


class StructuredEvaluator:
    """
    Implements documentation evaluation against a set of questions. Returns score and feedback in structured format.
    Guides documentation generating llm with a list of notes for correction.
    """

    def __init__(
        self,
        llm: LLM | None = None,
        eval_template: str = "",
        score_threshold: float = float("inf"),
    ) -> None:
        self.score_threshold = score_threshold
        self.llm_program = OpenAIPydanticProgram.from_defaults(
            output_cls=None,
            llm=llm or Settings.llm,
            prompt=PromptTemplate(eval_template),
            tool_choice={"type": "function", "function": {"name": "Evaluation"}},
        )

    def evaluate_response(self, response: PydanticResponse) -> EvaluatedResponse:
        eval_response = self.llm_program(response=response.get_response().response)
        score, feedback = self.parser_function(eval_response)
        passing = score >= self.score_threshold
        return EvaluatedResponse(response=response, score=score, feedback=feedback, passing=passing)

    def parser_function(self, eval_response: PydanticResponse) -> Tuple[int, str]:
        score, feedback, i = 0, "", 1
        for _, v in json.loads(eval_response.json()).items():
            if v["score"] != 5:
                feedback += f"{i}. {v['feedback']}, score: {v['score']}/5\n"
                i += 1
            score += v["score"]
        return int(score), feedback
