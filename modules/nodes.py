import json
import sys
import os
import inspect
import csv
import nest_asyncio
import logging
import phoenix as px
from enum import Enum
from typing import Any, Dict, Tuple
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import CodeSplitter
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from .utils import get_all_nodes_packs, json2markdown
from .query_engine import NodeQueryEngine
from . import MAIN_CACHE, NAME
from loguru import logger


# for phoenix logging
import llama_index.core

use_phoenix = os.getenv("ENABLE_PHOENIX_LOGGING", "false").lower() == "true"

# set this env var to true if you want llm traces to be collected and displayed in phoenix
if use_phoenix:
    # llama-index applications will run as usual, llm calls logs will be available at http://127.0.0.1:6006/
    llama_index.core.set_global_handler("arize_phoenix")

nest_asyncio.apply()

# disable annoying logs from openai requests
logging.getLogger("httpx").setLevel(logging.WARNING)


def init_phoenix() -> None:
    """Configures phoenix session if phoenix logging is enabled."""
    if not use_phoenix:
        return
    if type(sys.stdout).__name__ == "ComfyUIManagerLogger":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        config = {
            "handlers": [
                {"sink": sys.stdout, "format": "{time} - {message}"},
                {"sink": sys.stderr, "format": "{time} - {message}"},
            ],
        }
        logger.configure(**config)
    sess = px.active_session()
    # clear before next run, instead of at the end, to allow debugging of outputs
    if sess is not None:
        sess.end()
    px.launch_app()


def log_phoenix() -> None:
    """Logs number of tokens for the current phoenix session."""
    if use_phoenix:
        return
    from phoenix.trace.dsl import SpanQuery

    query = SpanQuery().select(tokens_in="llm.token_count.prompt", tokens_out="llm.token_count.completion")
    # The Phoenix Client can take this query and return the dataframe
    info_df = px.Client().query_spans(query)
    if info_df is not None:
        logger.info(f"Total tokens in: {info_df.tokens_in.sum()}. Total tokens out: {info_df.tokens_out.sum()}")


class RegenerateOptions(Enum):
    no = "no"
    doc = "doc"
    index_doc = "index & doc"
    failed = "failed"


class LoadOpenAIModel:
    """Loads model and embed model. Note that this class is not responsoble for actually setting these models,
    it only creates models objects and passes those to other classes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (sorted(ALL_AVAILABLE_MODELS.keys()), {"default": "gpt-4-turbo-preview"}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "embed_model": (
                    sorted([x.value for x in OpenAIEmbeddingModelType]),
                    {"default": "text-embedding-3-small"},
                ),
            }
        }

    RETURN_TYPES = ("LLM_MODEL",)
    RETURN_NAMES = ("Model",)
    FUNCTION = "load_openai_model"
    CATEGORY = NAME

    def load_openai_model(self, model: str, temperature: int, embed_model: str) -> Dict[str, Any]:
        if "OPENAI_API_KEY" not in os.environ or os.environ["OPENAI_API_KEY"] == "":
            raise EnvironmentError("""The environment variable OPENAI_API_KEY is not set.
Please set it before proceeding (refer to the ENV_VARIABLE_GUIDE.md for details)."""
            )
        llm = OpenAI(model=model, temperature=temperature)
        embed_model = OpenAIEmbedding(model=embed_model)
        return ({"llm": llm, "embed_model": embed_model},)


comfy_nodes_index: VectorStoreIndex | None = None


class DocumentPack:
    """Wraps documentation functionality for any pack of nodes currently available in the system."""

    @classmethod
    def get_all_names(cls):
        cls.all_packs = get_all_nodes_packs()
        return [f"{pack_name}/{len(pack['nodes'])} nodes" for pack_name, pack in cls.all_packs.items()]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": [
                    sorted(cls.get_all_names()),
                ],
                "chunk_lines": ("INT", {"default": 40}),
                "chunk_lines_overlap": ("INT", {"default": 15}),
                "max_chars": ("INT", {"default": 1500}),
                "num_retries": ("INT", {"default": 5}),
                "top_k": ("INT", {"default": 10}),
                "regenerate": ([e.value for e in RegenerateOptions], {"default": RegenerateOptions.no.value}),
                "save_markdown": ("BOOLEAN", {"default": True}),
                "model": ("LLM_MODEL",),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "document"
    CATEGORY = NAME

    def document(self, *args, **kwargs) -> Tuple:
        pack_name = kwargs.pop("name").split("/")[0]
        nodes_list = self.all_packs[pack_name]["nodes"]
        return self._document(pack_name=pack_name, nodes_list=nodes_list, *args, **kwargs)

    def _document(
        self,
        pack_name: str,
        nodes_list: Dict[str, Any],
        chunk_lines: int,
        chunk_lines_overlap: int,
        max_chars: int,
        num_retries: int,
        top_k: int,
        regenerate: str,
        save_markdown: bool,
        model: Dict[str, Any],
    ) -> Tuple[str,]:
        init_phoenix()
        model_name = f"{model['llm'].model}_{model['embed_model'].model_name}_{model['llm'].temperature}"
        emb_name = model_name.split("_")[1] + f"_{chunk_lines}_{chunk_lines_overlap}_{max_chars}"
        query_name = model_name + f"_{chunk_lines}_{chunk_lines_overlap}_{max_chars}_{num_retries}_{top_k}"
        Settings.llm = model["llm"]
        Settings.embed_model = model["embed_model"]
        load_index_kwargs = dict(
            emb_name=emb_name,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
            max_chars=max_chars,
            regenerate=regenerate == "index & doc",
        )
        # adding comfy files because many packs import from there
        global comfy_nodes_index
        if comfy_nodes_index is None:
            comfy_nodes_index = self._load_index(pack_name="Comfy", **load_index_kwargs)
        indexes = [comfy_nodes_index]
        if pack_name != "Comfy":
            indexes.append(self._load_index(pack_name=pack_name, **load_index_kwargs))
        node_query_engine = NodeQueryEngine(indexes=indexes, num_retries=num_retries, top_k=top_k)
        is_max, no_skip, no_error = [], [], []
        for node_name in nodes_list:
            debug_info = self._document_node(
                node_query_engine, query_name, pack_name, node_name, regenerate, save_markdown
            )
            score, max_score = debug_info["_eval_score"].split("/")
            is_max.append(int(score) == int(max_score))
            no_skip.append(int(not debug_info["_has_unknown"]))
            no_error.append(debug_info["_error"] is None)
        Settings.llm = "default"
        Settings.embed_model = "default"
        log_str = f"""Nodes with max score: {sum(is_max)}/{len(is_max)}\n
Nodes without skipped fields: {sum(no_skip)}/{len(no_skip)}\n
Nodes without errors: {sum(no_error)}/{len(no_error)}"""
        log_phoenix()
        return (log_str,)

    def _load_index(
        self,
        emb_name: str,
        pack_name: str,
        chunk_lines: int,
        chunk_lines_overlap: int,
        max_chars: int,
        regenerate: bool,
    ) -> VectorStoreIndex:
        # index will be stored in cache dir, its name is built from passed parameters names
        persist_dir = MAIN_CACHE / "indices" / emb_name / pack_name
        if not regenerate and persist_dir.exists() and any(persist_dir.iterdir()):
            logger.debug(f"Loading index for {pack_name} from {persist_dir}...")
            return load_index_from_storage(StorageContext.from_defaults(persist_dir=persist_dir))
        logger.debug(f"Generating index for {pack_name} to {persist_dir}...")
        persist_dir.mkdir(parents=True, exist_ok=True)
        code_path = self.all_packs[pack_name]["code_path"]
        code_files = [code_path] if str(code_path).endswith(".py") else list(code_path.rglob("*.py"))
        if pack_name == "Comfy":
            code_files = [f for f in code_files if "custom_nodes" not in str(f)]
        documents = SimpleDirectoryReader(input_files=code_files).load_data()
        splitter = CodeSplitter(
            language="python", chunk_lines=chunk_lines, chunk_lines_overlap=chunk_lines_overlap, max_chars=max_chars
        )
        index = VectorStoreIndex(splitter.get_nodes_from_documents(documents), use_async=True)
        index.storage_context.persist(persist_dir=persist_dir)
        return index

    def _document_node(
        self,
        node_query_engine: NodeQueryEngine,
        query_name: str,
        pack_name: str,
        node_name: str,
        regenerate: str,
        save_markdown: bool,
    ) -> Dict[str, str]:
        # doc json will be stored in cache dir, its name is built from passed parameters names
        json_file = MAIN_CACHE / "documented_nodes" / query_name / "json" / pack_name / f"{node_name}.json"
        if save_markdown:
            # can also be converted postfactum for precomputed jsons, set regenerate=no for that
            md_file = MAIN_CACHE / "documented_nodes" / query_name / "md" / pack_name / f"{node_name}.md"
            md_file.parent.mkdir(parents=True, exist_ok=True)
        old_score, old_unknown = None, None
        if regenerate in ["no", "failed"] and json_file.exists():
            with open(json_file, "r") as file:
                old_response = json.load(file)
                old_debug_info = old_response["_debug_info"]
                if regenerate == "no":
                    logger.debug(f"Loading existing response for {node_name} from {json_file}...")
                    if save_markdown and not md_file.exists():
                        json2markdown(old_response, md_file)
                    return old_debug_info
                if regenerate == "failed":
                    if old_debug_info["_error"] is None:
                        score, max_score = old_debug_info["_eval_score"].split("/")
                        old_score, old_unknown = int(score) / int(max_score), old_debug_info["_has_unknown"]
                        if not old_unknown and old_score == 1:
                            logger.debug(f"Loading existing response for {node_name} from {json_file}...")
                            if save_markdown and not md_file.exists():
                                json2markdown(old_response, md_file)
                            return old_debug_info
        logger.debug(f"Generating response for {node_name} to {json_file}...")
        node_info = self.all_packs[pack_name]["nodes"][node_name]
        response = node_query_engine.query(node_name, node_info)
        debug_info = response["_debug_info"]
        if old_score is not None:
            # don't update the result if we made it even worse
            score, max_score = debug_info["_eval_score"].split("/")
            new_score, new_unknown = int(score) / int(max_score), debug_info["_has_unknown"]
            if old_unknown:
                if new_unknown and new_score < old_score:
                    logger.debug(f"Unable to improve previous response for '{node_name}' from '{json_file}'.")
                    if save_markdown and not md_file.exists():
                        json2markdown(old_response, md_file)
                    return old_debug_info
            else:
                if new_unknown or not new_unknown and new_score < old_score:
                    logger.debug(f"Unable to improve previous response for '{node_name}' from '{json_file}'.")
                    if save_markdown and not md_file.exists():
                        json2markdown(old_response, md_file)
                    return old_debug_info
        json_file.parent.mkdir(parents=True, exist_ok=True)
        response["_repo_info"] = self.all_packs[pack_name]["repo_info"]
        with open(json_file, "w") as file:
            json.dump(response, file, indent=4)
        logger.debug(f"Saving generated response for '{node_name}' to '{json_file}'.")
        if save_markdown:
            json2markdown(response, md_file)
        return debug_info


class DocumentNode(DocumentPack):
    """Wraps documentation functionality for one specific node."""

    @classmethod
    def get_all_names(cls):
        cls.all_packs = get_all_nodes_packs()
        return [
            f"{pack_name}/{node['display_name']}"
            for pack_name, pack in cls.all_packs.items()
            for node in pack["nodes"].values()
        ]

    def document(self, *args, **kwargs) -> Tuple:
        name = kwargs.pop("name")
        pack_name = name.split("/")[0]
        node_display_name = "/".join(name.split("/")[1:])
        for node_name, node_info in self.all_packs[pack_name]["nodes"].items():
            if node_info["display_name"] == node_display_name:
                break
        nodes_list = [node_name]
        return self._document(pack_name=pack_name, nodes_list=nodes_list, *args, **kwargs)


class LogAllNodesToTable:
    """Logs all nodes loaded in the system to the csv table."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_name": ("STRING", {"default": str(MAIN_CACHE / "info_table.csv")}),
                "include_current": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "log_to_table"

    def log_to_table(self, file_name: str, include_current: bool) -> None:
        Path(file_name).parent.mkdir(exist_ok=True, parents=True)
        all_packs = get_all_nodes_packs()
        node_dirs = Path(inspect.getfile(self.__class__)).parts
        current_pack_name = node_dirs[node_dirs.index("custom_nodes") + 1]
        with open(file_name, "w") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["Node name", "Node pack name", "Github link", "Commit hash"])
            writer.writeheader()
            for pack_name, pack_info in all_packs.items():
                if not include_current and pack_name == current_pack_name:
                    continue
                repo_url = pack_info["repo_info"]["_repo_url"]
                commit_hash = pack_info["repo_info"]["_commit_hash"]
                if repo_url == "":
                    repo_url = "unknown"
                if commit_hash == "":
                    commit_hash = "unknown"
                for node_name in pack_info["nodes"]:
                    writer.writerow(
                        {
                            "Node name": node_name,
                            "Node pack name": pack_name,
                            "Github link": repo_url,
                            "Commit hash": commit_hash,
                        }
                    )
        return ()


NODE_CLASS_MAPPINGS = {
    "LoadOpenAIModel": LoadOpenAIModel,
    "DocumentNode": DocumentNode,
    "DocumentPack": DocumentPack,
    "LogAllNodesToTable": LogAllNodesToTable,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadOpenAIModel": "Load OpenAI Model",
    "DocumentNode": "Document Node",
    "DocumentPack": "Document Pack",
    "LogAllNodesToTable": "Log All Nodes To Table",
}
