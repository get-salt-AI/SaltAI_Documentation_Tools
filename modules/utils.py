import inspect
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from llama_index.core import Response
from git import Repo, InvalidGitRepositoryError, NoSuchPathError


def get_repo_info(code_path: Path) -> Dict[str, str]:
    """Detects repo url and commit ref for given path if it's possible."""
    try:
        repo = Repo(code_path)
        return {"_repo_url": next(repo.remote().urls), "_commit_hash": str(repo.head.commit)}
    except (InvalidGitRepositoryError, NoSuchPathError):
        return {"_repo_url": "", "_commit_hash": ""}


def get_all_nodes_packs() -> Dict[str, Any]:
    """Collect all imported packs and stores all imported nodes classes for each pack."""
    from nodes import NODE_CLASS_MAPPINGS as all_loaded_nodes
    from nodes import NODE_DISPLAY_NAME_MAPPINGS as all_loaded_nodes_names

    all_packs = {}
    for node_name, node_class in all_loaded_nodes.items():
        node_dirs = Path(inspect.getfile(node_class)).parts
        if "custom_nodes" in node_dirs:
            pack_idx = node_dirs.index("custom_nodes") + 1
            pack_name = node_dirs[pack_idx]
        else:
            if "PlaiGenerationAPI" in node_dirs:
                pack_idx = node_dirs.index("PlaiGenerationAPI") + 1
            else:
                pack_idx = node_dirs.index("ComfyUI")
            pack_name = "Comfy"
        pack_name = pack_name.replace(".py", "").replace("/", "_")
        all_packs.setdefault(pack_name, {"nodes": {}})
        display_name = all_loaded_nodes_names[node_name] if node_name in all_loaded_nodes_names else node_name
        node_name = node_name.replace("/", "_")
        all_packs[pack_name]["nodes"][node_name] = {"class": node_class, "display_name": display_name}
        if "code_path" not in all_packs[pack_name]:
            code_path = Path().joinpath(*node_dirs[: pack_idx + 1])
            all_packs[pack_name]["code_path"] = code_path
            all_packs[pack_name]["repo_info"] = get_repo_info(code_path)
    return all_packs


def collapse_repeating_parameters(params_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Collapses repeating parameters like `input_blocks.0`,...`input_blocks.10` into 1 parameter `input_blocks.i`."""
    collapsed = {}
    pattern_seen = {}
    for param_category in params_dict:
        collapsed[param_category] = {}
        for param_name, param_type in params_dict[param_category].items():
            pattern = r"\.\d+"
            generic_pattern, n = re.subn(pattern, ".{}", param_name)
            if n > 0:
                letters = (letter for letter in "ijklmnopqrstuvwxyzabcdefgh")
                generic_pattern = re.sub(r"\{\}", lambda _: next(letters), generic_pattern)
                if generic_pattern not in pattern_seen:
                    pattern_seen[generic_pattern] = True
                    collapsed[param_category][generic_pattern] = param_type
            else:
                collapsed[param_category][param_name] = param_type
    return collapsed


def match_combo(lst: List[Any] | Tuple[Any]):
    """Detects comfy dtype for a combo parameter."""
    types_matcher = {"str": "STRING", "float": "FLOAT", "int": "INT", "bool": "BOOLEAN"}
    if len(lst) > 0:
        return f"COMBO[{types_matcher.get(type(lst[0]).__name__, 'STRING')}]"
    else:
        return "COMBO[STRING]"


def update_node_info(node_info: Dict[str, Any]) -> Dict[str, Any]:
    """Collects available information from node class to use in the pipeline."""
    node_class = node_info["class"]
    input_parameters, output_parameters = {}, {}
    for k, v in node_class.INPUT_TYPES().items():
        if k in ["required", "optional"]:
            input_parameters[k] = {}
            for k0, v0 in v.items():
                if isinstance(v0, list):
                    # these are actually badly written nodes that won't work
                    input_parameters[k][k0] = match_combo(v0)
                elif isinstance(v0[0], list):
                    # this is a correct notation
                    input_parameters[k][k0] = match_combo(v0[0])
                else:
                    input_parameters[k][k0] = v0[0]
    return_types = [
        match_combo(x) if isinstance(x, list) or isinstance(x, tuple) else x for x in node_class.RETURN_TYPES
    ]
    return_names = getattr(node_class, "RETURN_NAMES", [t.lower() for t in return_types])
    for t, n in zip(return_types, return_names):
        output_parameters[n] = t
    return {
        "input_parameters": collapse_repeating_parameters(input_parameters),
        "output_parameters": output_parameters,
        "source_code": inspect.getsource(node_class),
        "display_name": node_info["display_name"],
        "output_node": str(getattr(node_class, "OUTPUT_NODE", False)),
        "category": str(getattr(node_class, "CATEGORY", None)),
    }


def format_response(node_name: str, response: Response, node_info: Dict[str, Any], eval_score: str) -> Dict[str, Any]:
    """Final postprocessing of the response before saving json file."""
    has_unknown = False
    response = json.loads(response.response.json())
    response["source_code"] = node_info["source_code"]
    response["display_name"] = node_info["display_name"]
    response["output_node"] = node_info["output_node"]
    response["category"] = node_info["category"]
    response["class"] = node_name
    if response["documentation"] is not None:
        response["documentation"] = response["documentation"].replace("\n", "")
    else:
        response["documentation"] = "unknown"
        has_unknown = True
    input_types, output_types = response.pop("input_types"), response.pop("output_types")
    input_types_actual, output_types_actual = node_info["input_parameters"], node_info["output_parameters"]
    input_types_formatted, output_types_formatted = {}, {}
    input_types = input_types or {}
    output_types = output_types or {}
    input_types = {el["name"]: {k: v for k, v in el.items() if k != "name"} for el in input_types}
    output_types = {el["name"]: {k: v for k, v in el.items() if k != "name"} for el in output_types}
    # if predicted type actually exists in the node class save llm generated documentation
    # otherwise take only type name and comfy_dtype from the source class
    for param_category in ["required", "optional"]:
        if param_category not in input_types_actual:
            continue
        input_types_formatted[param_category] = {}
        for param_name, param_type in input_types_actual[param_category].items():
            input_types_formatted[param_category][param_name] = {"comfy_dtype": param_type}
            if param_name in input_types:
                for entry_name in ["documentation", "python_dtype"]:
                    if input_types[param_name][entry_name] is None:
                        has_unknown = True
                        input_types[param_name][entry_name] = "unknown"
                    entry_value = input_types[param_name][entry_name].replace("\n", "")
                    input_types_formatted[param_category][param_name][entry_name] = entry_value
            else:
                input_types_formatted[param_category][param_name].update(
                    {"documentation": "unknown", "python_dtype": "unknown"}
                )
                has_unknown = True
    for param_name, param_type in output_types_actual.items():
        output_types_formatted[param_name] = {"comfy_dtype": param_type}
        if param_name in output_types:
            for entry_name in ["documentation", "python_dtype"]:
                if output_types[param_name][entry_name] is None:
                    has_unknown = True
                    output_types[param_name][entry_name] = "unknown"
                output_types_formatted[param_name][entry_name] = output_types[param_name][entry_name].replace("\n", "")
        else:
            output_types_formatted[param_name].update({"documentation": "unknown", "python_dtype": "unknown"})
            has_unknown = True
    if "ui" in output_types:
        # won't be reflected in actual output_types since it's not in fact a parameter
        output_types_formatted["ui"] = {"documentation": output_types["ui"]["documentation"].replace("\n", "")}
    response["input_types"], response["output_types"] = input_types_formatted, output_types_formatted
    response["_debug_info"] = {"_eval_score": eval_score, "_has_unknown": has_unknown, "_error": None}
    return response


def node_to_id(node: Dict[str, Any], pipeline: Dict[str, Any]) -> int:
    for pipe_node in pipeline["nodes"]:
        if pipe_node["type"] == node["class"] or pipe_node["type"] == node["display_name"]:
            return pipe_node["id"]


def id_to_node(id: int, pipeline: Dict[str, Any]) -> str:
    for pipe_node in pipeline["nodes"]:
        if pipe_node["id"] == id:
            return pipe_node["type"]


def json2markdown(json_dict, md_file):
    """Example of json to markdown converter. You are welcome to change formatting per specfic request."""
    if json_dict["_debug_info"]["_error"] is not None:
        return ""
    indent = 0
    markdown_str = f"# {json_dict['display_name']}\n"
    class_name = f"- Class name: `{json_dict['class']}`\n"
    category = f"- Category: `{json_dict['category']}`\n"
    output_node = f"- Output node: `{json_dict['output_node']}`\n"
    markdown_str += f"## Documentation\n{class_name}{category}{output_node}\n{json_dict['documentation']}\n"
    markdown_str += "## Input types\n"
    for type_category, type_dict in json_dict["input_types"].items():
        markdown_str += f"### {type_category.capitalize()}\n"
        for type_name, type_info in type_dict.items():
            markdown_str += "    " * indent + f"- **`{type_name}`**\n"
            indent += 1
            markdown_str += "    " * indent + f"- {type_info.pop('documentation')}\n"
            for entry_name, entry_info in type_info.items():
                entry_name = entry_name.capitalize().replace("_", " ")
                markdown_str += "    " * indent + f"- {entry_name}: `{entry_info}`\n"
            indent -= 1
    markdown_str += "## Output types\n"
    if len(json_dict["output_types"]) == 0:
        markdown_str += "The node doesn't have output types\n"
    for type_name, type_info in json_dict["output_types"].items():
        markdown_str += "    " * indent + f"- **`{type_name}`**\n"
        indent += 1
        for entry_name, entry_info in type_info.items():
            if entry_name == "documentation":
                markdown_str += "    " * indent + f"- {entry_info}\n"
            else:
                entry_name = entry_name.capitalize().replace("_", " ")
                markdown_str += "    " * indent + f"- {entry_name}: `{entry_info}`\n"
        indent -= 1
    infra_type = f"- Infra type: `{json_dict['infra_type']}`\n"
    if json_dict["common_nodes"] == "":
        common_nodes = " unknown"
    else:
        common_nodes = "\n"
        for node in json_dict["common_nodes"].split(","):
            common_nodes += f"    - {node}\n"
    common_nodes = f"- Common nodes:{common_nodes}\n"
    markdown_str += f"## Usage tips\n{infra_type}{common_nodes}\n{json_dict['usage_doc']}\n"
    markdown_str += f"## Source code\n```python\n{json_dict['source_code']}\n```\n"
    with open(md_file, "w") as file:
        file.write(markdown_str)
