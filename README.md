# SaltAI_Documentation_Tools

Welcome to the **SaltAI Documentation Tools Pack**, a toolkit designed to automate the generation of documentation for ComfyUI components.

Leveraging the power of Large Language Models (LLM) and Retrieval-Augmented Generation (RAG),
this pack streamlines the process, making it effortless to create, update, and maintain comprehensive documentation for any ComfyUI node or node pack.

The documentation for our platform **Salt** was generated with only the use of the proposed pipeline, please visit
[our site](https://get-salt-ai.github.io/SaltAI-Web-Docs/md/) to check it out.


## Getting Started

### Prerequisites

This project requires the following dependencies:

- **ComfyUI**: Ensure you have ComfyUI installed. The pack has been validated with specific revisions of ComfyUI, which can be found on [GitHub](https://github.com/comfyanonymous/ComfyUI). The validated revisions are:
  - `36f7face37320ec3c6ac85ec3c625738241deaa9`
  - `327ca1313d756c4b443790a53ab0afa1945d3f3e`

- **ComfyUI-Custom-Scripts**: This is required to run an example workflow. You can find it on [GitHub](https://github.com/pythongosssss/ComfyUI-Custom-Scripts).


### Installing

This project has been tested with Python 3.11.0.

#### Environmental Variables

Set the following environmental variables before running the project:

- `OPENAI_KEY`: Your personal OpenAI API key, required for sending requests to the OpenAI API. The price will depend on the model you use.
- `ENABLE_PHOENIX_LOGGING`: Set this to `true` to enable tracing callbacks with [Phoenix](https://docs.arize.com/phoenix/tracing/how-to-tracing/instrumentation/llamaindex) (optional).

#### Installation Steps

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

If you encounter the error `AttributeError: type object 'SpanAttributes' has no attribute 'HTTP_REQUEST_METHOD'` while executing the pack, try installing `opentelemetry-semantic-conventions=0.44.b0`.


## Usage

### Nodes

#### `LoadOpenAIModel`

Load OpenAI language model and embedding model based on provided names, return dictionary with models objects.


#### `DocumentPack` and `DocumentNode`

Locate all nodes loaded in the system (including ComfyUI default nodes and custom node packs) and generate documentation for
the selected pack (node) in json format.

Generation process includes:
- Embed pack code and store it as a vector store index (will be cached in `cache/indices/<parameters_signature>/<pack_name>` directory).
- For each node, retrieve relevant context from the index and pass it to LLM guided by a set of prompts and forced to return structured json output.
- Evaluate the generated documentation against a set of questions. If any problems detected, refine query with the evaluation feedback and repeat the request.
- The resulting documentation will be stored in `cache/documented_nodes/<parameters_signature>/json` directory, split by packs, one json file per node.
- Optionally, you convert json files to markdown format and save those to `cache/documented_nodes/<parameters_signature>/md` directory.
- Node output is a string containing debug information - statistics on errors and mismatches in documentation. Analyse it and regenerate if required.

Example outputs are provided in the folder `examples/json` and `examples/md`.

Additionally, you can setup CI/CD to deploy markdown files to your repo github pages.


#### `LogAllNodesToTable`

Creates csv table containing information about all the nodes available in the system.

Columns include : `Node name`, `Node pack name`, `Github link`, `Commit hash`.


### Example workflow
Please try this simple example [workflow](examples/workflow.json)


## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details
