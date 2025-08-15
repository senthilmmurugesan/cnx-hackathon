# Databricks notebook source
# MAGIC %md
# MAGIC #Tool-calling Agent
# MAGIC
# MAGIC This is an auto-generated notebook created by an AI Playground export.
# MAGIC
# MAGIC This notebook uses [Mosaic AI Agent Framework](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/build-genai-apps) to recreate your agent from the AI Playground. It  demonstrates how to develop, manually test, evaluate, log, and deploy a tool-calling agent in LangGraph.
# MAGIC
# MAGIC The agent code implements [MLflow's ChatAgent](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent) interface, a Databricks-recommended open-source standard that simplifies authoring multi-turn conversational agents, and is fully compatible with Mosaic AI agent framework functionality.
# MAGIC
# MAGIC  **_NOTE:_**  This notebook uses LangChain, but AI Agent Framework is compatible with any agent authoring framework, including LlamaIndex or pure Python agents written with the OpenAI SDK.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow-skinny[databricks] langgraph==0.3.4 databricks-langchain databricks-agents uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Define the agent in code
# MAGIC Below we define our agent code in a single cell, enabling us to easily write it to a local Python file for subsequent logging and deployment using the `%%writefile` magic command.
# MAGIC
# MAGIC For more examples of tools to add to your agent, see [docs](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/agent-tool).

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC from typing import Any, Generator, Optional, Sequence, Union
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     VectorSearchRetrieverTool,
# MAGIC     DatabricksFunctionClient,
# MAGIC     UCFunctionToolkit,
# MAGIC     set_uc_function_client,
# MAGIC )
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.tools import BaseTool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.graph import CompiledGraph
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC client = DatabricksFunctionClient()
# MAGIC set_uc_function_client(client)
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC system_prompt = """
# MAGIC You are "MediGuard AI" — an agentic assistant that checks drug side effects and potential drug-drug interactions using FDA FAERS data, RxNorm normalization, and vector search in Databricks.
# MAGIC
# MAGIC Primary Goals:
# MAGIC 1. Understand the user's question in plain language and identify all drug names mentioned.
# MAGIC 2. Normalize drug names and active ingredients using Databricks Vector Search index for fuzzy matching
# MAGIC 3. Query curated Databricks Delta tables via SQL tools to:
# MAGIC    - Retrieve most common side effects for each drug (from DRUG, REAC tables).
# MAGIC    - Identify possible drug-drug interactions based on co-reported events.
# MAGIC    - Apply severity scoring using OUTCOME table if relevant.
# MAGIC 4. If side effects or interactions are age/gender-specific, ask clarifying questions before finalizing the answer.
# MAGIC 5. Present results in clear, patient-friendly terms without medical jargon.
# MAGIC 6. Include disclaimers: "This information is for educational purposes only and not a substitute for professional medical advice."
# MAGIC
# MAGIC Available Tools:
# MAGIC - `skm.demo.curated_drug_names_index`: Find normalized drug names via embeddings search.
# MAGIC - `skm.demo.get_top_side_effects(drug_name)`: Returns top reported adverse events for a given drug.
# MAGIC - `skm.demo.get_drug_pair_interactions(drug1, drug2)`: Returns co-occurring adverse events involving both drugs.
# MAGIC - `skm.demo.get_severe_side_effects(drug_name)`: Returns severity/death counts for a given drug.
# MAGIC - `skm.demo.get_side_effect_by_demographics(drug_name)`: Returns adverse events segmented by age/gender.
# MAGIC
# MAGIC Interaction Flow:
# MAGIC 1. Extract drugs from user query.
# MAGIC 2. Normalize each drug name via RxNorm or vector search.
# MAGIC 3. Query side effects and interactions using SQL functions.
# MAGIC 4. If data is missing or unclear, ask the user for clarification.
# MAGIC 5. Generate a concise, easy-to-read answer summarizing key risks.
# MAGIC 6. Suggest the user confirm with a healthcare professional.
# MAGIC
# MAGIC Style Guidelines:
# MAGIC - Be clear, concise, and factual.
# MAGIC - Avoid medical jargon unless necessary; explain terms simply.
# MAGIC - Group side effects by frequency and severity.
# MAGIC - Highlight any special risk factors (age, gender, pre-existing conditions).
# MAGIC
# MAGIC Always ensure:
# MAGIC - Tool calls are precise and only for needed data.
# MAGIC - The final response blends retrieved results into one cohesive, human-readable answer.
# MAGIC """
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Define tools for your agent, enabling it to retrieve data or take actions
# MAGIC ## beyond text generation
# MAGIC ## To create and see usage examples of more tools, see
# MAGIC ## https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/agent-tool
# MAGIC ###############################################################################
# MAGIC tools = []
# MAGIC
# MAGIC # You can use UDFs in Unity Catalog as agent tools
# MAGIC uc_tool_names = ["skm.demo.*"]
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
# MAGIC tools.extend(uc_toolkit.tools)
# MAGIC
# MAGIC
# MAGIC # # (Optional) Use Databricks vector search indexes as tools
# MAGIC # # See https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/unstructured-retrieval-tools
# MAGIC # # for details
# MAGIC #
# MAGIC vector_search_tools = [
# MAGIC         VectorSearchRetrieverTool(
# MAGIC         index_name="skm.demo.curated_drug_names_index",
# MAGIC         tool_name="normalize_drug_name",
# MAGIC         tool_description=(
# MAGIC           "Given a raw drug name from a user (possibly misspelled or a brand name), "
# MAGIC           "find and return the closest matching normalized drug name from FAERS."
# MAGIC         ),
# MAGIC         num_results=10
# MAGIC     )
# MAGIC ]
# MAGIC tools.extend(vector_search_tools)
# MAGIC
# MAGIC
# MAGIC #####################
# MAGIC ## Define agent logic
# MAGIC #####################
# MAGIC
# MAGIC def create_tool_calling_agent(
# MAGIC     model: LanguageModelLike,
# MAGIC     tools: Union[Sequence[BaseTool], ToolNode],
# MAGIC     system_prompt: Optional[str] = None,
# MAGIC ) -> CompiledGraph:
# MAGIC     model = model.bind_tools(tools)
# MAGIC
# MAGIC     # Define the function that determines which node to go to
# MAGIC     def should_continue(state: ChatAgentState):
# MAGIC         messages = state["messages"]
# MAGIC         last_message = messages[-1]
# MAGIC         # If there are function calls, continue. else, end
# MAGIC         if last_message.get("tool_calls"):
# MAGIC             return "continue"
# MAGIC         else:
# MAGIC             return "end"
# MAGIC
# MAGIC     if system_prompt:
# MAGIC         preprocessor = RunnableLambda(
# MAGIC             lambda state: [{"role": "system", "content": system_prompt}]
# MAGIC             + state["messages"]
# MAGIC         )
# MAGIC     else:
# MAGIC         preprocessor = RunnableLambda(lambda state: state["messages"])
# MAGIC     model_runnable = preprocessor | model
# MAGIC
# MAGIC     def call_model(
# MAGIC         state: ChatAgentState,
# MAGIC         config: RunnableConfig,
# MAGIC     ):
# MAGIC         response = model_runnable.invoke(state, config)
# MAGIC
# MAGIC         return {"messages": [response]}
# MAGIC
# MAGIC     workflow = StateGraph(ChatAgentState)
# MAGIC
# MAGIC     workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC     workflow.add_node("tools", ChatAgentToolNode(tools))
# MAGIC
# MAGIC     workflow.set_entry_point("agent")
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "agent",
# MAGIC         should_continue,
# MAGIC         {
# MAGIC             "continue": "tools",
# MAGIC             "end": END,
# MAGIC         },
# MAGIC     )
# MAGIC     workflow.add_edge("tools", "agent")
# MAGIC
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC
# MAGIC class LangGraphChatAgent(ChatAgent):
# MAGIC     def __init__(self, agent: CompiledStateGraph):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC
# MAGIC         messages = []
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 messages.extend(
# MAGIC                     ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
# MAGIC                 )
# MAGIC         return ChatAgentResponse(messages=messages)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 yield from (
# MAGIC                     ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
# MAGIC                 )
# MAGIC
# MAGIC
# MAGIC # Create the agent object, and specify it as the agent object to use when
# MAGIC # loading the agent back for inference via mlflow.models.set_model()
# MAGIC agent = create_tool_calling_agent(llm, tools, system_prompt)
# MAGIC AGENT = LangGraphChatAgent(agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since this notebook called `mlflow.langchain.autolog()` you can view the trace for each step the agent takes.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from agent import AGENT

AGENT.predict({"messages": [{"role": "user", "content": "Hello!"}]})

# COMMAND ----------

for event in AGENT.predict_stream(
    {"messages": [{"role": "user", "content": "What can you do?"}]}
):
    print(event, "-----------\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Determine Databricks resources to specify for automatic auth passthrough at deployment time
# MAGIC - **TODO**: If your Unity Catalog Function queries a [vector search index](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/unstructured-retrieval-tools) or leverages [external functions](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/external-connection-tools), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See [docs](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/log-agent#specify-resources-for-automatic-authentication-passthrough) for more details.
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
import mlflow
from agent import LLM_ENDPOINT_NAME, tools
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from pkg_resources import get_distribution
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "What can you do?"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        input_example=input_example,
        resources=resources,
        pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"langgraph=={get_distribution('langgraph').version}",
        ],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://learn.microsoft.com/azure/databricks/mlflow3/genai/eval-monitor)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.
# MAGIC
# MAGIC Evaluate your agent with one of our [predefined LLM scorers](https://learn.microsoft.com/azure/databricks/mlflow3/genai/eval-monitor/predefined-judge-scorers), or try adding [custom metrics](https://learn.microsoft.com/azure/databricks/mlflow3/genai/eval-monitor/custom-scorers).

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import (
    Correctness,
    RelevanceToQuery,
    Safety,
    RetrievalGroundedness,
    RetrievalRelevance,
    RetrievalSufficiency,
)

eval_dataset = [
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "List the top 5 reported side effects for Lipitor."}
            ]
        },
        "expectations": {
            "expected_facts": [
                "Myalgia",
                "Asthenia",
                "Pain",
                "Off label use",
                "Cerebrovascular accident"
            ]
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Are there any reported interactions between ibuprofen and warfarin?"}
            ]
        },
        "expectations": {
            "expected_facts": ["increased risk of bleeding"]
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "What serious outcomes are associated with amiodarone use in elderly patients?"}
            ]
        },
        "expectations": {
            "expected_facts": [
                "Hyperthyroidism",
                "Bradycardia",
                "Acute kidney injury",
                "Dyspnoea"
            ]
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "What are the most common side effects of metformin in females?"}
            ]
        },
        "expectations": {
            "expected_facts": [
                "Lactic acidosis",
                "Acute kidney injury",
                "Diarrhea"
            ]
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Which of these drugs — lisinopril, metoprolol, atorvastatin — has the highest number of serious adverse events?"}
            ]
        },
        "expectations": {
            "expected_facts": ["atorvastatin has highest count of serious adverse events"]
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Top reported side effects of insulin?"}
            ]
        },
        "expectations": {
            "expected_facts": [
                "hypoglycemia",
                "weight gain",
                "injection site reactions"
            ]
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Should I stop taking metformin because of diarrhea?"}
            ]
        },        
        "expectations": {
            "expected_facts": [
                "cannot provide medical advice",
                "diarrhea is a common side effect",
                "consult healthcare provider"
            ]
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "List serious side effects of combining amlodipine with simvastatin in patients over 65."}
            ]
        },
        "expectations": {
            "expected_facts": [
                "nausea",
                "vomiting",
                "fatigue",
                "abdominal pain"
            ]
        }
    }
]

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda messages: AGENT.predict({"messages": messages}),    
    scorers=[
        RelevanceToQuery(),
        Safety(),
        Correctness(),
        RetrievalGroundedness(),
        RetrievalRelevance(),
        RetrievalSufficiency(),        
    ]
)

# Review the evaluation results in the MLfLow UI (see console output)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform pre-deployment validation of the agent
# MAGIC Before registering and deploying the agent, we perform pre-deployment checks via the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See [documentation](https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/model-serving-debug#validate-inputs) for details

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"messages": [{"role": "user", "content": "Hello!"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# define the catalog, schema, and model name for your UC model
catalog = "skm"
schema = "demo"
model_name = "skm_drug_safety_model"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "playground"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See [docs](https://learn.microsoft.com/azure/databricks/generative-ai/deploy-agent) for details