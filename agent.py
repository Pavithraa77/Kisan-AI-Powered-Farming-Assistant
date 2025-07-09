from google.adk.agents import Agent
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image
import os
import base64
import io
from PIL import Image as PILImage
# --- LangChain Agent Integration (Conceptual) ---
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import tool
from langchain_core.tools import Tool
from langchain_google_vertexai import GenerativeModel
from langchain_google_vertexai import VertexRagStore, RagResource, Retrieval
from langchain_google_vertexai import VertexPredictionEndpoint
from langchain_google_vertexai import RagEmbeddingModelConfig, RagVectorDbConfig, rag          
# Load environment variables
from dotenv import load_dotenv
load_dotenv()               

# Initialize Google Cloud project and region from environment variables
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "agentic-hack")
REGION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")                              
# Initialize Vertex AI
vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=REGION)

def create_and_import_rag_corpus(project_id, region, corpus_display_name, gcs_paths):
    """
    Creates a RAG Corpus and imports files from GCS.
    This should be run once to set up your knowledge base.
    """
    print(f"Creating RAG Corpus: {corpus_display_name} in {region}...")
    try:
        # Check if corpus already exists by trying to list it (more robust than just creating)
        # This part is a bit tricky with the current SDK, as there's no direct `get_corpus_by_display_name`
        # A workaround would be to list all corpora and check names, or rely on a known corpus ID.
        # For simplicity in this example, we'll assume either it's new or the user provides the existing ID.
        
        # Configure embedding model
        embedding_model_config = rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model="publishers/google/models/text-embedding-005"
            )
        )

        rag_corpus = rag.create_corpus(
            display_name=corpus_display_name,
            backend_config=rag.RagVectorDbConfig(
                rag_embedding_model_config=embedding_model_config
            ),
        )
        print(f"Created RAG Corpus resource: {rag_corpus.name}")

        print(f"Importing files from {gcs_paths} to RAG Corpus...")
        rag.import_files(
            rag_corpus.name,
            gcs_paths,
            transformation_config=rag.TransformationConfig(
                chunking_config=rag.ChunkingConfig(
                    chunk_size=1024,
                    chunk_overlap=100,
                ),
            ),
            max_embedding_requests_per_min=1000,
        )
        print("Files imported successfully into RAG Corpus.")
        return rag_corpus.name
    except Exception as e:
        print(f"Error during RAG Corpus creation or import: {e}")
        print("If the corpus already exists, please update PRE_EXISTING_CORPUS_NAME with its resource ID.")
        raise

# Example usage (run this part once):
# You'll likely run this in a separate script or section to set up your corpus.
# Once created, copy the output `corpus_name` and set it as an environment variable
# or directly in your deployment configuration.
# For local testing, you might hardcode it AFTER creation.

# IMPORTANT: Replace this with the actual corpus name you obtained after running the setup script.
# This value should ideally also come from an environment variable in a deployed agent.
# For local development, you might temporarily hardcode it after initial creation.
# Example format: projects/PROJECT_ID/locations/REGION/ragCorpora/CORPUS_ID
PRE_EXISTING_CORPUS_NAME = os.getenv("RAG_CORPUS_NAME", "projects/agentic-hack/locations/us-central1/ragCorpora/YOUR_CORPUS_ID")
# Make sure to replace YOUR_CORPUS_ID with your actual corpus ID!

# --- RAG Tool for Agent ---

def get_agricultural_scheme_info(query: str) -> str:
    """
    Retrieves and explains government agricultural schemes related to the query
    using RAG.
    Args:
        query (str): The farmer's specific need, e.g., "subsidies for drip irrigation".
    Returns:
        str: A comprehensive explanation of relevant schemes, eligibility, and
             application links in simple terms, or an apology if information is not found.
    """
    try:
        # Check if PRE_EXISTING_CORPUS_NAME is set
        if not PRE_EXISTING_CORPUS_NAME or "YOUR_CORPUS_ID" in PRE_EXISTING_CORPUS_NAME:
            return "RAG Corpus is not configured. Please ensure 'RAG_CORPUS_NAME' environment variable is set with your Vertex AI RAG Corpus ID."

        rag_resource = RagResource(rag_corpus=PRE_EXISTING_CORPUS_NAME)
        rag_retrieval_tool = Tool.from_retrieval(
            retrieval=Retrieval(
                source=VertexRagStore(
                    rag_resources=[rag_resource],
                    similarity_top_k=5,
                ),
            )
        )

        rag_model = GenerativeModel(
            model_name="gemini-1.5-flash-001",
            tools=[rag_retrieval_tool],
            system_instruction="You are a helpful AI assistant specialized in Indian government agricultural schemes. Your task is to provide clear, simple, and accurate information to farmers based on the provided context. Include scheme names, eligibility criteria, and direct application links if available. If you cannot find relevant information, politely state that you don't have the information."
        )

        prompt = (
            f"A farmer is asking about: '{query}'.\n"
            "Based on the retrieved government agricultural scheme documents, "
            "please explain the relevant schemes in simple terms, "
            "list the eligibility requirements, and provide direct links "
            "to application portals if found. "
            "If a specific application link is not found in the documents, "
            "advise the farmer to visit the official website of the Ministry of Agriculture "
            "or their state's agriculture department portal. "
            "Prioritize schemes related to micro-irrigation like drip irrigation."
        )

        response = rag_model.generate_content(prompt)

        if response.candidates:
            return response.candidates[0].text
        else:
            return "I apologize, but I couldn't find specific information for your request at this moment. Please try rephrasing your query or visit the official website of the Ministry of Agriculture or your state's agriculture department portal."

    except Exception as e:
        print(f"Error in RAG tool: {e}")
        return "I encountered an error while processing your request. Please try again later."




def agricultural_scheme_info_tool(query: str) -> str:
    """
    Provides information on government agricultural schemes, eligibility, and application links.
    Use this tool when a farmer asks about specific needs related to agricultural subsidies,
    such as "subsidies for drip irrigation", "schemes for farm equipment", "crop insurance details".
    The input should be the specific need or question from the farmer.
    """
    return get_agricultural_scheme_info(query)

tools = [agricultural_scheme_info_tool]

# Initialize the LLM for the agent's reasoning using loaded environment variables
llm = ChatVertexAI(model_name="gemini-2.5-flash-lite-preview-06-17", project=GOOGLE_CLOUD_PROJECT, location=REGION)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful agricultural assistant for farmers. Your main goal is to understand their needs and provide relevant government scheme information using the available tools."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
def diagnose_plant_disease(image_data_base64: str) -> dict:
    try:
        model = GenerativeModel("gemini-2.5-flash-lite-preview-06-17")

        image_bytes = base64.b64decode(image_data_base64)
        input_image = Part.from_bytes(image_bytes, mime_type="image/jpeg")

        prompt = """
        Analyze this image of a plant.
        1. Identify if there are any signs of pest infestation or plant disease.
        2. If a pest or disease is identified, state its name clearly.
        3. Provide clear, actionable advice on how to treat or manage the issue. Focus on remedies that are
           likely to be locally available and affordable for a farmer in India (e.g., common organic solutions,
           widely available pesticides if necessary, cultural practices).
        4. If the plant appears healthy or no clear issue can be determined, state that.

        Format your response as follows:
        Diagnosis: [Disease/Pest Name or "Healthy" or "Undetermined"]
        Confidence: [High/Medium/Low or a percentage if the model provides it]
        Symptoms Observed: [List key symptoms visible in the image]
        Recommended Action: [Clear, step-by-step advice for treatment/prevention]
        Considerations for Farmers: [Practical tips, e.g., "Check local agricultural extension office for specific recommendations.", "Ensure good drainage."]
        """

        response = model.generate_content([input_image, prompt])

        if response.candidates:
            gemini_output = response.candidates[0].content.text
            return {
                "status": "success",
                "report": gemini_output
            }
        else:
            return {
                "status": "error",
                "error_message": "Gemini did not return a valid response for the image."
            }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"An error occurred during diagnosis: {e}"
        }


root_agent = Agent(
    name="crop_disease_diagnoser",
    model="gemini-2.5-flash-lite-preview-06-17",
    description="An AI assistant that helps farmers diagnose crop diseases and pests from photos and provides actionable remedies.",
    instruction="""
    I am an expert in plant pathology and pest management.
    A farmer can send me a photo of a diseased plant, and I will analyze it to identify the problem
    and provide practical, actionable advice on how to address it, keeping in mind local and affordable solutions
    for farmers in India.

    Please provide a clear photo of the affected plant part.
    """,
    tools=[diagnose_plant_disease]
)
