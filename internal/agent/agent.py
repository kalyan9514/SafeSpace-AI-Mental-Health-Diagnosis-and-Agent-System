"""
internal/agent/agent.py

LangChain-based conversational agent for the Streamlit interface.
Handles mental health Q&A, helpline suggestions, and nearby clinic search.
"""

import logging
import googlemaps
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
# from langchain.schema import SystemMessage
from config.config import settings

logger = logging.getLogger(__name__)


def _build_vectorstore():
    """Load the knowledge base and build a FAISS vectorstore using OpenAI embeddings."""
    loader = TextLoader(settings.knowledge_base_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embedding = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
    vectorstore = FAISS.from_documents(docs, embedding)
    return vectorstore.as_retriever()


def find_clinics(city: str, radius: int = 10000) -> list[dict]:
    """
    Search for nearby mental health clinics using the Google Maps API.
    Returns a list of clinic dicts with name, address, phone, and coordinates.
    """
    gmaps = googlemaps.Client(key=settings.google_maps_api_key)
    location = gmaps.geocode(city)[0]["geometry"]["location"]
    lat, lng = location["lat"], location["lng"]

    places_result = gmaps.places_nearby(
        location=(lat, lng),
        radius=radius,
        keyword="mental health center",
        type="health",
    )

    clinics = []
    for place in places_result.get("results", [])[:5]:
        details = gmaps.place(place_id=place["place_id"]).get("result", {})
        clinics.append({
            "name": details.get("name", "N/A"),
            "address": details.get("formatted_address", "N/A"),
            "phone": details.get("formatted_phone_number", "N/A"),
            "lat": details.get("geometry", {}).get("location", {}).get("lat"),
            "lng": details.get("geometry", {}).get("location", {}).get("lng"),
        })

    logger.info(f"Found {len(clinics)} clinics near {city}.")
    return clinics


def build_agent():
    """
    Build and return the LangChain agent with all tools attached.
    Call this once at Streamlit app startup.
    """
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=settings.openai_api_key,
        temperature=0,
    )

    retriever = _build_vectorstore()
    summary_chain = load_summarize_chain(llm, chain_type="map_reduce")

    def suggest_helpline(query: str) -> str:
        return (
            "📞 Helplines:\n"
            "- Crisis Text Line: Text HOME to 741741\n"
            "- National Suicide Prevention Lifeline: 1-800-273-TALK\n"
            "- Trevor Project (LGBTQ+): 1-866-488-7386"
        )

    def summarize_retrieved(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        return summary_chain.run(docs)

    tools = [
        Tool(
            name="SummarizedDocSearch",
            func=summarize_retrieved,
            description="Retrieve and summarize relevant mental health information from the knowledge base.",
        ),
        Tool(
            name="EmergencyHelpline",
            func=suggest_helpline,
            description="Provide emergency mental health helpline numbers.",
        ),
        Tool(
            name="NearbyMentalHealthClinics",
            func=lambda q: find_clinics(q),
            description="Find nearby mental health clinics given a city name.",
        ),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    logger.info("LangChain agent initialized.")
    return agent