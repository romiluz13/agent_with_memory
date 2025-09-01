"""
🎯 AGENT BUILDER - The Missing Piece
A helper class to make agent creation even simpler
"""

from typing import List, Optional, Dict, Any
from langchain.agents import tool
from src.core.agent_langgraph import MongoDBLangGraphAgent
import os


class AgentBuilder:
    """
    Simplified agent builder for common use cases.
    Makes your boilerplate even easier to use!
    """
    
    @staticmethod
    def create_customer_support_agent(
        company_name: str,
        custom_tools: Optional[List] = None,
        mongodb_uri: Optional[str] = None
    ) -> MongoDBLangGraphAgent:
        """Create a pre-configured customer support agent."""
        
        prompt = f"""
        You are a professional customer support agent for {company_name}.
        You are helpful, empathetic, and solution-oriented.
        Remember customer details and preferences for personalized service.
        Always aim to resolve issues efficiently while maintaining a friendly tone.
        
        You have access to these tools: {{tool_names}}
        """
        
        return MongoDBLangGraphAgent(
            mongodb_uri=mongodb_uri or os.getenv("MONGODB_URI"),
            agent_name=f"{company_name.lower()}_support",
            system_prompt=prompt,
            user_tools=custom_tools or [],
            database_name=f"{company_name.lower()}_agent"
        )
    
    @staticmethod
    def create_sales_agent(
        company_name: str,
        product_catalog: Optional[Dict[str, Any]] = None,
        custom_tools: Optional[List] = None,
        mongodb_uri: Optional[str] = None
    ) -> MongoDBLangGraphAgent:
        """Create a pre-configured sales agent."""
        
        prompt = f"""
        You are a knowledgeable sales representative for {company_name}.
        You understand customer needs and recommend appropriate solutions.
        You are consultative, not pushy, and focus on value.
        Remember customer preferences and buying history.
        
        You have access to these tools: {{tool_names}}
        """
        
        # Auto-create product lookup tool if catalog provided
        tools = custom_tools or []
        if product_catalog:
            @tool
            def lookup_product(product_name: str) -> str:
                """Look up product details from catalog."""
                for key, value in product_catalog.items():
                    if product_name.lower() in key.lower():
                        return f"{key}: {value}"
                return f"Product '{product_name}' not found in catalog"
            
            tools.append(lookup_product)
        
        return MongoDBLangGraphAgent(
            mongodb_uri=mongodb_uri or os.getenv("MONGODB_URI"),
            agent_name=f"{company_name.lower()}_sales",
            system_prompt=prompt,
            user_tools=tools,
            database_name=f"{company_name.lower()}_agent"
        )
    
    @staticmethod
    def create_research_agent(
        research_domain: str,
        custom_tools: Optional[List] = None,
        mongodb_uri: Optional[str] = None
    ) -> MongoDBLangGraphAgent:
        """Create a pre-configured research assistant."""
        
        prompt = f"""
        You are an expert research assistant specializing in {research_domain}.
        You help gather, analyze, and synthesize information.
        You cite sources and distinguish between facts and speculation.
        Remember research context and build upon previous findings.
        
        You have access to these tools: {{tool_names}}
        """
        
        return MongoDBLangGraphAgent(
            mongodb_uri=mongodb_uri or os.getenv("MONGODB_URI"),
            agent_name=f"{research_domain.lower()}_research",
            system_prompt=prompt,
            user_tools=custom_tools or [],
            database_name=f"{research_domain.lower()}_research"
        )
    
    @staticmethod
    def create_personal_assistant(
        user_name: str,
        preferences: Optional[Dict[str, str]] = None,
        custom_tools: Optional[List] = None,
        mongodb_uri: Optional[str] = None
    ) -> MongoDBLangGraphAgent:
        """Create a personalized AI assistant."""
        
        pref_text = ""
        if preferences:
            pref_text = f"User preferences: {', '.join([f'{k}: {v}' for k, v in preferences.items()])}"
        
        prompt = f"""
        You are {user_name}'s personal AI assistant.
        You learn and remember their preferences, habits, and needs.
        You provide proactive, personalized assistance.
        {pref_text}
        
        You have access to these tools: {{tool_names}}
        """
        
        return MongoDBLangGraphAgent(
            mongodb_uri=mongodb_uri or os.getenv("MONGODB_URI"),
            agent_name=f"{user_name.lower()}_assistant",
            system_prompt=prompt,
            user_tools=custom_tools or [],
            database_name=f"{user_name.lower()}_personal"
        )


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create a support agent in ONE LINE!
    support_agent = AgentBuilder.create_customer_support_agent("TechCorp")
    
    # Create a sales agent with product catalog
    sales_agent = AgentBuilder.create_sales_agent(
        "TechCorp",
        product_catalog={
            "Laptop Pro X1": "$1299 - High-performance laptop",
            "SmartPhone Z": "$799 - Latest smartphone",
            "Tablet Plus": "$499 - Professional tablet"
        }
    )
    
    # Create a research agent
    research_agent = AgentBuilder.create_research_agent("Machine Learning")
    
    # Create a personal assistant
    personal_agent = AgentBuilder.create_personal_assistant(
        "John",
        preferences={"communication": "concise", "tone": "professional"}
    )
    
    print("✅ All agents created successfully!")
    print("   Each agent has full memory capabilities and can be customized!")
