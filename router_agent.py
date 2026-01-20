#!/usr/bin/env python3
"""
Router Agent - Intelligent request routing to specialized agents.

This agent analyzes incoming requests and routes them to the appropriate
specialized agent based on the intent and domain of the request.

SUPPORTED AGENTS:
- graphweaver: Database FK discovery, knowledge graphs, embeddings
- loan: Loan application processing, credit checks, approvals

USAGE:
    python router_agent.py                    # Interactive mode
    python router_agent.py --auto "message"   # Single message mode
"""
import os
import sys
from typing import Optional, Any, Dict, List, TypedDict
from enum import Enum

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=False, write_through=True)

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage


# =============================================================================
# Agent Registry
# =============================================================================

class AgentType(str, Enum):
    """Available specialized agents."""
    GRAPHWEAVER = "graphweaver"
    LOAN = "loan"
    ROUTER = "router"  # Self-reference for meta queries


# Store for agent instances (lazy loaded)
_agent_instances: Dict[str, Any] = {}
_agent_histories: Dict[str, List[Dict]] = {}


def get_agent_instance(agent_type: AgentType):
    """Get or create an agent instance."""
    global _agent_instances
    
    if agent_type.value not in _agent_instances:
        if agent_type == AgentType.GRAPHWEAVER:
            # Import and create GraphWeaver agent
            try:
                from agent import create_graphweaver_agent
                _agent_instances[agent_type.value] = create_graphweaver_agent(verbose=True)
            except ImportError:
                return None
        elif agent_type == AgentType.LOAN:
            # Import and create Loan agent
            try:
                from loan_agent import create_loan_agent
                _agent_instances[agent_type.value] = create_loan_agent(verbose=True)
            except ImportError:
                return None
    
    return _agent_instances.get(agent_type.value)


def get_agent_history(agent_type: AgentType) -> List[Dict]:
    """Get conversation history for an agent."""
    global _agent_histories
    if agent_type.value not in _agent_histories:
        _agent_histories[agent_type.value] = []
    return _agent_histories[agent_type.value]


def add_to_agent_history(agent_type: AgentType, role: str, content: str):
    """Add a message to an agent's history."""
    history = get_agent_history(agent_type)
    history.append({"role": role, "content": content})
    # Keep only last 20 messages
    if len(history) > 20:
        _agent_histories[agent_type.value] = history[-20:]


# =============================================================================
# Router Tools
# =============================================================================

@tool
def route_to_graphweaver(message: str) -> str:
    """Route a request to the GraphWeaver agent for database and knowledge graph tasks.
    
    Use this for:
    - Database exploration and FK discovery
    - Knowledge graph building and analysis
    - Embeddings and semantic search
    - Business rules and lineage tracking
    - RDF/SPARQL operations
    - LTN rule learning
    
    Args:
        message: The user's request to process
        
    Returns:
        The GraphWeaver agent's response
    """
    try:
        agent = get_agent_instance(AgentType.GRAPHWEAVER)
        if agent is None:
            return "ERROR: GraphWeaver agent not available. Make sure agent.py is in the same directory."
        
        history = get_agent_history(AgentType.GRAPHWEAVER)
        messages = history + [{"role": "user", "content": message}]
        
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )
        
        # Extract response
        response = extract_agent_response(result)
        
        # Update history
        add_to_agent_history(AgentType.GRAPHWEAVER, "user", message)
        add_to_agent_history(AgentType.GRAPHWEAVER, "assistant", response)
        
        return f"[GraphWeaver Agent]\n\n{response}"
    except Exception as e:
        import traceback
        return f"ERROR routing to GraphWeaver: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def route_to_loan_agent(message: str) -> str:
    """Route a request to the Loan Application agent for loan processing tasks.
    
    Use this for:
    - Loan application submission and processing
    - Credit score checks and risk assessment
    - Loan approval/rejection decisions
    - Interest rate calculations
    - Payment schedule generation
    - Document verification
    - Loan status inquiries
    
    Args:
        message: The user's request to process
        
    Returns:
        The Loan agent's response
    """
    try:
        agent = get_agent_instance(AgentType.LOAN)
        if agent is None:
            return "ERROR: Loan agent not available. Make sure loan_agent.py is in the same directory."
        
        history = get_agent_history(AgentType.LOAN)
        messages = history + [{"role": "user", "content": message}]
        
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )
        
        # Extract response
        response = extract_agent_response(result)
        
        # Update history
        add_to_agent_history(AgentType.LOAN, "user", message)
        add_to_agent_history(AgentType.LOAN, "assistant", response)
        
        return f"[Loan Agent]\n\n{response}"
    except Exception as e:
        import traceback
        return f"ERROR routing to Loan Agent: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def list_available_agents() -> str:
    """List all available specialized agents and their capabilities.
    
    Returns:
        Description of available agents
    """
    output = """## Available Specialized Agents

### 1. GraphWeaver Agent (`route_to_graphweaver`)
**Domain**: Database exploration, knowledge graphs, metadata management

**Capabilities**:
- Database connection and table exploration
- Foreign key discovery (5-stage pipeline)
- Neo4j knowledge graph building and analysis
- Text and graph embeddings for semantic search
- Business rules execution with lineage tracking
- RDF/SPARQL triple store operations
- LTN logical rule learning

**Example requests**:
- "Connect to the database and show me the tables"
- "Discover all foreign key relationships"
- "Find tables similar to 'orders'"
- "What would break if I change the customers table?"

---

### 2. Loan Application Agent (`route_to_loan_agent`)
**Domain**: Loan processing, credit assessment, financial decisions

**Capabilities**:
- Submit and process loan applications
- Credit score evaluation
- Risk assessment and scoring
- Loan approval/rejection with reasoning
- Interest rate calculation
- Payment schedule generation
- Document verification
- Loan status tracking

**Example requests**:
- "Submit a loan application for $50,000"
- "Check credit score for customer ID 12345"
- "Calculate monthly payment for a 30-year mortgage at 6.5%"
- "What's the status of loan application LA-2024-001?"

---

## How Routing Works

The router agent analyzes your request and automatically routes it to the 
appropriate specialized agent. You can also explicitly request a specific agent.

**Tips**:
- Be specific about what you need
- Include relevant IDs, amounts, or identifiers
- The router maintains separate conversation history for each agent
"""
    return output


@tool
def get_agent_status() -> str:
    """Check the status and availability of all agents.
    
    Returns:
        Status of each agent
    """
    output = "## Agent Status\n\n"
    
    for agent_type in AgentType:
        if agent_type == AgentType.ROUTER:
            continue
            
        name = agent_type.value.title()
        instance = _agent_instances.get(agent_type.value)
        history = _agent_histories.get(agent_type.value, [])
        
        if instance is not None:
            status = "✅ Active"
        else:
            # Try to load
            try:
                get_agent_instance(agent_type)
                if _agent_instances.get(agent_type.value):
                    status = "✅ Active (just loaded)"
                else:
                    status = "❌ Not available"
            except:
                status = "❌ Not available"
        
        output += f"### {name} Agent\n"
        output += f"- Status: {status}\n"
        output += f"- History: {len(history)} messages\n\n"
    
    return output


@tool
def clear_agent_history(agent_name: str = "all") -> str:
    """Clear conversation history for one or all agents.
    
    Args:
        agent_name: Agent name ('graphweaver', 'loan') or 'all'
        
    Returns:
        Confirmation message
    """
    global _agent_histories
    
    if agent_name.lower() == "all":
        _agent_histories = {}
        return "✓ Cleared history for all agents"
    
    agent_name = agent_name.lower()
    if agent_name in _agent_histories:
        del _agent_histories[agent_name]
        return f"✓ Cleared history for {agent_name} agent"
    
    return f"No history found for agent: {agent_name}"


@tool
def analyze_request_intent(message: str) -> str:
    """Analyze a user's message to determine which agent should handle it.
    
    This is a helper tool for understanding request routing logic.
    
    Args:
        message: The message to analyze
        
    Returns:
        Analysis of the request intent
    """
    message_lower = message.lower()
    
    # GraphWeaver keywords
    graphweaver_keywords = [
        "database", "table", "column", "foreign key", "fk", "schema",
        "neo4j", "graph", "cypher", "knowledge graph",
        "embedding", "semantic search", "similar",
        "business rule", "lineage", "marquez",
        "rdf", "sparql", "fuseki", "ontology",
        "ltn", "rule learning", "validation"
    ]
    
    # Loan keywords
    loan_keywords = [
        "loan", "mortgage", "credit", "borrow", "lend",
        "interest rate", "apr", "payment", "installment",
        "approve", "reject", "application", "applicant",
        "income", "debt", "dti", "risk", "score",
        "collateral", "guarantee", "cosigner"
    ]
    
    graphweaver_matches = [kw for kw in graphweaver_keywords if kw in message_lower]
    loan_matches = [kw for kw in loan_keywords if kw in message_lower]
    
    output = f"## Intent Analysis\n\n"
    output += f"**Message**: {message[:100]}{'...' if len(message) > 100 else ''}\n\n"
    
    output += f"### GraphWeaver Signals ({len(graphweaver_matches)})\n"
    output += f"Keywords found: {', '.join(graphweaver_matches) if graphweaver_matches else 'None'}\n\n"
    
    output += f"### Loan Agent Signals ({len(loan_matches)})\n"
    output += f"Keywords found: {', '.join(loan_matches) if loan_matches else 'None'}\n\n"
    
    if len(graphweaver_matches) > len(loan_matches):
        output += "**Recommended**: Route to GraphWeaver Agent\n"
    elif len(loan_matches) > len(graphweaver_matches):
        output += "**Recommended**: Route to Loan Agent\n"
    elif len(graphweaver_matches) == 0 and len(loan_matches) == 0:
        output += "**Recommended**: Ask for clarification\n"
    else:
        output += "**Recommended**: Ask user to specify which domain\n"
    
    return output


# =============================================================================
# Helper Functions
# =============================================================================

def extract_agent_response(result) -> str:
    """Extract text response from agent result."""
    if not isinstance(result, dict):
        return str(result)
    
    messages = result.get("messages", [])
    if not messages:
        return str(result)
    
    last_msg = messages[-1]
    content = getattr(last_msg, 'content', None)
    
    if content is None:
        return str(last_msg)
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get('type') == 'text':
                text_parts.append(block.get('text', ''))
            elif hasattr(block, 'text'):
                text_parts.append(block.text)
        return '\n'.join(text_parts) if text_parts else str(content)
    
    return str(content)


# =============================================================================
# Router System Prompt
# =============================================================================

ROUTER_SYSTEM_PROMPT = """You are the Router Agent - an intelligent request dispatcher that routes user requests to the appropriate specialized agent.

## Your Role

You analyze incoming requests and determine which specialized agent is best suited to handle them. You then route the request and relay the response back to the user.

## Available Agents

### 1. GraphWeaver Agent (route_to_graphweaver)
For database and knowledge graph tasks:
- Database exploration (tables, columns, schemas)
- Foreign key discovery
- Neo4j knowledge graph operations
- Semantic search with embeddings
- Business rules and data lineage
- RDF/SPARQL queries
- LTN rule learning

### 2. Loan Application Agent (route_to_loan_agent)
For financial and loan processing tasks:
- Loan application submission
- Credit score checks
- Risk assessment
- Loan approval decisions
- Interest rate calculations
- Payment schedules
- Document verification

## Routing Guidelines

1. **Analyze the intent**: Look for domain-specific keywords and context
2. **Route confidently**: When the intent is clear, route immediately
3. **Ask for clarification**: Only when genuinely ambiguous
4. **Provide context**: Include relevant details when routing
5. **Relay responses**: Present agent responses clearly

## Keywords for Routing

**GraphWeaver signals**: database, table, column, FK, schema, graph, Neo4j, 
embedding, semantic, lineage, RDF, SPARQL, business rules

**Loan signals**: loan, mortgage, credit, interest, payment, approve, reject,
application, income, debt, risk, collateral

## Available Tools

- `route_to_graphweaver(message)` - Route to database/graph agent
- `route_to_loan_agent(message)` - Route to loan processing agent
- `list_available_agents()` - Show all agents and capabilities
- `get_agent_status()` - Check agent availability
- `clear_agent_history(agent_name)` - Clear conversation history
- `analyze_request_intent(message)` - Analyze which agent fits best

## Example Interactions

User: "Show me the tables in the database"
→ Route to GraphWeaver (database exploration)

User: "I want to apply for a $100,000 mortgage"
→ Route to Loan Agent (loan application)

User: "What can you help me with?"
→ Use list_available_agents to show capabilities

Be helpful, efficient, and route requests to the right agent!"""


# =============================================================================
# All Router Tools
# =============================================================================

ROUTER_TOOLS = [
    route_to_graphweaver,
    route_to_loan_agent,
    list_available_agents,
    get_agent_status,
    clear_agent_history,
    analyze_request_intent,
]


# =============================================================================
# Middleware
# =============================================================================

@wrap_tool_call
def handle_router_errors(request, handler):
    """Handle tool execution errors."""
    try:
        return handler(request)
    except Exception as e:
        import traceback
        error_msg = f"Router error: {type(e).__name__}: {e}"
        print(f"[ROUTER ERROR] {error_msg}")
        traceback.print_exc()
        return ToolMessage(
            content=error_msg,
            tool_call_id=request.tool_call["id"]
        )


# =============================================================================
# Agent Creation
# =============================================================================

def create_router_agent(verbose: bool = True):
    """Create the Router agent.
    
    Args:
        verbose: Enable verbose output
        
    Returns:
        Configured router agent
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable required")
    
    model = ChatAnthropic(
        model="claude-opus-4-5-20251101",
        temperature=0.1,
        max_tokens=4096,
    )
    
    agent = create_agent(
        model=model,
        tools=ROUTER_TOOLS,
        system_prompt=ROUTER_SYSTEM_PROMPT,
        middleware=[handle_router_errors],
    )
    
    return agent


# =============================================================================
# Interactive Mode
# =============================================================================

def run_interactive():
    """Run router in interactive mode."""
    agent = create_router_agent(verbose=True)
    history = []
    
    print("\n" + "="*60)
    print("  🔀 Router Agent - Intelligent Request Dispatcher")
    print("="*60)
    print("\nI route your requests to specialized agents:")
    print("  • GraphWeaver: Database, graphs, embeddings, rules")
    print("  • Loan Agent: Applications, credit, approvals")
    print("\nTry saying:")
    print("  • 'Show me what agents are available'")
    print("  • 'Connect to the database and list tables'")
    print("  • 'I want to apply for a $50,000 personal loan'")
    print("  • 'Check credit score for customer 12345'")
    print("\nType 'quit' to exit.\n")
    sys.stdout.flush()
    
    while True:
        try:
            sys.stdout.write("\033[92mYou:\033[0m ")
            sys.stdout.flush()
            user_input = sys.stdin.readline()
            
            if not user_input:
                print("\nEnd of input.")
                break
            
            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            messages = history + [{"role": "user", "content": user_input}]
            
            print("\n\033[96m🔀 Router:\033[0m ", end="")
            sys.stdout.flush()
            
            # Stream response
            full_response = ""
            tool_calls_seen = set()
            
            for chunk in agent.stream(
                {"messages": messages},
                stream_mode="values",
                config={"recursion_limit": 100}
            ):
                if "messages" in chunk and chunk["messages"]:
                    latest_message = chunk["messages"][-1]
                    
                    content = getattr(latest_message, 'content', '')
                    if isinstance(content, str) and content:
                        new_content = content[len(full_response):]
                        if new_content:
                            print(new_content, end="", flush=True)
                            full_response = content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text = block.get('text', '')
                                new_content = text[len(full_response):]
                                if new_content:
                                    print(new_content, end="", flush=True)
                                    full_response = text
                    
                    tool_calls = getattr(latest_message, 'tool_calls', None)
                    if tool_calls:
                        for tc in tool_calls:
                            tc_id = tc.get('id', '')
                            if tc_id and tc_id not in tool_calls_seen:
                                tool_calls_seen.add(tc_id)
                                print(f"\n\n🔧 **{tc.get('name', 'unknown')}**", flush=True)
            
            print("\n")
            sys.stdout.flush()
            
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": full_response})
            
            if len(history) > 20:
                history = history[-20:]
                
        except EOFError:
            print("\nEnd of input.")
            break
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()
    
    print("\n👋 Goodbye!")


def run_single_message(message: str):
    """Process a single message and exit."""
    agent = create_router_agent(verbose=True)
    
    print(f"\n🔀 Processing: {message}\n")
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"recursion_limit": 100}
    )
    
    response = extract_agent_response(result)
    print(response)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Router Agent - Request Dispatcher")
    parser.add_argument("--auto", type=str, help="Process a single message")
    parser.add_argument("message", nargs="?", help="Message to process (with --auto)")
    
    args = parser.parse_args()
    
    if args.auto:
        run_single_message(args.auto)
    elif args.message:
        run_single_message(args.message)
    else:
        run_interactive()


if __name__ == "__main__":
    main()