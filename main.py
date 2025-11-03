import asyncio
import json
import logging
import os
import shutil
import re
from typing import Dict, List, Optional, Any
from enum import Enum

import requests
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LLMProvider(Enum):
    """Enum for LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        
        # Determine which LLM provider to use
        self.llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        
        if self.llm_provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
        elif self.llm_provider == "gemini":
            self.api_key = os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        
        # Optional: Set default model names
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
        
        logging.info(f"Using LLM Provider: {self.llm_provider}")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key."""
        return self.api_key
    
    def get_llm_provider(self) -> str:
        """Get the LLM provider."""
        return self.llm_provider
    
    def get_model_name(self) -> str:
        """Get the appropriate model name based on provider."""
        if self.llm_provider == "openai":
            return self.openai_model
        elif self.llm_provider == "gemini":
            return self.gemini_model
        return ""


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.stdio_context: Optional[Any] = None
        self.session: Optional[ClientSession] = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.capabilities: Optional[Dict[str, Any]] = None

    async def initialize(self) -> None:
        """Initialize the server connection."""
        server_params = StdioServerParameters(
            command=shutil.which("npx") if self.config['command'] == "npx" else self.config['command'],
            args=self.config['args'],
            env={**os.environ, **self.config['env']} if self.config.get('env') else None
        )
        try:
            self.stdio_context = stdio_client(server_params)
            read, write = await self.stdio_context.__aenter__()
            self.session = ClientSession(read, write)
            await self.session.__aenter__()
            self.capabilities = await self.session.initialize()
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[Any]:
        """List available tools from the server."""
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        
        tools_response = await self.session.list_tools()
        tools = []
        
        supports_progress = (
            self.capabilities 
            and 'progress' in self.capabilities
        )
        
        if supports_progress:
            logging.info(f"Server {self.name} supports progress tracking")
        
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == 'tools':
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))
                    if supports_progress:
                        logging.info(f"Tool '{tool.name}' will support progress tracking")
        
        return tools

    async def execute_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Any:
        """Execute a tool once (no retry logic here)."""
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        try:
            supports_progress = (
                self.capabilities 
                and 'progress' in self.capabilities
            )

            if supports_progress:
                logging.info(f"Executing {tool_name} with progress tracking...")
                result = await self.session.call_tool(
                    tool_name, 
                    arguments,
                    progress_token=f"{tool_name}_execution"
                )
            else:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)

            return result

        except Exception as e:
            logging.error(f"Tool execution error: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                if self.session:
                    try:
                        await self.session.__aexit__(None, None, None)
                    except Exception as e:
                        logging.warning(f"Warning during session cleanup for {self.name}: {e}")
                    finally:
                        self.session = None

                if self.stdio_context:
                    try:
                        await self.stdio_context.__aexit__(None, None, None)
                    except (RuntimeError, asyncio.CancelledError) as e:
                        logging.info(f"Note: Normal shutdown message for {self.name}: {e}")
                    except Exception as e:
                        logging.warning(f"Warning during stdio cleanup for {self.name}: {e}")
                    finally:
                        self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM."""
        args_desc = []
        if 'properties' in self.input_schema:
            for param_name, param_info in self.input_schema['properties'].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get('required', []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)
        
        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str, provider: str, model_name: str) -> None:
        self.api_key: str = api_key
        self.provider: str = provider
        self.model_name: str = model_name

    def _convert_messages_for_gemini(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to Gemini format."""
        gemini_messages = []
        
        # Track system content to prepend to next user message
        pending_system_content = ""
        
        for msg in messages:
            if msg["role"] == "system":
                # Accumulate system messages to prepend to next user message
                pending_system_content += msg["content"] + "\n\n"
            elif msg["role"] == "user":
                # Combine any pending system content with user message
                content = pending_system_content + msg["content"] if pending_system_content else msg["content"]
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
                pending_system_content = ""  # Reset after using
            elif msg["role"] == "assistant":
                # If there's pending system content and no user message follows,
                # add it as context to the assistant message
                if pending_system_content and gemini_messages:
                    # Add as a user message for context
                    gemini_messages.append({
                        "role": "user", 
                        "parts": [{"text": pending_system_content.strip()}]
                    })
                    pending_system_content = ""
                
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": msg["content"]}]
                })
        
        # Handle any remaining system content at the end
        if pending_system_content and gemini_messages:
            gemini_messages.append({
                "role": "user",
                "parts": [{"text": pending_system_content.strip()}]
            })
        
        return gemini_messages

    def get_response_openai(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from OpenAI."""
        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "messages": messages,
            "model": self.model_name,
            "temperature": 0.7,
            "top_p": 1,
            "stream": False
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            error_message = f"Error getting OpenAI response: {str(e)}"
            logging.error(error_message)
            
            if e.response is not None:
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")
                
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."

    def get_response_gemini(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from Google Gemini."""
        # Correct API endpoint for Gemini
        base_url = "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
        
        # Convert messages to Gemini format
        gemini_messages = self._convert_messages_for_gemini(messages)
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": 0.7,
                "topP": 1.0,
                "maxOutputTokens": 2048,
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Extract text from Gemini response
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        return parts[0]["text"]
            
            return "No response generated from Gemini."
            
        except requests.exceptions.RequestException as e:
            error_message = f"Error getting Gemini response: {str(e)}"
            logging.error(error_message)
            
            if e.response is not None:
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")
                
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the configured LLM provider."""
        if self.provider == "openai":
            return self.get_response_openai(messages)
        elif self.provider == "gemini":
            return self.get_response_gemini(messages)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: List[Server], llm_client: LLMClient, max_retries: int = 3, max_tool_calls: int = 10) -> None:
        self.servers: List[Server] = servers
        self.llm_client: LLMClient = llm_client
        self.max_retries: int = max_retries
        self.max_tool_calls: int = max_tool_calls  # Prevent infinite loops

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))
        
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def execute_tool_with_intelligent_retry(
        self, 
        tool_call: Dict[str, Any], 
        conversation_context: List[Dict[str, str]]
    ) -> str:
        """
        Execute a tool with intelligent retry logic where LLM learns from errors.
        
        Args:
            tool_call: The tool call dictionary with 'tool' and 'arguments'
            conversation_context: The message history for context
            
        Returns:
            Success message with results or failure message
        """
        retry_count = 0
        error_history = []
        
        while retry_count < self.max_retries:
            try:
                logging.info(f"Attempt {retry_count + 1}/{self.max_retries}: Executing tool '{tool_call['tool']}'")
                logging.info(f"With arguments: {json.dumps(tool_call['arguments'], indent=2)}")
                
                # Find the server with this tool and execute
                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        result = await server.execute_tool(
                            tool_call["tool"], 
                            tool_call["arguments"]
                        )
                        
                        if isinstance(result, dict) and 'progress' in result:
                            progress = result['progress']
                            total = result['total']
                            logging.info(f"Progress: {progress}/{total} ({(progress/total)*100:.1f}%)")
                        
                        if result.isError == True or (result.content and "error" in result.content):
                            logging.error(f"Error on attempt {retry_count + 1}: {result}")
                            return f"ERROR: {result}"

                        logging.info(f"✓ Tool executed successfully on attempt {retry_count + 1}")
                        return f"SUCCESS: {result}"
                
                return f"ERROR: No server found with tool: {tool_call['tool']}"
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                error_history.append({
                    "attempt": retry_count,
                    "error": error_msg,
                    "arguments": tool_call["arguments"]
                })
                
                logging.warning(f"✗ Attempt {retry_count} failed: {error_msg}")
                
                if retry_count >= self.max_retries:
                    # All retries exhausted
                    logging.error(f"All {self.max_retries} attempts failed")
                    return self._format_final_failure(tool_call, error_history)
                
                # Ask LLM to analyze the error and suggest a fix AUTOMATICALLY
                logging.info(f"🔄 Auto-retry: Asking LLM to analyze error and fix automatically...")
                new_tool_call = await self._ask_llm_to_fix_error(
                    tool_call, 
                    error_msg, 
                    error_history,
                    conversation_context
                )
                
                if new_tool_call:
                    tool_call = new_tool_call
                    logging.info(f"✓ LLM auto-corrected the approach. Retrying immediately...")
                    await asyncio.sleep(0.5)  # Brief delay before retry
                    # Loop continues automatically - no user interaction needed
                else:
                    logging.error("LLM couldn't auto-fix the error")
                    return self._format_final_failure(tool_call, error_history)
        
        return self._format_final_failure(tool_call, error_history)

    async def _ask_llm_to_fix_error(
        self,
        original_tool_call: Dict[str, Any],
        error_message: str,
        error_history: List[Dict[str, Any]],
        conversation_context: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """
        Ask LLM to analyze the error and suggest a corrected tool call.
        
        Returns:
            Modified tool call dict or None if LLM can't fix it
        """
        # Get available tools for context
        all_tools = []
        for server in self.servers:
            tools = await server.list_tools()
            all_tools.extend(tools)
        
        tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
        
        error_analysis_prompt = f"""TOOL EXECUTION ERROR - AUTO-FIX REQUIRED

You are an automated error correction system. A tool call failed and you must fix it automatically.

Original tool call:
{json.dumps(original_tool_call, indent=2)}

Error message: {error_message}

Previous failed attempts:
{json.dumps(error_history, indent=2)}

Available tools:
{tools_description}

YOUR TASK: Analyze the error and provide a CORRECTED tool call that will work.

Common fixes:
1. MongoDB errors with $toDouble: Wrap in $cond to check if value is numeric first
2. Empty string errors: Add validation before conversion
3. Wrong field names: Check the actual schema
4. Pipeline errors: Simplify or reorder stages
5. Type conversion errors: Use $ifNull and type checking

YOU MUST respond with ONLY a JSON object (no explanations, no questions, no text):

If fixable:
{{
    "tool": "corrected-tool-name",
    "arguments": {{
        "arg": "corrected-value"
    }}
}}

If unfixable after {len(error_history)} attempts:
{{"unfixable": true, "reason": "brief technical reason"}}

CRITICAL: 
- DO NOT ask questions
- DO NOT request confirmation
- DO NOT explain your changes
- ONLY return the JSON object
- This is automatic - no human will see this"""

        # Create a fresh context for error analysis (not part of main conversation)
        analysis_messages = [
            {
                "role": "system",
                "content": error_analysis_prompt
            }
        ]
        
        try:
            llm_response = self.llm_client.get_response(analysis_messages)
            logging.info(f"LLM error analysis: {llm_response}")
            
            fixed_call = json.loads(llm_response)
            
            if fixed_call.get("unfixable"):
                logging.warning(f"LLM says error is unfixable: {fixed_call.get('reason')}")
                return None
            
            if "tool" in fixed_call and "arguments" in fixed_call:
                return fixed_call
            
            return None
            
        except json.JSONDecodeError as e:
            logging.error(f"LLM didn't return valid JSON for error analysis: {e}")
            return None
        except Exception as e:
            logging.error(f"Error during LLM analysis: {e}")
            return None

    def _format_final_failure(
        self, 
        tool_call: Dict[str, Any], 
        error_history: List[Dict[str, Any]]
    ) -> str:
        """Format a user-friendly failure message."""
        return f"""FINAL_FAILURE: I apologize, but I wasn't able to complete your request after {self.max_retries} attempts.

Tool attempted: {tool_call.get('tool', 'unknown')}

What went wrong:
{chr(10).join([f"- Attempt {err['attempt']}: {err['error']}" for err in error_history])}

I've tried different approaches but couldn't resolve the issue. This might be because:
• The tool requires different parameters than what I tried
• The requested operation isn't supported by the available tools
• There's a temporary service issue

Please try rephrasing your request or ask me to try a different approach."""

    def extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response, handling markdown code blocks."""
        # Try to parse as plain JSON first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
        # Pattern matches ```json ... ``` or just ``` ... ```
        json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
        
        # Try to find JSON object without code blocks
        # Look for content between first { and last }
        try:
            first_brace = response.find('{')
            last_brace = response.rfind('}')
            if first_brace != -1 and last_brace != -1:
                json_str = response[first_brace:last_brace + 1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        return None

    def _clean_tool_result(self, result: str) -> str:
        """Extract clean content from tool result."""
        if "SUCCESS:" in result and "content=" in result:
            try:
                text_match = re.search(r"text='([^']*)'", result)
                if text_match:
                    return text_match.group(1).replace('\\n', '\n')
            except:
                pass
        return result

    async def process_user_request_with_sequential_tools(
        self, 
        user_input: str,
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Process a user request that may require multiple sequential tool calls.
        
        This method implements an agent loop that:
        1. Gets LLM response
        2. If it's a tool call, executes it and feeds result back to LLM
        3. Repeats until LLM gives a final natural language response
        4. Returns the final response to user
        
        Args:
            user_input: The user's query
            messages: Conversation history
            
        Returns:
            Final natural language response
        """
        tool_call_count = 0
        
        while tool_call_count < self.max_tool_calls:
            # Get LLM response
            llm_response = self.llm_client.get_response(messages)
            logging.info(f"\n{'='*60}")
            logging.info(f"LLM Response #{tool_call_count + 1}: {llm_response[:200]}...")
            
            # Check if it's a tool call
            tool_call = self.extract_json_from_response(llm_response)
            
            if not tool_call or "tool" not in tool_call or "arguments" not in tool_call:
                # Natural language response - we're done!
                logging.info("✓ LLM provided final natural language response")
                return llm_response
            
            # It's a tool call - execute it
            tool_call_count += 1
            logging.info(f"🔧 Tool call #{tool_call_count}: {tool_call['tool']}")
            print(f"\n🔧 Executing tool #{tool_call_count}: {tool_call['tool']}...")
            
            # Execute the tool with retry logic
            result = await self.execute_tool_with_intelligent_retry(tool_call, messages)
            
            # Check for failure
            if result.startswith("FINAL_FAILURE:"):
                logging.error("Tool execution failed after retries")
                return result
            
            # Clean the result for better LLM processing
            clean_result = self._clean_tool_result(result)
            
            # Add tool call and result to conversation history
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({
                "role": "user", 
                "content": f"""Tool execution completed. Result:

{clean_result}

Based on this result, decide:
1. If you need to call another tool to complete the user's goal, respond with the next tool call in JSON format
2. If the goal is complete, provide a natural language response explaining the results to the user

Remember the original goal: {user_input}"""
            })
            
            logging.info(f"✓ Tool result added to context. Continuing agent loop...")
        
        # Safety check: too many tool calls
        logging.warning(f"⚠️ Reached maximum tool calls limit ({self.max_tool_calls})")
        return f"I've made {self.max_tool_calls} tool calls but haven't completed the task. The task might be too complex or require a different approach. Here's what I found so far based on the tool results."

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            # Initialize all servers
            for server in self.servers:
                try:
                    await server.initialize()
                    logging.info(f"✓ Successfully initialized server: {server.name}")
                except Exception as e:
                    logging.error(f"Failed to initialize server {server.name}: {e}")
                    await self.cleanup_servers()
                    return
            
            # Get all available tools
            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)
                logging.info(f"Server {server.name} provides {len(tools)} tools")
            
            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
            
            system_message = f"""You are an AI agent with access to tools that help you complete user requests. You can call multiple tools in sequence to achieve complex goals.

AVAILABLE TOOLS:
{tools_description}

CRITICAL INSTRUCTIONS FOR MULTI-STEP WORKFLOWS:

1. SEQUENTIAL TOOL EXECUTION:
   - You can make MULTIPLE tool calls to complete a single user request
   - After each tool execution, you'll receive the result
   - Analyze the result and decide: do you need another tool call, or can you answer the user?
   - Continue calling tools until you have all information needed to answer the user

2. TOOL CALL FORMAT (when you need to call a tool):
   - Respond with ONLY a JSON object:
   {{
       "tool": "tool-name",
       "arguments": {{
           "argument-name": "value"
       }}
   }}
   - NO markdown code blocks (no ```)
   - NO explanatory text before or after the JSON

3. FINAL RESPONSE FORMAT (when you're done with tools):
   - After gathering all needed information via tools
   - Provide a clear, natural language response
   - Summarize findings from all tool calls
   - Answer the user's original question comprehensively

4. PLANNING COMPLEX TASKS:
   - Break down complex requests into sequential tool calls
   - Example: "Show me customers who bought product X and their total spending"
     → Call 1: Find customers who bought product X
     → Call 2: Calculate total spending for those customers
     → Final response: Summarize the results

5. If no tool is needed for a query, respond directly in natural language.

Remember: You are an autonomous agent. Make decisions about which tools to call and when to stop."""

            messages = [
                {
                    "role": "system",
                    "content": system_message
                }
            ]

            print("\n🤖 Multi-step Agent Chat Session Started!")
            print(f"📡 Using {self.llm_client.provider.upper()} with model: {self.llm_client.model_name}")
            print(f"📦 Loaded {len(all_tools)} tools from {len(self.servers)} server(s)")
            print(f"🔧 Max sequential tool calls per request: {self.max_tool_calls}")
            print("\nType 'quit' or 'exit' to end the session.\n")

            while True:
                try:
                    user_input = input("You: ").strip()
                    if user_input.lower() in ['quit', 'exit']:
                        logging.info("\nExiting...")
                        break

                    if not user_input:
                        continue

                    # Add user message to history
                    messages.append({"role": "user", "content": user_input})
                    
                    # Process request with sequential tool calling
                    final_response = await self.process_user_request_with_sequential_tools(
                        user_input, 
                        messages
                    )
                    
                    # Display final response to user
                    print(f"\n🤖 Assistant: {final_response}\n")
                    
                    # Add final response to history
                    messages.append({"role": "assistant", "content": final_response})

                except KeyboardInterrupt:
                    logging.info("\n\nInterrupted by user. Exiting...")
                    break
                except Exception as e:
                    logging.error(f"Error in chat loop: {e}", exc_info=True)
                    print(f"\n❌ An error occurred: {e}\n")
                    print("Continuing chat session...\n")
        
        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    server_config = config.load_config('servers_config.json')
    servers = [Server(name, srv_config) for name, srv_config in server_config['mcpServers'].items()]
    
    # Create LLM client with provider info
    llm_client = LLMClient(
        api_key=config.llm_api_key,
        provider=config.get_llm_provider(),
        model_name=config.get_model_name()
    )
    
    # Configure max_retries and max_tool_calls
    # max_retries: retry attempts per tool if it fails
    # max_tool_calls: maximum sequential tool calls per user request
    chat_session = ChatSession(
        servers, 
        llm_client, 
        max_retries=3,
        max_tool_calls=10  # Adjust based on your needs
    )
    await chat_session.start()

if __name__ == "__main__":
    asyncio.run(main())