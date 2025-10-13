import asyncio
import json
import logging
import os
import shutil
from typing import Dict, List, Optional, Any

import requests
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("OPENAI_API_KEY")

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
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key


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

    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM."""
        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "messages": messages,
            "model": "gpt-5-nano",
            "temperature": 1,
            "top_p": 1,
            "stream": False,
            "stop": None
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)
            
            if e.response is not None:
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")
                
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: List[Server], llm_client: LLMClient, max_retries: int = 3) -> None:
        self.servers: List[Server] = servers
        self.llm_client: LLMClient = llm_client
        self.max_retries: int = max_retries

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
                logging.info(f"With arguments: {tool_call['arguments']}")
                
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
                
                # Ask LLM to analyze the error and suggest a fix
                logging.info(f"Asking LLM to analyze error and retry...")
                new_tool_call = await self._ask_llm_to_fix_error(
                    tool_call, 
                    error_msg, 
                    error_history,
                    conversation_context
                )
                
                if new_tool_call:
                    tool_call = new_tool_call
                    logging.info(f"LLM suggested new approach: {tool_call}")
                    await asyncio.sleep(1)  # Brief delay before retry
                else:
                    logging.error("LLM couldn't suggest a fix")
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
        
        error_analysis_prompt = f"""The tool execution failed. Here's what happened:

Original tool call:
{json.dumps(original_tool_call, indent=2)}

Error: {error_message}

Previous failed attempts:
{json.dumps(error_history, indent=2)}

Available tools:
{tools_description}

Analyze the error and suggest a corrected tool call. Consider:
1. Are the argument names correct?
2. Are the argument values in the right format?
3. Are all required arguments provided?
4. Should you try a different tool?
5. Is the user's request achievable with available tools?

If you can fix it, respond with ONLY a JSON object:
{{
    "tool": "corrected-tool-name",
    "arguments": {{
        "arg": "corrected-value"
    }}
}}

If the error is unfixable or the request is impossible with available tools, respond with:
{{"unfixable": true, "reason": "explanation"}}"""

        # Create a temporary message context for error analysis
        analysis_messages = conversation_context[-3:] + [  # Last 3 messages for context
            {
                "role": "system",
                "content": error_analysis_prompt
            }
        ]
        
        try:
            llm_response = self.llm_client.get_response(analysis_messages)
            logging.info(f"LLM analysis: {llm_response}")
            
            fixed_call = json.loads(llm_response)
            
            if fixed_call.get("unfixable"):
                logging.warning(f"LLM says error is unfixable: {fixed_call.get('reason')}")
                return None
            
            if "tool" in fixed_call and "arguments" in fixed_call:
                return fixed_call
            
            return None
            
        except json.JSONDecodeError as e:
            logging.error(f"LLM didn't return valid JSON: {e}")
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

    async def process_llm_response(
        self, 
        llm_response: str,
        conversation_context: List[Dict[str, str]]
    ) -> str:
        """Process the LLM response and execute tools if needed."""
        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                # Use intelligent retry logic
                result = await self.execute_tool_with_intelligent_retry(
                    tool_call,
                    conversation_context
                )
                return result
            
            return llm_response
            
        except json.JSONDecodeError:
            return llm_response

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            # Initialize all servers
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return
            
            # Get all available tools
            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)
            
            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
            
            system_message = f"""You are a helpful assistant with access to these tools: 

{tools_description}

Choose the appropriate tool based on the user's question. If no tool is needed, reply directly.

IMPORTANT: When you need to use a tool, you must ONLY respond with the exact JSON object format below, nothing else:
{{
    "tool": "tool-name",
    "arguments": {{
        "argument-name": "value"
    }}
}}

After receiving a tool's response:
1. If it starts with "SUCCESS:", transform the data into a natural, conversational response
2. If it starts with "FINAL_FAILURE:", acknowledge the failure empathetically and suggest alternatives
3. Keep responses concise but informative
4. Focus on the most relevant information

Please use only the tools that are explicitly defined above."""

            messages = [
                {
                    "role": "system",
                    "content": system_message
                }
            ]

            print("\n🤖 Chat session started! Type 'quit' or 'exit' to end.\n")

            while True:
                try:
                    user_input = input("You: ").strip()
                    if user_input.lower() in ['quit', 'exit']:
                        logging.info("\nExiting...")
                        break

                    if not user_input:
                        continue

                    messages.append({"role": "user", "content": user_input})
                    
                    # Get LLM response
                    llm_response = self.llm_client.get_response(messages)
                    logging.info(f"\nLLM Response: {llm_response}")

                    # Process response (may trigger tool execution with retries)
                    result = await self.process_llm_response(llm_response, messages)
                    
                    if result != llm_response:
                        # Tool was executed
                        messages.append({"role": "assistant", "content": llm_response})
                        messages.append({"role": "system", "content": result})
                        
                        # Get final natural language response
                        final_response = self.llm_client.get_response(messages)
                        print(f"\n🤖 Assistant: {final_response}\n")
                        messages.append({"role": "assistant", "content": final_response})
                    else:
                        # Direct response
                        print(f"\n🤖 Assistant: {llm_response}\n")
                        messages.append({"role": "assistant", "content": llm_response})

                except KeyboardInterrupt:
                    logging.info("\n\nInterrupted by user. Exiting...")
                    break
                except Exception as e:
                    logging.error(f"Error in chat loop: {e}")
                    print(f"\n❌ An error occurred: {e}\n")
        
        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    server_config = config.load_config('servers_config.json')
    servers = [Server(name, srv_config) for name, srv_config in server_config['mcpServers'].items()]
    llm_client = LLMClient(config.llm_api_key)
    
    # You can configure max_retries here (default is 3)
    chat_session = ChatSession(servers, llm_client, max_retries=3)
    await chat_session.start()

if __name__ == "__main__":
    asyncio.run(main())