
import os
import json
import re
import chromadb
import openai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from urllib.parse import urlparse
import traceback # Added for LLM repair hook

load_dotenv()

# Configuration from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_BASE_URL = os.environ.get("OPENAI_API_BASE_URL") # Using this for the client, but model is Gemini
OPENAI_LLM_MODEL = os.environ.get("OPENAI_LLM_MODEL") # This should be your Gemini model name via OpenAI compatible endpoint
LANGCHAIN_EMBEDDING_MODEL = os.environ.get("LANGCHAIN_EMBEDDING_MODEL")
CHROMA_URL = os.environ.get("CHROMA_URL")

FUNCTION_COLLECTION_NAME = "dynamic_functions" # Stores definitions of LLM-generated functions

if not GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY not set")
if not OPENAI_API_BASE_URL: raise ValueError("OPENAI_API_BASE_URL not set")
if not OPENAI_LLM_MODEL: raise ValueError("OPENAI_LLM_MODEL not set")
if not LANGCHAIN_EMBEDDING_MODEL: raise ValueError("LANGCHAIN_EMBEDDING_MODEL not set")
if not CHROMA_URL: raise ValueError("CHROMA_URL not set")

# --- Definition for the Function Creation Tool (used by the executor itself) ---
FUNCTION_CREATION_TOOL_DEFINITION = {
    'type': 'function',
    'function': {
        'name': 'create_dynamic_function',
        'description': 'Define and create a new Python function string that can be executed later. Takes the desired function name, a high-level description of its purpose, and a parameter schema for the function itself. The generated function code will be executed in an environment where it can call external functions via an \'external_apis\' dictionary, whose available functions are described by the host if provided.',
        'parameters': {
            'type': 'object',
            'properties': {
                'new_function_name': {
                    'type': 'string',
                    'description': 'The name for the new Python function (use snake_case).',
                },
                'new_function_description': {
                    'type': 'string',
                    'description': 'A clear, high-level description of what the new function should achieve.',
                },
                'new_function_parameters_schema': {
                    'type': 'object',
                    'description': 'A JSON schema object describing the parameters the new function will accept in its `params` argument.',
                    'properties': {
                        'type': {'type': 'string', 'enum': ['object']},
                        'properties': {'type': 'object'},
                        'required': {'type': 'array', 'items': {'type': 'string'}}
                    },
                    'required': ['type', 'properties']
                }
                # host_provided_api_description is passed programmatically, not part of this schema for LLM to fill
            },
            'required': ['new_function_name', 'new_function_description', 'new_function_parameters_schema'],
        },
    },
}


def generate_function_creation_prompt(name, description, parameters_schema, host_provided_api_description=None):
    """Generates the prompt for the LLM to create a Python function string."""
    if not parameters_schema or not isinstance(parameters_schema, dict) or \
       parameters_schema.get('type') != 'object' or not isinstance(parameters_schema.get('properties'), dict):
        raise ValueError("Invalid parameters_schema provided for new function creation.")

    required_params_from_schema = parameters_schema.get('required', [])
    properties_from_schema = parameters_schema.get('properties', {})

    param_access_examples = []
    for param_name, param_info in properties_from_schema.items():
        param_type = param_info.get('type', 'any')
        is_required = param_name in required_params_from_schema
        access_method = f"params['{param_name}']" if is_required else f"params.get('{param_name}')"
        param_access_examples.append(f"- Access '{param_name}' ({param_type}, {'required' if is_required else 'optional'} from schema): `{access_method}`")

    param_access_section = "\n".join(param_access_examples) if param_access_examples \
        else "- This function takes no specific parameters from its schema; it might use contextual data from `params` if provided by the host (e.g. `params.get('soul_id')`, `params.get('artifact_properties')`)."

    api_usage_section = """
    **Using Host-Provided APIs:**
    - To interact with the host system or other services, use the `external_apis` dictionary.
    - This dictionary is provided by the host environment when your function is executed.
    - It contains callable functions provided by the host.
    - Call these functions like: `result = external_apis['some_host_api_name'](arguments_dictionary_for_host_api)`
    - The `arguments_dictionary_for_host_api` should contain parameters as expected by that specific host API.
    - Your function's `params` argument may contain contextual data (like `params.get('soul_id')`, `params.get('location_id')`, `params.get('artifact_properties')`) needed by the host APIs. Ensure you pass these correctly.
"""

    if host_provided_api_description:
        api_usage_section += f"""
    **Available Host APIs (in `external_apis` dictionary):**
    ```
    {host_provided_api_description}
    ```
    When deciding which host API to call, refer to the descriptions above. Ensure you construct the `arguments_dictionary_for_host_api` precisely as required by that API.
    If your function's `params` has relevant data (e.g. `params['direction']`), use it to build the arguments for the host API.
    Contextual data like `params.get('soul_id')` or `params.get('location_id')` might be implicitly available in `params` if the host provides it, and may be needed by host APIs.
"""
    else:
        api_usage_section += """
    - No specific host API descriptions were provided for this task. If you need to call host functions, assume they exist in `external_apis` and are well-known, or make your function self-contained if possible.
"""
    example_code_structure_template = """
def {function_name}(params):
    # `params` is a dictionary passed by the host environment.
    # It contains arguments as defined in your function's parameters_schema.
    # It might also contain additional contextual data from the host (e.g., params.get('soul_id'), params.get('artifact_properties')).
    # Remember, if a host API returns a JSON string, use `json.loads()` to parse it.

    try:
        # CORRECT INDENTATION IS CRITICAL.
        # Example:
        #   required_data = params['my_required_param'] # Access required param
        #   optional_data = params.get('my_optional_param', 'default_value') # Access optional param
        #   
        #   if 'host_api_example' in external_apis:
        #       api_args = {{'data_for_api': required_data, 'user_context': params.get('soul_id')}}
        #       host_result_str = external_apis['host_api_example'](api_args)
        #       # host_data = json.loads(host_result_str) # If API returns JSON string
        #       return f"Host API said: {{host_result_str}}"
        #   else:
        #       return f"Logic for {function_name} with {{required_data}}. No host API call needed or API not found."

        # Replace the above example with your actual logic.
        return f"Function '{function_name}' executed. Input params: {{str(params)}}. Implement your logic here."

    except KeyError as e:
        return f"Error in {function_name}: Missing expected key in params: {{e}}"
    except Exception as error:
        return f"Error executing {function_name}: {{error}}"
"""
    example_code_structure = example_code_structure_template.format(function_name=name)

    return f"""You are an expert Python function generator. Your task is to create a single, standalone Python function string based on the provided specification.

    **Function Specification:**
    - Name: {name}
    - Description: {description}
    - Parameters Schema (for the 'params' argument this function will receive):
    ```json
    {json.dumps(parameters_schema, indent=2)}
    ```

    {api_usage_section}

    **CRITICAL Instructions for Python Code Generation:**
    1.  Write a single Python function named precisely `{name}`.
    2.  The function MUST accept a single argument: a dictionary named `params`.
    3.  The function should perform the action described in its high-level description, primarily by calling functions from the `external_apis` dictionary.
    4.  The function MUST return a single string indicating the result or outcome.
    5.  **PYTHON INDENTATION IS PARAMOUNT!** Ensure all logic, especially within `try` and `except` blocks, is correctly indented.
    6.  Do NOT include any comments, explanations, or surrounding text outside the function definition itself. Output only the `def {name}(params): ...` block.
    7.  Do NOT include markdown code block markers (e.g., ```python or ```) in your output.
    8.  Handle potential errors gracefully. Return an informative error string starting with "Error: ".
    9.  **Remember**: If a host API is documented to return a JSON string, you MUST use `json.loads(result_string)` to parse it into a Python dictionary or list before accessing its elements. The `json` module is available.

    **Schema Parameter Access Examples (from `params` dictionary):**
    {param_access_section}

    **Example Function Structure (Pay ATTENTION to INDENTATION and json.loads):**
    ```python
{example_code_structure.strip()}
    ```

    **Your Task:**
    Generate *only* the Python function code for `{name}` based *exactly* on the specification provided above. Ensure all Python syntax, especially indentation, is flawless.
    """


class DynamicFunctionExecutor:
    def __init__(self):
        # Ensure GEMINI_API_KEY is used for the OpenAI client if that's the intended setup for compatibility layer
        self.openai_client = openai.OpenAI(base_url=OPENAI_API_BASE_URL, api_key=GEMINI_API_KEY)
        self.embeddings_client = GoogleGenerativeAIEmbeddings(model=LANGCHAIN_EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)
        parsed_url = urlparse(CHROMA_URL)
        self.chroma_client = chromadb.HttpClient(host=parsed_url.hostname, port=parsed_url.port)
        self.collection = None
        self.is_debug = True # Set to False to reduce console noise
        self.llm_repair_callback = None # For LLM-assisted repair hook

    def debug_log(self, *args):
        if self.is_debug: print('[DEBUG DynamicExecutor]', *args)

    def initialize_store(self):
        self.debug_log("Initializing DynamicFunctionExecutor store...")
        try:
            self.collection = self.chroma_client.get_or_create_collection(name=FUNCTION_COLLECTION_NAME)
            self.debug_log(f"Chroma collection '{FUNCTION_COLLECTION_NAME}' ready.")
            self._ensure_function_creation_tool_definition_in_store()
            self.debug_log("DynamicFunctionExecutor store initialized successfully.")
        except Exception as error:
            print(f"Error initializing DynamicFunctionExecutor store: {error}"); raise error

    def clear_function_store(self):
        self.debug_log(f"Attempting to delete and recreate collection: {FUNCTION_COLLECTION_NAME}")
        try:
            self.chroma_client.delete_collection(name=FUNCTION_COLLECTION_NAME)
            self.collection = self.chroma_client.get_or_create_collection(name=FUNCTION_COLLECTION_NAME)
            self.debug_log(f"Collection '{FUNCTION_COLLECTION_NAME}' cleared and recreated.")
            self._ensure_function_creation_tool_definition_in_store() # Re-add the essential one
        except Exception as e:
            self.debug_log(f"Error clearing collection (it might not have existed): {e}")
            # Attempt to create it anyway if deletion failed
            try:
                self.collection = self.chroma_client.get_or_create_collection(name=FUNCTION_COLLECTION_NAME)
                self._ensure_function_creation_tool_definition_in_store()
            except Exception as creation_error:
                 print(f"CRITICAL Error: Failed to get or create collection even after clear attempt: {creation_error}")
                 raise creation_error


    def _ensure_function_creation_tool_definition_in_store(self):
        func_name = FUNCTION_CREATION_TOOL_DEFINITION['function']['name']
        try:
            if not self.collection: 
                 self.debug_log(f"Collection is None in _ensure_function_creation_tool_definition_in_store. Re-initializing basic.")
                 self.collection = self.chroma_client.get_or_create_collection(name=FUNCTION_COLLECTION_NAME)

            existing_doc = self.collection.get(ids=[func_name])
            if existing_doc and existing_doc['ids'] and func_name in existing_doc['ids']:
                self.debug_log(f"'{func_name}' definition already in Chroma store.")
                return
        except Exception as e:
             self.debug_log(f"Note: Could not check for existing '{func_name}' in Chroma, will attempt to add. Error: {e}")

        self.debug_log(f"Adding '{func_name}' definition to Chroma store for discovery...")
        self.collection.add(
            ids=[func_name],
            embeddings=[self.embeddings_client.embed_query(f"{func_name}: {FUNCTION_CREATION_TOOL_DEFINITION['function']['description']}")],
            metadatas=[{
                'name': func_name,
                'description': FUNCTION_CREATION_TOOL_DEFINITION['function']['description'],
                'parameters_schema_json': json.dumps(FUNCTION_CREATION_TOOL_DEFINITION['function']['parameters']),
                'code_string': "# Executed locally by DynamicFunctionExecutor",
                'is_internal_special_function': True
            }],
            documents=[f"Definition for {func_name} (special internal tool)."]
        )

    def create_dynamic_function(self, new_function_name, new_function_description, new_function_parameters_schema, host_provided_api_description=None):
        self.debug_log(f"Attempting to create dynamic function: {new_function_name}")
        if not all([new_function_name, new_function_description, new_function_parameters_schema]):
            return "Error: Missing required arguments for function creation."
        if not new_function_name.isidentifier() or new_function_name in ["params", "external_apis", "json"]: # "json" added to reserved
            return f"Error: Invalid or reserved function name '{new_function_name}'."

        try:
            prompt = generate_function_creation_prompt(new_function_name, new_function_description, new_function_parameters_schema, host_provided_api_description)
        except ValueError as e:
            return f"Error: Invalid parameters_schema: {e}"
        except Exception as e:
            print(f"Error during prompt generation for {new_function_name}: {e}")
            return f"Error: Failed to generate prompt for {new_function_name}. {e}"

        generated_code_string = "" 
        try:
            self.debug_log(f"Calling LLM ({OPENAI_LLM_MODEL}) for function: {new_function_name}")
            response = self.openai_client.chat.completions.create(
                model=OPENAI_LLM_MODEL, messages=[{'role': 'user', 'content': prompt}], temperature=0.0
            )

            if response.choices and len(response.choices) > 0 and response.choices[0].message:
                generated_code_string = response.choices[0].message.content.strip()
            else:
                self.debug_log(f"LLM response problematic. Response: {response}")
                raise Exception("LLM response did not contain expected choices or message content.")

            if not generated_code_string: raise Exception("LLM returned empty code string.")

            sanitized_code = self._sanitize_generated_code(generated_code_string, new_function_name)
            self.debug_log(f"Sanitized code for {new_function_name}:\n{sanitized_code}")

            # Test compilation
            compile(sanitized_code, '<string>', 'exec')
            self.debug_log(f"Syntax validation passed for {new_function_name}.")

        except Exception as error:
            print(f"Error during LLM code generation/validation for {new_function_name}: {error}")
            return f"Error: Failed to generate/validate code for {new_function_name}. {error}"

        try:
            self.debug_log(f"Upserting function '{new_function_name}' definition to Chroma DB...")
            embedding = self.embeddings_client.embed_query(f"{new_function_name}: {new_function_description}")
            self.collection.upsert(
                ids=[new_function_name],
                embeddings=[embedding],
                metadatas=[{
                    'name': new_function_name,
                    'description': new_function_description,
                    'parameters_schema_json': json.dumps(new_function_parameters_schema),
                    'code_string': sanitized_code,
                    'is_internal_special_function': False
                }],
                documents=[f"Dynamic function: {new_function_name}. {new_function_description}"]
            )
            return f"Successfully created/updated dynamic function: {new_function_name}"
        except Exception as db_error:
            return f"Error: Failed to store function definition {new_function_name}. {db_error}"

    def _sanitize_generated_code(self, code_string, function_name):
        sanitized = code_string.replace('```python', '').replace('```', '').strip()
        # Try to extract only the function definition block if LLM includes extra text
        match = re.search(rf"^(def\s+{function_name}\s*\(\s*params\s*\):.*?)(?:\n\s*def\s|\n\n\n|\Z)", sanitized, re.DOTALL | re.MULTILINE)
        block_to_return = sanitized # Default to the whole sanitized string
        if match:
            block = match.group(1).strip()
            # Ensure the extracted block actually starts with the function definition
            if block.startswith(f"def {function_name}(params):"):
                block_to_return = block
            else: # If regex was too greedy or pattern is off, try a simpler find
                direct_def_start = f"def {function_name}(params):"
                start_index = sanitized.find(direct_def_start)
                if start_index != -1:
                    block_to_return = sanitized[start_index:] # Take from def onwards
        elif not sanitized.startswith(f"def {function_name}(params):"):
             self.debug_log(f"Warning: Sanitized code for {function_name} does not start with expected def statement. Using as-is. Code: {sanitized[:200]}...")

        return block_to_return.strip()


    def get_function_definition(self, function_name):
        try:
            results = self.collection.get(ids=[function_name], include=['metadatas', 'documents'])
            if not results or not results['ids'] or not results['metadatas'] or not results['metadatas'][0]: return None

            metadata = results['metadatas'][0]
            doc_content = results['documents'][0] if results['documents'] and results['documents'][0] else None

            return {
                'name': metadata.get('name'),
                'description': metadata.get('description') or doc_content,
                'parameters_schema': json.loads(metadata.get('parameters_schema_json', '{}')),
                'code_string': metadata.get('code_string'),
                'is_internal_special_function': metadata.get('is_internal_special_function', False)
            }
        except Exception as e:
            self.debug_log(f"Error getting function definition for '{function_name}': {e}")
            return None

    def get_available_function_definitions_for_llm_discovery(self, context_query, count=5):
        self.debug_log(f"Querying for function schemas relevant to: '{context_query}' (max {count})")
        try:
            available_defs = [{
                'name': FUNCTION_CREATION_TOOL_DEFINITION['function']['name'],
                'description': FUNCTION_CREATION_TOOL_DEFINITION['function']['description'],
                'parameters': FUNCTION_CREATION_TOOL_DEFINITION['function']['parameters'],
            }]
            # This part could be expanded to query Chroma for other dynamic functions if needed for more complex scenarios
            # For now, only the creation tool is explicitly returned for LLM discovery in this context.
            return available_defs
        except Exception as e:
            print(f"Error querying ChromaDB for available functions: {e}")
            return [FUNCTION_CREATION_TOOL_DEFINITION['function']] # Fallback

    def execute_dynamic_function(self, function_name, params_for_function, external_apis_dict=None):
        self.debug_log(f"Attempting to execute function: {function_name} with params: {json.dumps(params_for_function)[:200]}...")

        if function_name == FUNCTION_CREATION_TOOL_DEFINITION['function']['name']:
            self.debug_log(f"Executing special internal function: {function_name}")
            host_api_desc_for_new_func = params_for_function.pop('host_provided_api_description_for_new_func', None)
            try:
                required_args = ['new_function_name', 'new_function_description', 'new_function_parameters_schema']
                for arg in required_args:
                    if arg not in params_for_function:
                        return f"Error: Missing required parameter '{arg}' for {function_name}."
                return self.create_dynamic_function(
                    params_for_function['new_function_name'],
                    params_for_function['new_function_description'],
                    params_for_function['new_function_parameters_schema'],
                    host_api_desc_for_new_func
                )
            except KeyError as e: return f"Error: Missing required parameter for {function_name}: {e}"
            except Exception as error: return f"Error: Failed to execute {function_name}. {error}"

        func_def = self.get_function_definition(function_name)
        if not func_def: return f"Error: Function '{function_name}' not found."
        if func_def.get('is_internal_special_function'):
            return f"Error: Cannot directly execute special internal function '{function_name}' this way."

        code_string = func_def.get('code_string')
        schema = func_def.get('parameters_schema')
        if not code_string: return f"Error: Invalid or missing code string for function '{function_name}'."

        try:
            # Prepare the parameters that will be passed to the dynamic function.
            # This is a copy so we can modify it for specific cases if needed.
            params_for_actual_call = params_for_function.copy()
            if function_name == "df_genesis_engine":
                # Ensure external_apis is available inside the 'params' argument for df_genesis_engine
                # because its LLM-generated code expects to retrieve it from there.
                params_for_actual_call['external_apis'] = external_apis_dict if external_apis_dict is not None else {}
            if schema and schema.get('required'):
                for param_key in schema['required']:
                    if param_key not in params_for_function:
                        raise ValueError(f"Missing required parameter '{param_key}' in params_for_function for {function_name}.")

            # Prepare execution scope
            # json module is explicitly added for parsing API responses if they are JSON strings
            allowed_builtins = {'print':print,'len':len,'str':str,'int':int,'float':float,'bool':bool,'dict':dict,'list':list,'tuple':set,'set':set,'isinstance':isinstance,'Exception':Exception,'ValueError':ValueError,'TypeError':TypeError,'KeyError':KeyError, 'json': json}
            execution_globals = {
                'params': params_for_actual_call, # Use the potentially modified params for the exec scope 'params' variable
                'external_apis': external_apis_dict if external_apis_dict is not None else {},
                '__builtins__': allowed_builtins,
                'json': json # Make json module directly available
            }

            compiled_code = compile(code_string, f'<function:{function_name}>', 'exec')
            exec(compiled_code, execution_globals) # Defines the function in execution_globals

            actual_function_to_call = execution_globals.get(function_name)
            if not callable(actual_function_to_call):
                 raise Exception(f"Code string did not define a callable function named '{function_name}'. Defined names: {list(execution_globals.keys())}")

            self.debug_log(f"Executing dynamically loaded function {function_name} from string...")
            result = actual_function_to_call(params_for_actual_call) # Call the function with its designated params

            if not isinstance(result, str): # Ensure result is a string as expected by game logic
                self.debug_log(f"Warning: Dynamic function {function_name} did not return a string. Converting: {result}")
                result = str(result)
            return result
        except Exception as exec_error:
            error_message_for_client = f"Error: Failed to execute dynamic function {function_name}. Details: {exec_error}"
            print(f"Error executing dynamic function '{function_name}': {exec_error}\nTraceback:\n{traceback.format_exc()}")

            if self.llm_repair_callback and callable(self.llm_repair_callback):
                self.debug_log(f"LLM REPAIR HOOK: Invoking callback for function '{function_name}' due to error.")
                try:
                    api_desc_for_repair = None 
                    if hasattr(self, 'get_host_api_description_for_repair_context'): # If a method is set to provide this
                        api_desc_for_repair = self.get_host_api_description_for_repair_context()
                    
                    self.llm_repair_callback(
                        function_name=function_name,
                        error=exec_error,
                        traceback_str=traceback.format_exc(),
                        params_for_function=params_for_function,
                        host_api_description_for_repair_context=api_desc_for_repair 
                    )
                except Exception as repair_hook_error:
                    print(f"CRITICAL: Error in LLM repair hook callback invocation itself: {repair_hook_error}\n{traceback.format_exc()}")
            else:
                self.debug_log(f"LLM REPAIR HOOK: No callback registered or not callable for function '{function_name}'.")
            
            return error_message_for_client
