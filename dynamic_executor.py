import os
import json
import re
import chromadb
import openai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from urllib.parse import urlparse
import traceback # Added for LLM repair hook

import uuid

load_dotenv()

# Configuration from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_BASE_URL = os.environ.get("OPENAI_API_BASE_URL") # Using this for the client, but model is Gemini
OPENAI_LLM_MODEL = os.environ.get("OPENAI_LLM_MODEL") # This should be your Gemini model name via OpenAI compatible endpoint
LANGCHAIN_EMBEDDING_MODEL = os.environ.get("LANGCHAIN_EMBEDDING_MODEL")
CHROMA_URL = os.environ.get("CHROMA_URL")

FUNCTION_COLLECTION_NAME = "dynamic_functions" # Stores definitions of LLM-generated functions
MAX_SYNTAX_REPAIR_RETRIES = 3 # Increased retries slightly

if not GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY not set")
if not OPENAI_API_BASE_URL: raise ValueError("OPENAI_API_BASE_URL set") # Corrected check
if not OPENAI_LLM_MODEL: raise ValueError("OPENAI_LLM_MODEL not set")
if not LANGCHAIN_EMBEDDING_MODEL: raise ValueError("LANGCHAIN_EMBEDDING_MODEL not set")
if not CHROMA_URL: raise ValueError("CHROMA_URL not set")

# --- Definition for the Function Creation Tool (used by the executor itself) ---
FUNCTION_CREATION_TOOL_DEFINITION = {
    'type': 'function',
    'function': {
        'name': 'create_dynamic_function',
        'description': 'Define and create a new Python function string that can be executed later. Takes the desired function name, a high-level description of its purpose, and a parameter schema for the function itself. The generated function code will be executed in an environment where it can call external functions via an \'external_apis\' dictionary, whose available functions are described by the host if provided. The generated function code should NOT include any try/except blocks, as the executor handles top-level error catching.',
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


def generate_function_creation_prompt(name, description, parameters_schema, host_provided_api_description=None, is_repair=False, previous_code=None, error_message=None):
    """Generates the prompt for the LLM to create or repair a Python function string."""
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
    - The `external_apis` dictionary, containing callable host functions, is directly available as a global-like variable within your function's execution scope. You do **not** need to access it from the `params` dictionary (e.g., do not use `params['external_apis']`).
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
    # It might also contain additional contextual data from the host.
    #
    # NECESSARY IMPORTS: If you use modules like `uuid` for `uuid.uuid4()`,
    # you MUST include `import uuid` at the START of this function body.
    # Similarly for `json` (`import json`) if needed for complex JSON manipulation.
    # The executor handles top-level error catching, so DO NOT include try/except blocks.

    # Example of accessing params and calling a host API:
    #   import uuid # Example import
    #   import json # Example import for json.loads if needed
    #   required_data = params['my_required_param']
    #   optional_data = params.get('my_optional_param', 'default_value')
    #   unique_id_for_something = str(uuid.uuid4())
    #
    #   # Note: 'external_apis' is used directly
    #   api_args = {{'data_for_api': required_data, 'context': params.get('soul_id'), 'id': unique_id_for_something}}
    #   host_result_str = external_apis['host_api_example'](api_args)
    #   # If API returns JSON string:
    #   #   host_data = json.loads(host_result_str)
    #   #   return f"Host API said: {{host_data.get('message', host_result_str)}}"
    #   # Any json.JSONDecodeError will be caught by the executor.
    #   return f"Host API result: {{host_result_str}}"
    #
    # Replace the above example with your actual logic.
    # Ensure all paths return a string.
    return f"Function '{function_name}' executed. Input params: {{str(params)}}. Implement your logic here."
"""
    example_code_structure = example_code_structure_template.format(function_name=name)

    if is_repair:
        repair_specific_guidance = ""
        # Removed specific try/except error messages as they are no longer expected
        repair_specific_guidance = f"The Python code you previously generated for function '{name}' had an error: {error_message}"

        return f"""You are an expert Python function generator.
{repair_specific_guidance}

The faulty code was:
```python
{previous_code}
```

Please correct this error and provide the complete, valid Python function code again.
**CRITICAL Instructions for Python Code Generation (Reminder):**
1.  Write a single Python function named precisely `{name}`.
2.  The function MUST accept a single argument: a dictionary named `params`.
3.  The executor handles top-level error catching, so **DO NOT include any `try` or `except` blocks** in your generated code.
4.  If you use modules like `uuid` (e.g., for `uuid.uuid4()`) or `json` (for `json.loads()`), you MUST include the `import uuid` or `import json` statement at the beginning of the function body.
5.  Access the `external_apis` dictionary directly (it's available in the function's scope). Do NOT use `params['external_apis']`.
6.  Do NOT include any comments, explanations, or surrounding text outside the function definition itself. Output only the `def {name}(params): ...` block.
7.  Do NOT include markdown code block markers (e.g., ```python or ```) in your output.
8.  The function should perform the action described in its original high-level description: {description}
9.  Original Parameters Schema (for the 'params' argument):
    ```json
    {json.dumps(parameters_schema, indent=2)}
    ```
10. Available Host APIs (in `external_apis` dictionary) if needed:
    ```
    {host_provided_api_description if host_provided_api_description else "No specific host APIs were described for the original task."}
    ```
11. If a host API is documented to return a JSON string, you MUST use `json.loads(result_string)` to parse it. Any errors during this, including `json.JSONDecodeError`, will be caught by the executor.
12. Ensure all Python syntax, especially indentation, is flawless. All paths should return a string.

**Your Task:**
Generate *only* the corrected Python function code for `{name}`.
"""

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
    5.  **PYTHON INDENTATION IS PARAMOUNT!** Ensure all logic is correctly indented.
    6.  The executor handles top-level error catching, so **DO NOT include any `try` or `except` blocks** in your generated code.
    7.  If you use modules like `uuid` (e.g., for `uuid.uuid4()`) or `json` (for `json.loads()`), you MUST include the `import uuid` or `import json` statement at the beginning of the function body.
    8.  Access the `external_apis` dictionary directly (it's available in the function's scope). Do NOT use `params['external_apis']`.
    9.  Do NOT include any comments, explanations, or surrounding text outside the function definition itself. Output only the `def {name}(params): ...` block.
    10. Do NOT include markdown code block markers (e.g., ```python or ```) in your output.
    11. **Remember**: If a host API is documented to return a JSON string, you MUST use `json.loads(result_string)` to parse it into a Python dictionary or list before accessing its elements. Any errors during this, including `json.JSONDecodeError`, will be caught by the executor.

    **Schema Parameter Access Examples (from `params` dictionary):**
    {param_access_section}

    **Example Function Structure (Pay ATTENTION to INDENTATION, IMPORTS, direct `external_apis` access, and the ABSENCE of try-except blocks):**
    ```python
{example_code_structure.strip()}
    ```

    **Your Task:**
    Generate *only* the Python function code for `{name}` based *exactly* on the specification provided above. Ensure all Python syntax, especially indentation, is flawless. All paths should return a string.
    """


class DynamicFunctionExecutor:
    def __init__(self):
        # Ensure GEMINI_API_KEY is used for the OpenAI client if that's the intended setup for compatibility layer
        self.openai_client = openai.OpenAI(base_url=OPENAI_API_BASE_URL, api_key=GEMINI_API_KEY, max_retries=5)
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
        if not new_function_name.isidentifier() or new_function_name in ["params", "external_apis", "json", "uuid"]: # Added json and uuid as reserved contexts
            return f"Error: Invalid or reserved function name '{new_function_name}'."

        generated_code_string = ""
        last_error = None
        sanitized_code = ""


        for attempt in range(MAX_SYNTAX_REPAIR_RETRIES + 1):
            try:
                if attempt == 0: # First attempt
                    prompt = generate_function_creation_prompt(
                        new_function_name, new_function_description, new_function_parameters_schema,
                        host_provided_api_description
                    )
                else: # Repair attempt
                    self.debug_log(f"Syntax repair attempt {attempt} for {new_function_name}.")
                    prompt = generate_function_creation_prompt(
                        new_function_name, new_function_description, new_function_parameters_schema,
                        host_provided_api_description,
                        is_repair=True, previous_code=generated_code_string, error_message=str(last_error)
                    )

                self.debug_log(f"Calling LLM ({OPENAI_LLM_MODEL}) for function: {new_function_name} (Attempt: {attempt+1})")
                response = self.openai_client.chat.completions.create(
                    model=OPENAI_LLM_MODEL, messages=[{'role': 'user', 'content': prompt}], temperature=0.0
                )

                if response.choices and len(response.choices) > 0 and response.choices[0].message:
                    generated_code_string = response.choices[0].message.content.strip()
                else:
                    self.debug_log(f"LLM response problematic. Response: {response}")
                    last_error = Exception("LLM response did not contain expected choices or message content.")
                    continue # Retry if possible

                if not generated_code_string:
                    last_error = Exception("LLM returned empty code string.")
                    continue # Retry if possible

                sanitized_code = self._sanitize_generated_code(generated_code_string, new_function_name)
                self.debug_log(f"Sanitized code for {new_function_name} (Attempt {attempt+1}):\n{sanitized_code}")

                compile(sanitized_code, '<string>', 'exec') # Test compilation
                self.debug_log(f"Syntax validation passed for {new_function_name} on attempt {attempt+1}.")
                last_error = None # Clear last error if compilation succeeded
                break # Success, exit retry loop

            except SyntaxError as se:
                self.debug_log(f"SyntaxError during LLM code generation/validation for {new_function_name} (Attempt {attempt+1}): {se}")
                last_error = se
            except Exception as e:
                self.debug_log(f"Non-SyntaxError during LLM code generation/validation for {new_function_name} (Attempt {attempt+1}): {e}")
                last_error = e
                break # Don't retry for non-syntax errors during generation call itself

        if last_error: # If all retries failed or a non-syntax error occurred
            error_message = f"Error: Failed to generate/validate code for {new_function_name} after {attempt + 1} attempt(s). Last error: {last_error}"
            print(error_message)
            return error_message

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
                    'code_string': sanitized_code, # Use the successfully compiled code
                    'is_internal_special_function': False
                }],
                documents=[f"Dynamic function: {new_function_name}. {new_function_description}"]
            )
            return f"Successfully created/updated dynamic function: {new_function_name}"
        except Exception as db_error:
            return f"Error: Failed to store function definition {new_function_name}. {db_error}"


    def _sanitize_generated_code(self, code_string, function_name):
        sanitized = code_string.replace('```python', '').replace('```', '').strip()
        # Try to extract only the function definition, even if there's extra text.
        # This regex looks for `def function_name(params):` and captures everything until
        # what looks like the start of another function def or too many blank lines.
        match = re.search(rf"^(def\s+{function_name}\s*\(\s*params\s*\):.*?)(?:\n\s*(?:def\s|@|\#\#\#|\s*\n\s*\n)|# --- End of function ---|\Z)", sanitized, re.DOTALL | re.MULTILINE)

        block_to_return = sanitized # Fallback to the whole sanitized string
        if match:
            block = match.group(1).strip()
            # Ensure the extracted block actually starts with the function definition
            if block.startswith(f"def {function_name}(params):"):
                block_to_return = block
            else: # If the regex grabbed something but it's not the start, try a simpler find
                direct_def_start = f"def {function_name}(params):"
                start_index = sanitized.find(direct_def_start)
                if start_index != -1:
                    # This is a bit naive, assumes the function is the last thing or only thing.
                    block_to_return = sanitized[start_index:]
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
            # This is simplified for now, always returning the creation tool.
            # A real implementation might query ChromaDB here based on context_query.
            available_defs = [{
                'name': FUNCTION_CREATION_TOOL_DEFINITION['function']['name'],
                'description': FUNCTION_CREATION_TOOL_DEFINITION['function']['description'],
                'parameters': FUNCTION_CREATION_TOOL_DEFINITION['function']['parameters'],
            }]
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
            params_for_actual_call = params_for_function.copy()
            # For df_genesis_engine, params_for_function is empty, but the function itself might not declare 'params' if it's not used.
            # It still needs external_apis though.
            if function_name == "df_genesis_engine":
                # df_genesis_engine doesn't take 'params' in its definition for the LLM,
                # but the exec environment needs it.
                pass


            if schema and schema.get('required'):
                for param_key in schema['required']:
                    if param_key not in params_for_function:
                        # Check if the parameter is actually expected by the function's generated code
                        # This is tricky without parsing the Python AST. For now, rely on the schema.
                        # If a function (like df_genesis_engine) has an empty schema but gets params, it's okay.
                        # If it has a required param in schema but not provided, it's an error.
                        if schema.get('properties', {}).get(param_key) is not None: # Check if param is actually in schema properties
                             raise ValueError(f"Missing required parameter '{param_key}' in params_for_function for {function_name}.")


            # Standard library modules available to the dynamic functions
            # These are imported within the generated function code if needed.
            allowed_builtins = {
                'print':print,'len':len,'str':str,'int':int,'float':float,'bool':bool,
                'dict':dict,'list':list,'tuple':tuple,'set':set, # Corrected tuple to be the type, not a new set
                'isinstance':isinstance,
                'Exception':Exception,'ValueError':ValueError,'TypeError':TypeError,'KeyError':KeyError,
                '__import__': __import__, 'globals': globals
                # 'json' and 'uuid' are provided in the execution_globals directly
            }
            execution_globals = {
                # 'params': params_for_actual_call, # This is passed to the function directly
                'external_apis': external_apis_dict if external_apis_dict is not None else {},
                '__builtins__': allowed_builtins,
                'json': json, # Make json module available
                'uuid': uuid  # Make uuid module available
            }

            # The generated code should be a complete function definition.
            # We exec it to define the function in our execution_globals.
            compiled_code = compile(code_string, f'<function:{function_name}>', 'exec')
            exec(compiled_code, execution_globals)

            actual_function_to_call = execution_globals.get(function_name)
            if not callable(actual_function_to_call):
                 raise Exception(f"Code string did not define a callable function named '{function_name}'. Defined names: {list(execution_globals.keys())}")

            self.debug_log(f"Executing dynamically loaded function {function_name} from string...")
            # The dynamically defined function will take 'params' as its argument.
            result = actual_function_to_call(params_for_actual_call)


            if not isinstance(result, str):
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
                    # This part is a bit conceptual, assuming the main server can provide its API list for repair context
                    # if hasattr(self, 'get_host_api_description_for_repair_context'):
                    #     api_desc_for_repair = self.get_host_api_description_for_repair_context()

                    self.llm_repair_callback(
                        function_name=function_name,
                        error=exec_error,
                        traceback_str=traceback.format_exc(),
                        params_for_function=params_for_function, # Pass the original params
                        host_api_description_for_repair_context=api_desc_for_repair
                    )
                except Exception as repair_hook_error:
                    print(f"CRITICAL: Error in LLM repair hook callback invocation itself: {repair_hook_error}\n{traceback.format_exc()}")
            else:
                self.debug_log(f"LLM REPAIR HOOK: No callback registered or not callable for function '{function_name}'.")

            return error_message_for_client

