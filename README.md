# dynamic_executor -- Orb of Ingenuity Demo

[![dynamic_executor -- Orb of Ingenuity Demo](https://img.youtube.com/vi/T1sTRsDjW24/0.jpg)](https://www.youtube.com/watch?v=T1sTRsDjW24) 

![dynamic_executor -- Orb of Ingenuity Demo](https://github.com/neuroidss/dynamic_executor/blob/main/Screenshot%20from%202025-05-16%2009-28-22.png?raw=true)

## Overview

The Orb of Ingenuity Demo is a dynamic, LLM-driven text-based adventure game. It showcases how Large Language Models (LLMs) can be utilized not just for content generation, but for creating and modifying executable game logic and player abilities (tools/artifacts) at runtime. The core of the project is a system where the game world, its rules, puzzles, and even the behaviors of in-game items are defined and can be expanded by LLM-generated Python code.

Players can interact with the environment using artifacts. A special artifact, the "Orb of Ingenuity," allows players to describe a new tool or ability they wish to create. The system then uses an LLM to translate this natural language description into a new, executable Python function, which is integrated into the game as a new artifact for the player to use.

## Core Features

*   **LLM-Generated Game World:** The initial game environment, including locations, interactive objects, puzzles, and basic artifacts, is created by an LLM-powered "Genesis Engine" (a dynamic function itself).
*   **Dynamic Function Creation & Execution:** LLMs generate Python code for new game mechanics, such as artifact behaviors or puzzle responses. These functions are stored and can be executed by the game server.
*   **Player-Driven Tool Creation:** Through the "Orb of Ingenuity," players can describe new abilities. The LLM generates the underlying Python function and a corresponding in-game artifact.
*   **Tool Use:** Players interact with the game by using artifacts. Each artifact's use triggers its linked dynamic Python function, affecting the game state.
*   **Persistent Game State:** The game world, including dynamically created artifacts and functions, can be saved and loaded.
*   **Real-time Interaction:** Player actions and world updates are communicated in real-time using WebSockets (Flask-SocketIO).
*   **Automated Demo Mode:** A `demo_script.json` allows for an automated playthrough, showcasing the dynamic tool creation and use loop.
*   **Host API System:** LLM-generated functions interact with the game world through a secure set of predefined "Host APIs," ensuring controlled modifications to the game state.

## How It Works

The system comprises a simple web-based frontend, a Python backend server, and a dynamic function execution engine that interfaces with an LLM and a vector database.

**1. Frontend (Client - `index.html`):**
    *   A basic HTML/JavaScript interface.
    *   Communicates with the backend via WebSockets.
    *   Displays game state (location, inventory, world log).
    *   Allows players to select and use artifacts from their inventory.
    *   Prompts for user input when an artifact (like the Orb of Ingenuity) requires it.

**2. Backend (Server - `server.py`):**
    *   Built with Flask and Flask-SocketIO for real-time, event-driven communication.
    *   Manages the central `game_state` (locations, souls, artifacts, etc.).
    *   Handles player connections, disconnections, and actions.
    *   Implements a `GamePrimitiveHandler` which exposes a set of "Host APIs." These are Python functions that LLM-generated code can call to safely interact with and modify the game world (e.g., `host_core_add_location_to_gamestate`, `host_create_temporary_object`, `host_apply_effect_on_environment_object`).
    *   Orchestrates the demo script execution if enabled.
    *   Interfaces with the `DynamicFunctionExecutor` for creating and running LLM-generated code.

**3. Dynamic Function Executor (`dynamic_executor.py`):**
    *   The core of the LLM-driven dynamic code capabilities.
    *   `create_dynamic_function()`:
        *   Takes a function name, a natural language description of its purpose, and a JSON schema for its parameters.
        *   Constructs a detailed prompt for an LLM (Gemini, via an OpenAI-compatible API). This prompt instructs the LLM to generate a Python function string based on the description, making use of the available Host APIs.
        *   The generated Python code string is sanitized and validated (syntax check).
        *   The function's metadata (name, description, parameter schema) and the code string are stored in a ChromaDB vector database, embedded for semantic search.
    *   `execute_dynamic_function()`:
        *   Retrieves a function's definition and code string from ChromaDB by its name.
        *   Executes the Python code in a restricted environment.
        *   The `external_apis` dictionary (containing the Host APIs from `server.py`) and any necessary `params` (like `soul_id`, `artifact_properties`, user-provided arguments) are injected into the execution scope of the dynamic function.
        *   The function's return value (expected to be a string, often a JSON string from a Host API) is passed back to the server.
    *   `FUNCTION_CREATION_TOOL_DEFINITION`: A special, internally defined tool that allows the LLM to request the creation of new functions.

**4. LLM Interaction & Data Flow (Example: Orb of Ingenuity Tool Creation):**

    *   **Initial World Setup:**
        1.  `initial_prompt.json` contains a series of commands. First, it instructs the `DynamicFunctionExecutor` (via its `create_dynamic_function` tool) to generate several core dynamic functions, including `df_genesis_engine`, `df_interact_with_pedestal`, and `df_initiate_orb_tool_creation`.
        2.  Once `df_genesis_engine` is created, it's executed. This function makes a series of calls to `host_core_*` APIs to populate the initial game state (locations, environment objects, artifacts like the "Orb of Ingenuity").

    *   **Player Creates a New Tool (e.g., "Light Bridge"):**
        1.  **Player Action:** The player uses the "Orb of Ingenuity" artifact.
        2.  **Server:** This triggers its linked dynamic function, `df_initiate_orb_tool_creation`. This simple function returns the specific string "EVENT:PROMPT_USER_FOR_TOOL_DESCRIPTION".
        3.  **Client:** Receives this event string and prompts the player to describe the tool they want to create (e.g., "Create a temporary light bridge to the keyhole platform").
        4.  **Player Input:** The player submits their description.
        5.  **Server (`_internal_submit_tool_description_logic`):**
            *   Receives the description and the ID of the catalyst artifact (the Orb).
            *   Prepares arguments for calling `dynamic_executor.execute_dynamic_function` with the `create_dynamic_function` tool name. These arguments include:
                *   A new unique function name (e.g., `df_user_3c97_fcae`).
                *   A refined description for the LLM, incorporating the player's request and instructions to use specific Host APIs (e.g., `host_create_temporary_object` for a light bridge).
                *   A schema for the new function's parameters (if any).
                *   A description of all available Host APIs (generated by `generate_api_description_for_llm_prompt`).
        6.  **`DynamicFunctionExecutor` (`create_dynamic_function` flow):**
            *   Sends the composed prompt to the LLM.
            *   The LLM generates the Python code string for the new function (e.g., code that calls `external_apis['host_create_temporary_object'](...)` with appropriate arguments for a light bridge).
            *   The new function's code and metadata are stored in ChromaDB.
        7.  **Server:**
            *   If function creation was successful, the server creates a new artifact in the game state (e.g., "Orb: Create a temporary light..."). This new artifact's `linked_dynamic_function_name` is set to the name of the LLM-generated function (e.g., `df_user_3c97_fcae`).
            *   The new artifact is added to the player's inventory.
        8.  **Player Use:** The player can now see the new "Orb: Create a temporary light..." artifact in their inventory and use it. Using it will execute the newly LLM-generated `df_user_3c97_fcae` function, which in turn calls the `host_create_temporary_object` API to make the light bridge appear in the game.

## Technologies Used

*   **Backend:** Python 3
    *   Flask: Web framework.
    *   Flask-SocketIO: Real-time WebSocket communication.
    *   eventlet: Concurrent networking library for SocketIO.
*   **LLM Interaction:**
    *   OpenAI Python client (configured to use a Gemini model via an OpenAI-compatible API endpoint).
    *   Langchain Google Generative AI Embeddings (`langchain-google-genai`): For generating embeddings for ChromaDB.
*   **Vector Database:** ChromaDB: To store and retrieve dynamic function definitions and their Python code.
*   **Frontend:** HTML, CSS, JavaScript (vanilla).
*   **Data Serialization:** JSON (for game state, initial prompts, demo scripts, API communication).
*   **Environment Management:** `python-dotenv` for managing API keys and configuration.

## Setup and Running

1.  **Prerequisites:**
    *   Python 3.8+
    *   Access to a Gemini LLM via an OpenAI-compatible API endpoint.
    *   A running ChromaDB instance.

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/neuroidss/dynamic_executor
    cd dynamic_executor
    ```

3.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the project root with the following (adjust values as necessary):
    ```env
    GEMINI_API_KEY="your_gemini_api_key"
    OPENAI_API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"
    OPENAI_LLM_MODEL="gemini-2.5-flash-preview-04-17" # Or your specific Gemini model
    LANGCHAIN_EMBEDDING_MODEL="models/embedding-001" # Or your preferred embedding model
    CHROMA_URL="http://localhost:8000" # URL of your ChromaDB instance
    PORT="3001"
    ```
    *Ensure your `OPENAI_API_BASE_URL` and `OPENAI_LLM_MODEL` are correctly set up if you are proxying Gemini through an OpenAI-like interface, or adjust `dynamic_executor.py` if using Gemini's native SDK.*

5.  **Run the Server:**
    *   To start fresh (clearing previous save and dynamic functions) and run the demo:
        ```bash
        python server.py --recreate --demo demo_script.json
        ```
    *   To load existing state (if `game_save.json` exists) and run normally:
        ```bash
        python server.py
        ```
    *   The `run.sh` script in the `run.txt` log is an example: `./run.sh --recreate --demo demo_script.json`

6.  **Access the Game:**
    Open your web browser and navigate to `http://localhost:3001` (or the configured port).

## Key Files

*   `server.py`: The main Flask-SocketIO application, game logic, Host APIs, and demo orchestration.
*   `dynamic_executor.py`: Handles LLM communication for generating Python functions, interacts with ChromaDB, and executes the dynamic code.
*   `initial_prompt.json`: Defines the instructions for the LLM to create the "Genesis Engine" and other foundational dynamic functions, which in turn build the initial game world.
*   `demo_script.json`: A JSON-based script that automates a sequence of player actions to demonstrate the game's features, especially the Orb of Ingenuity.
*   `index.html`: The client-side user interface.
*   `game_save.json`: Stores the persistent state of the game, including dynamically generated artifacts.
*   `requirements.txt`: Python dependencies.
*   `.env` (user-created): For storing sensitive API keys and configurations.

## Project Structure

```
.
├── dynamic_executor.py       # Handles dynamic function creation and execution via LLM
├── server.py                 # Main Flask server, game logic, SocketIO events
├── initial_prompt.json       # Prompts/commands for initial world and function generation
├── demo_script.json          # Script for automated demo playthrough
├── game_save.json            # Persistent game state (created after first run)
├── index.html                # Frontend UI for the game
├── requirements.txt          # Python dependencies
├── run.txt                   # Example console output from a demo run
└── README.md                 # This file
```

## Future Enhancements / Ideas

*   More sophisticated sandboxing for dynamic code execution.
*   Allowing LLM-generated functions to call other LLM-generated functions.
*   More complex Host APIs for richer game interactions.
*   UI for visualizing and managing dynamic functions.
*   Player-to-player interaction with dynamically created tools.
*   LLM-assisted debugging or repair of generated functions.
