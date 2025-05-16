import eventlet
eventlet.monkey_patch()

import os
import json
import uuid
import time
from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, emit, disconnect
from dotenv import load_dotenv
import random
import re
import argparse # For command-line arguments
import signal # For graceful shutdown
import sys # For exiting
import traceback # For LLM repair

load_dotenv()
from dynamic_executor import DynamicFunctionExecutor

PORT = int(os.environ.get("PORT", 3001))
INITIAL_PROMPT_FILE = 'initial_prompt.json'
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), 'public')
SAVE_FILE = "game_save.json" # For persistence

game_state = {
    'souls': {},
    'locations': {},
    'artifacts': {},
    'environment_objects': {},
    'puzzle_states': {},
    'temporary_objects': {},
    'world_log': ["Welcome to the Orb of Ingenuity Demo!"],
    'server_metadata': {'next_player_number': 1}
}
connected_souls_by_sid = {} # sid: soul_id

demo_context = {
    "is_active": False,
    "script_path": None,
    "script_data": [],
    "current_step": 0,
    "player_soul_id": None,
    "player_sid": None,
    "orb_catalyst_artifact_id_pending": None,
    "initial_prompt_processed": False # New flag for demo mode
}

app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path='')
socketio = SocketIO(app, cors_allowed_origins="*")

dynamic_executor = DynamicFunctionExecutor()
game_api_registry = {} # For LLM-callable host APIs
game_primitives_handler = None # Will be initialized in main()

class GamePrimitiveHandler: # Renamed to reflect it handles host APIs now primarily
    def __init__(self, gs_ref, sio_ref):
        self.gs = gs_ref # gs_ref is a function that returns the current game_state
        self.sio = sio_ref

    def _get_gs(self): # Helper to always get the current game_state
        return self.gs()

    # --- Host Core APIs (for LLM-generated Genesis Engine) ---
    def host_core_add_location_to_gamestate(self, params):
        gs = self._get_gs()
        loc_id = params.get('id')
        name = params.get('name', loc_id)
        description = params.get('description', "An undescribed place.")
        landmarks = params.get('landmarks', {})
        if not loc_id: return "Error API (host_core_add_location_to_gamestate): 'id' is required."
        if loc_id in gs['locations']: return f"Error API (host_core_add_location_to_gamestate): Location ID '{loc_id}' already exists."
        gs['locations'][loc_id] = {'id': loc_id, 'name': name, 'description': description, 'exits': {}, 'landmarks': landmarks}
        # log_to_world implicitly called by genesis engine via host_log_message_to_world
        return f"Host API: Location '{name}' ({loc_id}) created."

    def host_core_add_artifact_to_gamestate(self, params):
        gs = self._get_gs()
        art_id = params.get('id', str(uuid.uuid4()))
        name = params.get('name', "Mysterious Artifact")
        description = params.get('description', "An artifact of unknown purpose.")
        properties = params.get('properties', {})
        linked_func_name = params.get('linked_dynamic_function_name')

        if art_id in gs['artifacts']: return f"Error API (host_core_add_artifact_to_gamestate): Artifact ID '{art_id}' already exists."
        gs['artifacts'][art_id] = {
            'id': art_id, 'name': name, 'description': description,
            'linked_dynamic_function_name': linked_func_name,
            'properties': properties
        }
        return f"Host API: Artifact '{name}' ({art_id}) created."

    def host_core_add_env_object_to_gamestate(self, params):
        gs = self._get_gs()
        obj_id = params.get('id')
        loc_id = params.get('location_id')
        obj_type = params.get('type', 'generic_object')
        details = params.get('details', {})
        if not obj_id or not loc_id: return "Error API (host_core_add_env_object_to_gamestate): 'id' and 'location_id' are required."
        if loc_id not in gs['locations']: return f"Error API (host_core_add_env_object_to_gamestate): Location '{loc_id}' not found."
        gs['environment_objects'][obj_id] = {'id': obj_id, 'location_id': loc_id, 'type': obj_type, 'details': details}
        return f"Host API: Environment object '{obj_id}' ({obj_type}) created in {loc_id}."

    def host_core_initialize_puzzle_state(self, params):
        gs = self._get_gs()
        puzzle_id = params.get('id')
        initial_state = params.get('initial_state', {})
        if not puzzle_id: return "Error API (host_core_initialize_puzzle_state): 'id' is required."
        gs['puzzle_states'][puzzle_id] = {'id': puzzle_id, **initial_state}
        return f"Host API: Puzzle '{puzzle_id}' initialized."

    def host_log_message_to_world(self, params): # New API for Genesis Engine logging
        message = params.get("message", "Untagged message from host API.")
        # The prefix is now handled by the LLM in its message string as per initial_prompt
        log_to_world(f"{message}") # Directly log the message
        return "Host API: Message logged to world."

    # --- Existing Host APIs (for player-interactive dynamic functions) ---
    def api_apply_effect_on_environment_object(self, params):
        gs = self._get_gs()
        obj_id = params.get('object_id'); effect_to_apply = params.get('effect_details')
        if not obj_id or obj_id not in gs['environment_objects']: return json.dumps({"error": "Invalid object_id."})
        if not effect_to_apply or not isinstance(effect_to_apply, dict): return json.dumps({"error": "Invalid effect_details."})
        env_obj = gs['environment_objects'][obj_id]
        for key, value in effect_to_apply.items(): env_obj['details'][key] = value
        log_to_world(f"Effect applied to env object '{obj_id}': {json.dumps(effect_to_apply)}")
        return json.dumps({"message": f"Effect applied to '{obj_id}'."})


    def api_check_puzzle_condition(self, params):
        gs = self._get_gs()
        puzzle_id = params.get('puzzle_id')
        if puzzle_id == "elemental_trial":
            active_count = 0
            for obj_id, obj_data in gs['environment_objects'].items():
                if obj_data['location_id'] == 'trial_chamber' and obj_data['type'] == 'elemental_pedestal' and obj_data['details'].get('is_active'):
                    active_count +=1
            required_active = gs['puzzle_states'].get(puzzle_id, {}).get('target_pedestals', 3)
            if active_count >= required_active:
                return json.dumps({'condition_met': True, 'message': 'All elemental pedestals are active!'})
            else:
                return json.dumps({'condition_met': False, 'message': f'{active_count}/{required_active} pedestals active.'})
        elif puzzle_id == "vault_access_puzzle":
            bridge_exists = any(
                obj_data['type'] == 'light_bridge' and obj_data['location_id'] == 'vault_approach' and
                obj_data['to'] == 'keyhole_platform_exit' and time.time() <= obj_data['creation_time'] + obj_data['duration']
                for obj_id, obj_data in gs['temporary_objects'].items()
            )
            if bridge_exists:
                return json.dumps({'condition_met': True, 'message': 'A way to the keyhole is clear!'})
            else:
                return json.dumps({'condition_met': False, 'message': 'The chasm blocks the way to the keyhole.'})
        return json.dumps({'condition_met': False, 'message': f'Unknown puzzle or condition for {puzzle_id}.'})

    def api_trigger_world_event(self, params):
        gs = self._get_gs()
        event_id = params.get('event_id'); acting_soul_id = params.get('soul_id')
        if event_id == "elemental_trial_success":
            log_to_world("The Elemental Trial is complete! The sealed door in the Trial Chamber rumbles open.")
            # Orb is already created by genesis, no need to give it here.
            # Logic to open door:
            if 'trial_chamber' in gs['locations'] and 'sealed_door_exit' in gs['locations']['trial_chamber'].get('landmarks',{}):
                 gs['locations']['trial_chamber']['description'] += " The once sealed stone door now stands open."
                 # Could add an actual exit here if desired: gs['locations']['trial_chamber']['exits']['north'] = 'vault_approach' # Example
            return json.dumps({"message": "Elemental Trial complete! A hidden door opens."})
        if event_id == "open_vault_door":
            log_to_world("The Unreachable Vault door rumbles open!")
            if 'unreachable_vault' in gs['puzzle_states']: gs['puzzle_states']['unreachable_vault']['is_open'] = True
            if 'vault_approach' in gs['locations'] and 'vault_door_main' in gs['locations']['vault_approach'].get('landmarks',{}):
                 gs['locations']['vault_approach']['description'] += " The massive vault door now stands open."
            return json.dumps({"message": "Vault door opened!"})
        return json.dumps({"message": f"World event '{event_id}' triggered."})

    def api_create_temporary_object(self, params):
        gs = self._get_gs()
        obj_type = params.get('type'); duration = params.get('duration', 10)
        from_landmark_id = params.get('from_landmark_id', 'player_current_pos'); to_landmark_id = params.get('to_landmark_id')
        location_id = params.get('location_id')
        if not all([obj_type, to_landmark_id, location_id]):
            return json.dumps({"error": "Missing type, to_landmark_id, or location_id for temporary object."})
        temp_obj_id = f"temp_{obj_type}_{str(uuid.uuid4())[:4]}"
        gs['temporary_objects'][temp_obj_id] = {
            'id': temp_obj_id, 'type': obj_type, 'location_id': location_id,
            'from': from_landmark_id, 'to': to_landmark_id,
            'creation_time': time.time(), 'duration': duration
        }
        log_to_world(f"A {obj_type} appeared from '{from_landmark_id}' to '{to_landmark_id}' in {gs['locations'][location_id]['name']}. It will last {duration}s.")
        return json.dumps({"message": f"{obj_type} created to '{to_landmark_id}'."})

    def api_get_entity_data(self, params):
        gs = self._get_gs()
        entity_id = params.get('entity_id')
        if not entity_id or entity_id not in gs['souls']: return json.dumps({"error": "Invalid entity_id."})
        soul_data = gs['souls'][entity_id]
        return json.dumps({"id": soul_data["id"], "name": soul_data["name"], "location_id": soul_data["location_id"]})

    def api_get_location_data(self, params):
        gs = self._get_gs()
        loc_id = params.get('location_id')
        if not loc_id or loc_id not in gs['locations']: return json.dumps({"error": "Invalid location_id."})
        loc_data = gs['locations'][loc_id]
        return json.dumps({"id": loc_data["id"], "name": loc_data["name"], "description": loc_data["description"], "exits": loc_data.get("exits",{}), "landmarks": loc_data.get("landmarks", {})})

    def api_get_environment_object_data(self, params):
        gs = self._get_gs()
        obj_id = params.get('object_id')
        if not obj_id or obj_id not in gs['environment_objects']:
            return json.dumps({"error": f"Environment object '{obj_id}' not found."})
        env_obj = gs['environment_objects'][obj_id]
        return json.dumps({'id': env_obj['id'], 'type': env_obj['type'], 'location_id': env_obj['location_id'], 'details': env_obj['details']})

    def _primitive_give_artifact_to_soul(self, params): # Kept as internal helper, might be used post-genesis
        gs = self._get_gs()
        soul_id, artifact_id = params.get('soul_id'), params.get('artifact_id')
        if not soul_id or not artifact_id: return "Error: Primitive give_artifact needs soul_id and artifact_id."
        if soul_id not in gs['souls'] or artifact_id not in gs['artifacts']: return f"Error: Invalid soul or artifact ID. Soul: {soul_id in gs['souls']}, Art: {artifact_id in gs['artifacts']}"
        if artifact_id not in gs['souls'][soul_id]['inventory']:
            gs['souls'][soul_id]['inventory'].append(artifact_id)
            log_to_world(f"{gs['souls'][soul_id]['name']} obtained {gs['artifacts'][artifact_id]['name']}.")
        return f"Artifact {gs['artifacts'][artifact_id]['name']} given to {gs['souls'][soul_id]['name']}."


def get_current_game_state(): return game_state

def register_host_api_for_llm(api_name, description_for_llm, parameters_schema_for_llm, callable_function):
    global game_api_registry
    game_api_registry[api_name] = {
        'description_for_llm': description_for_llm,
        'parameters_schema_for_llm': parameters_schema_for_llm,
        'callable_function': callable_function
    }
    send_debug_info(None, f"Host API '{api_name}' registered for LLM use.")

def generate_api_description_for_llm_prompt():
    if not game_api_registry: return "No host APIs are currently available."
    descriptions = ["Host APIs available in `external_apis` dictionary:", "Note: If an API is documented to return a JSON string, you MUST use `json.loads(result_string)` to parse it into a Python dictionary or list before accessing its elements."]
    for name, info in game_api_registry.items():
        param_desc_parts = []
        if info['parameters_schema_for_llm'] and info['parameters_schema_for_llm'].get('properties'):
            for pname, pinfo in info['parameters_schema_for_llm']['properties'].items():
                ptype = pinfo.get('type', 'any'); pdesc = pinfo.get('description', '')
                req = "(required)" if pname in info['parameters_schema_for_llm'].get('required', []) else "(optional)"
                param_desc_parts.append(f"  - `{pname}` ({ptype}) {req}: {pdesc}")
        param_str = "\n".join(param_desc_parts) if param_desc_parts else "  (No specific parameters defined in schema)"
        descriptions.append(f"\n- `external_apis['{name}'](args_dict)`:\n  Description: {info['description_for_llm']}\n  Expected `args_dict` structure:\n{param_str}")
    return "\n".join(descriptions)

def get_external_apis_for_execution():
    return {name: info['callable_function'] for name, info in game_api_registry.items()}

def log_to_world(message, broadcast=True):
    print("[WORLD]", message); game_state['world_log'].append(message)
    if len(game_state['world_log']) > 100: game_state['world_log'] = game_state['world_log'][-100:]
    if broadcast: broadcast_game_state_to_all_relevant() # Ensure this updates clients

def send_debug_info(sid, message):
    print(f"[DEBUG {sid if sid else 'GLOBAL'}] {message}")
    if sid: socketio.emit('debugInfo', f"[SERVER DEBUG] {message}", room=sid)
    elif demo_context.get("is_active") and demo_context.get("player_sid"):
        socketio.emit('debugInfo', f"[SERVER DEBUG] {message}", room=demo_context["player_sid"])

def process_initial_prompt_commands():
    send_debug_info(None, f"Processing initial prompt file: {INITIAL_PROMPT_FILE}")
    if demo_context["is_active"] and demo_context["initial_prompt_processed"]:
        send_debug_info(None, "Initial prompt already processed for this demo session. Skipping.")
        return
    try:
        with open(INITIAL_PROMPT_FILE, 'r') as f:
            prompt_commands = json.load(f)

        genesis_engine_created = False
        for idx, command_entry in enumerate(prompt_commands):
            command_name = command_entry.get("name")
            args = command_entry.get("args")
            if not command_name or args is None: continue

            send_debug_info(None, f"Initial Prompt CMD {idx+1}: {command_name} with args: {json.dumps(args)[:100]}...")
            result = "Error: Unknown command during initial prompt."

            if command_name == 'create_dynamic_function':
                args['host_provided_api_description_for_new_func'] = generate_api_description_for_llm_prompt()
                result = dynamic_executor.execute_dynamic_function(command_name, args, get_external_apis_for_execution())
                if "Successfully created/updated dynamic function: df_genesis_engine" in result:
                    genesis_engine_created = True
            elif command_name == 'df_genesis_engine': # Execute the genesis engine
                if not genesis_engine_created:
                     result = "Error: df_genesis_engine called before it was created by a preceding 'create_dynamic_function' command."
                else:
                    log_to_world("Server: Executing df_genesis_engine to build the world...", broadcast=True)
                    result = dynamic_executor.execute_dynamic_function(command_name, args, get_external_apis_for_execution())
                    log_to_world(f"Server: df_genesis_engine execution finished. Result: {result}", broadcast=True)
            else:
                # In this new model, initial_prompt.json should primarily use create_dynamic_function and then execute df_genesis_engine.
                # Other direct dynamic function calls could be supported but are not the main pattern for world setup.
                result = dynamic_executor.execute_dynamic_function(command_name, args, get_external_apis_for_execution())


            send_debug_info(None, f"Initial CMD '{command_name}' Result: {result}")
            if isinstance(result, str) and "Error:" in result:
                 print(f"Error processing initial prompt command: {command_name} - {result}")
                 log_to_world(f"Server Error during initial setup with {command_name}: {result}", broadcast=True)
            eventlet.sleep(0.01)

        if demo_context["is_active"]:
            demo_context["initial_prompt_processed"] = True
        send_debug_info(None, "Finished processing initial prompt commands.")

    except Exception as e:
        error_msg = f"Critical error during initial prompt processing: {e}\n{traceback.format_exc()}"
        print(error_msg)
        log_to_world(f"Server CRITICAL ERROR during initial setup: {e}", broadcast=True)
        send_debug_info(None, error_msg)


def get_filtered_game_state_for_soul(soul_id):
    soul = game_state['souls'].get(soul_id)
    if not soul: return {'error': "Soul not found"}
    current_loc_id = soul.get('location_id')
    current_loc = game_state['locations'].get(current_loc_id) if current_loc_id else None

    active_temp_objects_desc = []
    if current_loc_id:
        for obj_id, obj_data in list(game_state['temporary_objects'].items()):
            if obj_data['location_id'] == current_loc_id:
                if time.time() > obj_data['creation_time'] + obj_data['duration']:
                    del game_state['temporary_objects'][obj_id]
                    log_to_world(f"{obj_data['type']} from {obj_data['from']} to {obj_data['to']} vanished.", broadcast=False) # Avoid double broadcast
                    broadcast_game_state_to_all_relevant() # Single broadcast after potential deletions
                else:
                    active_temp_objects_desc.append(f"{obj_data['type']} to {obj_data['to']} (expires soon)")

    return {
        'playerSoul': {'id': soul['id'], 'name': soul['name'], 'locationId': current_loc_id},
        'currentLocation': {
            'name': current_loc['name'] if current_loc else "The Void",
            'description': current_loc['description'] if current_loc else "Lost in an unformed expanse.",
            'exits': current_loc.get('exits',{}) if current_loc else {},
            'landmarks': current_loc.get('landmarks', {}) if current_loc else {},
            'temporary_notes': ", ".join(active_temp_objects_desc) if active_temp_objects_desc else "None"
        },
        'inventory': [{'id': aid, 'name': game_state['artifacts'][aid]['name'],
                       'description': game_state['artifacts'][aid]['description'],
                       'toolName': game_state['artifacts'][aid].get('linked_dynamic_function_name')
                      } for aid in soul.get('inventory', []) if aid in game_state['artifacts']],
        'worldLog': game_state['world_log'][-20:], # Show more log for demo
    }

def send_game_state_update(sid, soul_id):
    state = get_filtered_game_state_for_soul(soul_id)
    socketio.emit('gameStateUpdate', state, room=sid)

def broadcast_game_state_to_all_relevant():
    for sid, soul_id in list(connected_souls_by_sid.items()):
        if soul_id in game_state['souls']: send_game_state_update(sid, soul_id)

def send_available_actions(sid, soul_id):
    state = get_filtered_game_state_for_soul(soul_id)
    if 'inventory' in state: socketio.emit('availableActions', state['inventory'], room=sid)


@app.route('/')
def index(): return send_from_directory(app.static_folder, 'index.html')

@socketio.on('connect')
def handle_connect():
    global game_primitives_handler # Ensure we're referencing the global one
    sid = request.sid
    player_soul_id = str(uuid.uuid4()) # Create new soul ID regardless

    next_player_num = game_state['server_metadata'].get('next_player_number', 1)
    player_name = f"Player_{next_player_num}"
    game_state['server_metadata']['next_player_number'] = next_player_num + 1
    start_loc_id = 'trial_chamber' # Default, will be set by genesis engine

    if demo_context["is_active"] and not demo_context["player_soul_id"]:
        player_name = "DemoPlayer"
        demo_context["player_soul_id"] = player_soul_id
        demo_context["player_sid"] = sid
        send_debug_info(sid, f"DemoPlayer connected ({player_soul_id}). SID {sid}.")
        log_to_world(f"{player_name} has entered. The world is about to be born from the digital ether!", broadcast=True)
        # Defer item giving until after genesis

        if not demo_context["initial_prompt_processed"]:
            send_debug_info(sid, "First demo player connected. Initiating world genesis via initial_prompt.json.")
            log_to_world("Server: Initiating World Genesis Protocol...", broadcast=True)
            eventlet.spawn_n(process_initial_prompt_commands) # Run in background to not block connect
            # Note: Player won't have items or be in the right location until genesis finishes
            # and _internal_perform_action_logic updates them. Consider a small delay or specific event.
    else: # Not a demo player or demo player already connected
        send_debug_info(sid, f"Player {player_name} ({player_soul_id}) connected. SID {sid}.")
        if not game_state['locations']: # If world isn't built (e.g. non-demo, fresh start)
            log_to_world("Server: No world found. Attempting to build from initial prompt.", broadcast=True)
            process_initial_prompt_commands() # Blocking for non-demo


    # Common soul setup
    game_state['souls'][player_soul_id] = {'id': player_soul_id, 'name': player_name, 'location_id': None, 'inventory': [], 'type': 'player', 'socket_id': sid}
    connected_souls_by_sid[sid] = player_soul_id
    socketio.emit('assignSoulId', player_soul_id, room=sid)

    # Post-genesis placement and item giving:
    # This needs to happen AFTER df_genesis_engine has run.
    # We'll add a socket event or make df_genesis_engine directly update the player.
    # For now, we assume df_genesis_engine is robust or we handle it after it's done.
    
    # A simple way for the demo: df_genesis_engine creates artifacts.
    # Then, after it runs, we give them to the DemoPlayer.
    def finalize_player_setup_after_genesis(p_soul_id, p_sid):
        send_debug_info(p_sid, f"Finalizing setup for {p_soul_id} post-genesis.")
        gs = game_state # current game_state
        if 'trial_chamber' in gs['locations']:
            gs['souls'][p_soul_id]['location_id'] = 'trial_chamber'
            log_to_world(f"{gs['souls'][p_soul_id]['name']} materializes in the newly formed 'Trial Chamber'.", broadcast=True)
        else:
            log_to_world(f"Warning: 'trial_chamber' not found for {gs['souls'][p_soul_id]['name']} after genesis.", broadcast=True)
            gs['souls'][p_soul_id]['location_id'] = list(gs['locations'].keys())[0] if gs['locations'] else None


        # Give items created by Genesis Engine
        artifacts_to_give = ["ember_01", "water_01", "wind_01", "key_01", "orb_01"]
        if game_primitives_handler: # Ensure handler is available
            for art_id_key in artifacts_to_give:
                # Find artifact by ID (assuming IDs are stable as per new initial_prompt)
                if art_id_key in gs['artifacts']:
                     game_primitives_handler._primitive_give_artifact_to_soul({'soul_id': p_soul_id, 'artifact_id': art_id_key})
                else:
                    log_to_world(f"Warning: Artifact {art_id_key} not found post-genesis for player {p_soul_id}", broadcast=True)
        
        send_game_state_update(p_sid, p_soul_id)
        send_available_actions(p_sid, p_soul_id)
        if demo_context["is_active"] and p_soul_id == demo_context["player_soul_id"] and demo_context["current_step"] == 0:
            if demo_context["script_data"]: # Ensure script is loaded
                send_debug_info(p_sid, "Spawning demo script execution.")
                eventlet.spawn_n(execute_demo_script_async)


    if demo_context["is_active"] and player_soul_id == demo_context["player_soul_id"]:
        # Wait a bit for genesis to complete if it was spawned
        def delayed_finalize():
            max_wait = 90 # Wait up to 90 seconds for genesis (was 20)
            waited = 0
            while not demo_context["initial_prompt_processed"] and waited < max_wait:
                eventlet.sleep(1)
                waited +=1
            if demo_context["initial_prompt_processed"]:
                send_debug_info(sid, "Genesis marked complete. Finalizing player setup.")
                finalize_player_setup_after_genesis(player_soul_id, sid)
            else:
                send_debug_info(sid, "Genesis did not complete in time. Player setup might be incomplete.")
                log_to_world("Server Warning: World genesis timed out. The world may be unstable.", broadcast=True)
                # Still try to finalize with whatever state exists
                finalize_player_setup_after_genesis(player_soul_id, sid)

        eventlet.spawn_n(delayed_finalize)
    else: # Non-demo player or non-first demo player
        # If world already exists, finalize immediately
        if game_state['locations']: # A proxy for "world is built"
            finalize_player_setup_after_genesis(player_soul_id, sid)
        else: # World not built, and this isn't the demo trigger connection
            log_to_world(f"Player {player_name} connected, but world genesis is pending or failed.", broadcast=True)


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid; soul_id = connected_souls_by_sid.pop(sid, None)
    if soul_id and soul_id in game_state['souls']:
        log_to_world(f"{game_state['souls'][soul_id]['name']} disconnected.")
        game_state['souls'][soul_id]['socket_id'] = None # Mark as disconnected
        if demo_context["is_active"] and demo_context["player_soul_id"] == soul_id:
            send_debug_info(None, "DemoPlayer disconnected. Stopping demo script if running.")
            # Reset demo context for a potential new demo run without server restart
            demo_context["player_soul_id"] = None
            demo_context["player_sid"] = None
            demo_context["current_step"] = 0
            demo_context["orb_catalyst_artifact_id_pending"] = None
            # demo_context["initial_prompt_processed"] = False # Allow re-genesis on next demo player connect
    # Do not broadcast here, let individual actions trigger broadcasts


@socketio.on('performAction')
def handle_perform_action(data):
    sid = request.sid
    if demo_context["is_active"] and demo_context["player_sid"] == sid and demo_context["script_data"]: # If demo script is running
        socketio.emit('actionResult', {'success': False, 'message': "DemoPlayer actions are scripted. Manual input ignored."}, room=sid)
        return

    player_soul_id = connected_souls_by_sid.get(sid)
    if not player_soul_id or player_soul_id not in game_state['souls']:
        socketio.emit('actionResult', {'success': False, 'message': "Soul not recognized."}, room=sid); return

    artifact_id = data.get('artifactId'); client_args = data.get('args', {})
    result_message, success_flag, prompt_for_tool_artifact_id = _internal_perform_action_logic(player_soul_id, artifact_id, client_args)

    if prompt_for_tool_artifact_id:
        socketio.emit('actionResult', {'success': success_flag, 'message': result_message, 'prompt_for_tool_artifact_id': prompt_for_tool_artifact_id}, room=sid)
    else:
        socketio.emit('actionResult', {'success': success_flag, 'message': str(result_message)}, room=sid)

    broadcast_game_state_to_all_relevant()
    send_available_actions(sid, player_soul_id)


def _internal_perform_action_logic(player_soul_id, artifact_id_or_name, action_args, artifact_name_starts_with=False):
    soul = game_state['souls'][player_soul_id]
    artifact = None; actual_artifact_id = None

    if artifact_name_starts_with:
        for inv_art_id in soul.get('inventory', []):
            art_data = game_state['artifacts'].get(inv_art_id)
            if art_data and art_data['name'].startswith(artifact_id_or_name):
                artifact = art_data; actual_artifact_id = inv_art_id; break
        if not artifact: return f"Artifact starting with name '{artifact_id_or_name}' not found in inventory.", False, None
    else:
        actual_artifact_id = artifact_id_or_name # Assume it's an ID
        artifact = game_state['artifacts'].get(actual_artifact_id)
        if not artifact : # Try by exact name if ID lookup fails
            for inv_art_id in soul.get('inventory', []):
                art_data = game_state['artifacts'].get(inv_art_id)
                if art_data and art_data['name'] == artifact_id_or_name:
                    artifact = art_data; actual_artifact_id = inv_art_id; break
            if not artifact: return f"Artifact ID or name '{artifact_id_or_name}' not found.", False, None

    if actual_artifact_id not in soul.get('inventory', []): return "Artifact not possessed.", False, None
    dynamic_function_name = artifact.get('linked_dynamic_function_name')
    if not dynamic_function_name: return "Artifact is inert or misconfigured.", False, None

    params_for_function = {
        'soul_id': player_soul_id, 'location_id': soul['location_id'],
        'artifact_id': actual_artifact_id, 'artifact_properties': artifact.get('properties', {}),
        **action_args
    }
    send_debug_info(soul.get('socket_id'), f"Soul {player_soul_id} using '{artifact['name']}' for '{dynamic_function_name}' with args: {json.dumps(params_for_function)[:100]}")
    execution_result = dynamic_executor.execute_dynamic_function(dynamic_function_name, params_for_function, get_external_apis_for_execution())
    send_debug_info(soul.get('socket_id'), f"Dynamic func '{dynamic_function_name}' result: {execution_result}")

    success = not (isinstance(execution_result, str) and execution_result.startswith("Error:"))
    if execution_result == "EVENT:PROMPT_USER_FOR_TOOL_DESCRIPTION":
        if demo_context["is_active"] and demo_context["player_soul_id"] == player_soul_id:
            demo_context["orb_catalyst_artifact_id_pending"] = actual_artifact_id
        return execution_result, True, actual_artifact_id
    else:
        # If the result is a JSON string from an API call, parse it for a user-friendly message
        try:
            parsed_result = json.loads(execution_result)
            user_message = parsed_result.get('message', execution_result)
            if 'error' in parsed_result: success = False; user_message = parsed_result['error']
            return user_message, success, None
        except (json.JSONDecodeError, TypeError):
            return execution_result, success, None


@socketio.on('submitToolDescription')
def handle_submit_tool_description(data):
    sid = request.sid
    if demo_context["is_active"] and demo_context["player_sid"] == sid and demo_context["script_data"]:
        log_to_world("[DEMO WARNING] DemoPlayer tried to submit tool description via socket. Script should handle this.", broadcast=True)
        return

    player_soul_id = connected_souls_by_sid.get(sid)
    if not player_soul_id: return

    description = data.get('description'); catalyst_artifact_id = data.get('catalyst_artifact_id')
    message, success = _internal_submit_tool_description_logic(player_soul_id, description, catalyst_artifact_id)
    socketio.emit('actionResult', {'success': success, 'message': message}, room=sid)
    broadcast_game_state_to_all_relevant()
    send_available_actions(sid, player_soul_id)


def _internal_submit_tool_description_logic(player_soul_id, description, catalyst_artifact_id):
    soul = game_state['souls'][player_soul_id]
    catalyst_artifact = game_state['artifacts'].get(catalyst_artifact_id)
    if not description or not catalyst_artifact: return "Missing description or catalyst artifact for tool creation.", False

    new_func_name = f"df_user_{player_soul_id[:4]}_{str(uuid.uuid4())[:4]}"
    # Refined prompt for light bridge specifically
    refined_description_for_llm = (
        f"Player described: '{description}'. This function MUST create a temporary light bridge. "
        "1. It needs 'soul_id' and 'location_id' from params (implicitly provided by host). "
        "2. Call external_apis['host_create_temporary_object'] with args: "
        "{'type': 'light_bridge', 'duration': 15, 'from_landmark_id': 'player_current_pos', "
        "'to_landmark_id': 'keyhole_platform_exit', 'location_id': params['location_id']}. "
        "3. Return the JSON string result from 'host_create_temporary_object' directly. "
        "Do NOT add flavor text, just return the API's JSON string."
    )
    new_func_params_schema = {"type": "object", "properties": {}} # No explicit client args for this one

    func_creation_args = {
        'new_function_name': new_func_name,
        'new_function_description': refined_description_for_llm,
        'new_function_parameters_schema': new_func_params_schema,
        'host_provided_api_description_for_new_func': generate_api_description_for_llm_prompt()
    }
    send_debug_info(soul.get('socket_id'), f"Attempting to create new tool function '{new_func_name}' for: {description}")
    func_creation_result = dynamic_executor.execute_dynamic_function("create_dynamic_function", func_creation_args, get_external_apis_for_execution())

    if isinstance(func_creation_result, str) and func_creation_result.startswith("Error:"):
        log_to_world(f"{soul['name']}'s attempt to use Orb of Ingenuity failed: {func_creation_result}", broadcast=True)
        return f"Orb fizzles: {func_creation_result}", False
    else:
        new_charged_artifact_id = f"charged_orb_{str(uuid.uuid4())[:8]}"
        charged_orb_name = f"Orb: {description[:25]}..."
        charged_orb_desc = f"The Orb of Ingenuity, now resonating with the power to: {description}"
        
        # Use host_core_add_artifact_to_gamestate directly or via game_primitives_handler
        create_params = {
            'id': new_charged_artifact_id, 'name': charged_orb_name, 'description': charged_orb_desc,
            'linked_dynamic_function_name': new_func_name, 'properties': {'creator_soul_id': player_soul_id}
        }
        charged_art_result_msg = game_primitives_handler.host_core_add_artifact_to_gamestate(create_params)

        if "Error API" not in charged_art_result_msg:
            game_primitives_handler._primitive_give_artifact_to_soul({'soul_id': player_soul_id, 'artifact_id': new_charged_artifact_id})
            final_msg = f"The Orb of Ingenuity glows! You've learned to manifest '{charged_orb_name}'!"
            log_to_world(f"{soul['name']}: {final_msg}", broadcast=True)
            return final_msg, True
        else:
            final_msg = f"Orb created function '{new_func_name}', but failed to create usable artifact. {charged_art_result_msg}"
            log_to_world(f"{soul['name']}'s Orb attempt had issues: {final_msg}", broadcast=True)
            return final_msg, False


@socketio.on('requestState')
def handle_request_state():
    sid = request.sid; player_soul_id = connected_souls_by_sid.get(sid)
    if player_soul_id and player_soul_id in game_state['souls']:
        send_game_state_update(sid, player_soul_id); send_available_actions(sid, player_soul_id)

def save_game_state():
    print("[SERVER] Saving game state...")
    try:
        with open(SAVE_FILE, 'w') as f: json.dump(game_state, f, indent=2)
        print("[SERVER] Game state saved.")
    except Exception as e: print(f"[SERVER] Error saving game state: {e}\n{traceback.format_exc()}")

def load_game_state():
    global game_state
    if os.path.exists(SAVE_FILE):
        print("[SERVER] Loading game state from save...")
        try:
            with open(SAVE_FILE, 'r') as f: loaded_data = json.load(f)
            game_state.update(loaded_data)
            for key in ['souls', 'locations', 'artifacts', 'environment_objects', 'puzzle_states', 'temporary_objects', 'world_log', 'server_metadata']:
                if key not in game_state: game_state[key] = {} if key != 'world_log' else []
            if 'next_player_number' not in game_state['server_metadata']: game_state['server_metadata']['next_player_number'] = 1
            print("[SERVER] Game state loaded.")
        except Exception as e:
            print(f"[SERVER] Error loading game state: {e}. Starting with default.")
            # Default state re-assigned in main if needed
    else:
        print("[SERVER] No save file found. Starting with default empty state.")

def signal_handler_save_on_exit(sig, frame):
    print('[SERVER] SIGINT received, saving game state before exit...')
    save_game_state(); sys.exit(0)

def trigger_llm_repair_process(function_name, error, traceback_str, params_for_function, host_api_description_for_repair_context):
    send_debug_info(None, f"LLM REPAIR TRIGGERED: For function '{function_name}' due to error: {error}")
    # Placeholder for actual repair logic using LLM

def load_demo_script(filepath):
    try:
        with open(filepath, 'r') as f: script_data = json.load(f)
        send_debug_info(None, f"[DEMO] Successfully loaded demo script: {filepath}")
        return script_data
    except Exception as e:
        send_debug_info(None, f"[DEMO] Error loading demo script {filepath}: {e}")
        return None

def execute_demo_script_async():
    send_debug_info(demo_context.get("player_sid"), "[DEMO] Starting script execution...")
    log_to_world("[DEMO] Demo sequence initiated.", broadcast=True)

    while demo_context["is_active"] and demo_context["script_data"] and \
          demo_context["current_step"] < len(demo_context["script_data"]):
        if not demo_context["player_soul_id"] or demo_context["player_soul_id"] not in game_state["souls"]:
            send_debug_info(None, "[DEMO] DemoPlayer not found or disconnected. Stopping script.")
            demo_context["is_active"] = False; break

        step = demo_context["script_data"][demo_context["current_step"]]
        action = step.get("action"); log_msg = step.get("log_message", f"Executing demo step {demo_context['current_step'] + 1}: {action}")
        send_debug_info(demo_context["player_sid"], f"[DEMO] {log_msg}")
        log_to_world(f"[DEMO] {log_msg}", broadcast=True)

        if action == "WAIT": eventlet.sleep(step.get("duration_ms", 1000) / 1000.0)
        elif action == "USE_ARTIFACT":
            name_to_use = step.get("artifact_name", step.get("artifact_name_starts_with"))
            is_prefix = bool(step.get("artifact_name_starts_with"))
            res_msg, success, p_art_id = _internal_perform_action_logic(demo_context["player_soul_id"], name_to_use, step.get("args", {}), artifact_name_starts_with=is_prefix)
            log_to_world(f"[DEMO] Action result: {res_msg}", broadcast=True)
            if res_msg == "EVENT:PROMPT_USER_FOR_TOOL_DESCRIPTION": demo_context["orb_catalyst_artifact_id_pending"] = p_art_id
            elif not success: send_debug_info(demo_context["player_sid"], f"[DEMO] Error in USE_ARTIFACT: {res_msg}. Stopping."); demo_context["is_active"] = False; break
        elif action == "DESCRIBE_TOOL":
            if not demo_context["orb_catalyst_artifact_id_pending"]:
                send_debug_info(demo_context["player_sid"], "[DEMO] Error: DESCRIBE_TOOL called but no Orb catalyst ID pending. Stopping."); demo_context["is_active"] = False; break
            res_msg, success = _internal_submit_tool_description_logic(demo_context["player_soul_id"], step.get("description"), demo_context["orb_catalyst_artifact_id_pending"])
            log_to_world(f"[DEMO] Tool description result: {res_msg}", broadcast=True)
            demo_context["orb_catalyst_artifact_id_pending"] = None
            if not success: send_debug_info(demo_context["player_sid"], f"[DEMO] Error in DESCRIBE_TOOL: {res_msg}. Stopping."); demo_context["is_active"] = False; break
        elif action == "COMMENT": pass
        demo_context["current_step"] += 1
        broadcast_game_state_to_all_relevant()
        if demo_context["player_sid"] and demo_context["player_soul_id"]: send_available_actions(demo_context["player_sid"], demo_context["player_soul_id"])
        eventlet.sleep(0.2) # Breather

    if demo_context["is_active"]: send_debug_info(demo_context.get("player_sid"), "[DEMO] Script finished.")
    log_to_world("[DEMO] Demo sequence complete.", broadcast=True)
    demo_context["is_active"] = False

def main():
    global game_state, game_primitives_handler, demo_context
    parser = argparse.ArgumentParser(description="Orb of Ingenuity Demo Server")
    parser.add_argument('--recreate', action='store_true', help='Recreate world, deleting save and dynamic functions.')
    parser.add_argument('--demo', type=str, metavar='SCRIPT_FILE', help='Run in demo mode with specified JSON script.')
    args = parser.parse_args()

    app.config['RECREATING_WORLD'] = args.recreate
    default_game_state = {'souls': {}, 'locations': {}, 'artifacts': {}, 'environment_objects': {}, 'puzzle_states': {}, 'temporary_objects': {}, 'world_log': ["Welcome!"], 'server_metadata': {'next_player_number': 1}}

    if args.demo:
        demo_script_content = load_demo_script(args.demo)
        if demo_script_content:
            demo_context["is_active"] = True; demo_context["script_path"] = args.demo
            demo_context["script_data"] = demo_script_content; demo_context["current_step"] = 0
            demo_context["initial_prompt_processed"] = False # Reset for demo
            print(f"[SERVER] Demo mode activated with script: {args.demo}")
            # game_state['world_log'] = [f"[DEMO] Server starting with script: {args.demo}"] # Log for demo player
        else:
            print(f"[SERVER] Failed to load demo script {args.demo}. Starting normally.")
            demo_context["is_active"] = False

    print("Initializing server for Orb of Ingenuity Demo...")
    dynamic_executor.initialize_store()
    dynamic_executor.llm_repair_callback = trigger_llm_repair_process

    if args.recreate:
        print("[SERVER] --recreate flag set. Clearing existing save and dynamic functions.")
        if os.path.exists(SAVE_FILE):
            try: os.remove(SAVE_FILE); print(f"[SERVER] Deleted save file: {SAVE_FILE}")
            except Exception as e: print(f"[SERVER] Error deleting save file {SAVE_FILE}: {e}")
        dynamic_executor.clear_function_store()
        game_state = json.loads(json.dumps(default_game_state)) # Deep copy
        game_state['world_log'] = ["Welcome (World Recreated)!"]
        if demo_context["is_active"]: demo_context["initial_prompt_processed"] = False
    else:
        load_game_state()
        if not game_state['locations'] and demo_context["is_active"]: # If loaded state is empty and in demo, ensure re-processing
             demo_context["initial_prompt_processed"] = False


    game_primitives_handler = GamePrimitiveHandler(get_current_game_state, socketio)

    # Register Host Core APIs (used by Genesis Engine)
    register_host_api_for_llm('host_core_add_location_to_gamestate', "Core API: Creates a new location.", {'type':'object', 'properties': {'id':{'type':'string'}, 'name':{'type':'string'}, 'description':{'type':'string'}, 'landmarks':{'type':'object'}}, 'required':['id','name','description']}, game_primitives_handler.host_core_add_location_to_gamestate)
    register_host_api_for_llm('host_core_add_artifact_to_gamestate', "Core API: Creates a new artifact.", {'type':'object', 'properties': {'id':{'type':'string'}, 'name':{'type':'string'}, 'description':{'type':'string'}, 'properties':{'type':'object'}, 'linked_dynamic_function_name':{'type':'string'}}, 'required':['id','name','description']}, game_primitives_handler.host_core_add_artifact_to_gamestate)
    register_host_api_for_llm('host_core_add_env_object_to_gamestate', "Core API: Creates an environment object.", {'type':'object', 'properties': {'id':{'type':'string'}, 'location_id':{'type':'string'}, 'type':{'type':'string'}, 'details':{'type':'object'}}, 'required':['id','location_id','type']}, game_primitives_handler.host_core_add_env_object_to_gamestate)
    register_host_api_for_llm('host_core_initialize_puzzle_state', "Core API: Initializes a puzzle's state.", {'type':'object', 'properties': {'id':{'type':'string'}, 'initial_state':{'type':'object'}}, 'required':['id','initial_state']}, game_primitives_handler.host_core_initialize_puzzle_state)
    register_host_api_for_llm('host_log_message_to_world', "Core API: Logs a message to the global game world log.", {'type':'object', 'properties': {'message':{'type':'string'}}, 'required':['message']}, game_primitives_handler.host_log_message_to_world)

    # Register Player-Interactive APIs
    register_host_api_for_llm('host_apply_effect_on_environment_object', "Applies an effect to an environment object's 'details'. Returns JSON.", {'type':'object', 'properties': {'object_id':{'type':'string'}, 'effect_details':{'type':'object'}}, 'required':['object_id', 'effect_details']}, game_primitives_handler.api_apply_effect_on_environment_object)
    register_host_api_for_llm('host_check_puzzle_condition', "Checks puzzle conditions. Returns JSON: {'condition_met': bool, 'message': str}.", {'type':'object', 'properties': {'puzzle_id':{'type':'string'}}, 'required':['puzzle_id']}, game_primitives_handler.api_check_puzzle_condition)
    register_host_api_for_llm('host_trigger_world_event', "Triggers a world event. Returns JSON.", {'type':'object', 'properties': {'event_id':{'type':'string'}, 'soul_id':{'type':'string'}}, 'required':['event_id', 'soul_id']}, game_primitives_handler.api_trigger_world_event)
    register_host_api_for_llm('host_create_temporary_object', "Creates a temporary object. Returns JSON.", {'type':'object', 'properties': {'type':{'type':'string'},'duration':{'type':'integer'},'from_landmark_id': {'type':'string'},'to_landmark_id':{'type':'string'},'location_id':{'type':'string'}}, 'required':['type', 'to_landmark_id', 'location_id']}, game_primitives_handler.api_create_temporary_object)
    register_host_api_for_llm('host_get_entity_data', "Retrieves entity data. Returns JSON.", {'type':'object', 'properties': {'entity_id':{'type':'string'}}, 'required':['entity_id']}, game_primitives_handler.api_get_entity_data)
    register_host_api_for_llm('host_get_location_data', "Retrieves location data. Returns JSON.", {'type':'object', 'properties': {'location_id':{'type':'string'}}, 'required':['location_id']}, game_primitives_handler.api_get_location_data)
    register_host_api_for_llm('host_get_environment_object_data', "Retrieves env object data. Returns JSON.", {'type':'object', 'properties': {'object_id':{'type':'string'}}, 'required':['object_id']}, game_primitives_handler.api_get_environment_object_data)


    if not demo_context["is_active"]: # If not in demo mode, process initial prompt now
        send_debug_info(None, "Non-demo mode: Processing initial prompt at startup.")
        process_initial_prompt_commands()
    else:
        send_debug_info(None, "Demo mode: World genesis will occur upon first player connection.")
        # Ensure world log is clean for demo start if recreating
        if args.recreate : game_state['world_log'] = [f"[DEMO] Server awaiting DemoPlayer. World will be born live..."]


    signal.signal(signal.SIGINT, signal_handler_save_on_exit)
    print(f"Server starting on http://0.0.0.0:{PORT}")
    socketio.run(app, host='0.0.0.0', port=PORT, debug=False, use_reloader=False) # use_reloader=False important for single init

if __name__ == '__main__':
    main()
