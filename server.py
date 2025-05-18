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
    'temporary_objects': {}, # Stores active temporary objects like light bridges
    'world_log': ["Welcome to the Orb of Ingenuity Demo!"],
    'server_metadata': {'next_player_number': 1, 'initial_prompt_processing_started': False, 'initial_prompt_processing_complete': False},
    'world_properties': {}, # For LLM-defined global settings
    'world_event_handlers': {}, # For LLM-defined event handlers
}
connected_souls_by_sid = {} # sid: soul_id

demo_context = {
    "is_active": False,
    "script_path": None,
    "script_data": [],
    "current_step": 0,
    "player_soul_id": None,
    "player_sid": None,
    "orb_catalyst_artifact_id_pending_creation": None,
    "newly_created_tool_function_name_pending_artifact": None,
    "user_description_pending_artifact": None,
    "script_execution_started": False,
    "current_focused_landmark_key": None # NEW: For demo player logical focus
}

app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path='')
socketio = SocketIO(app, cors_allowed_origins="*")

dynamic_executor = DynamicFunctionExecutor()
game_api_registry = {}
game_primitives_handler = None

class GamePrimitiveHandler:
    def __init__(self, gs_ref, sio_ref):
        self.gs = gs_ref
        self.sio = sio_ref

    def _get_gs(self): return self.gs()

    def host_core_add_location_to_gamestate(self, params):
        gs = self._get_gs()
        loc_id = params.get('id')
        name = params.get('name', loc_id)
        description = params.get('description', "An undescribed place.")
        if not loc_id: return "Error API (host_core_add_location_to_gamestate): 'id' is required."
        if loc_id in gs['locations']: return f"Error API (host_core_add_location_to_gamestate): Location ID '{loc_id}' already exists."
        gs['locations'][loc_id] = {
            'id': loc_id, 'name': name, 'description': description,
            'exits': {},
            'landmarks': {},
            'client_visual_config': {}
        }
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
        return f"Host API: Artifact '{name}' ({art_id}) with func '{linked_func_name}' created."

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

    def host_log_message_to_world(self, params):
        message = params.get("message", "Untagged message from host API.")
        log_to_world(f"{message}")
        return "Host API: Message logged to world."

    def host_give_artifact_to_soul(self, params):
        gs = self._get_gs()
        soul_id, artifact_id = params.get('soul_id'), params.get('artifact_id')
        if not soul_id or not artifact_id: return json.dumps({"error": "Primitive give_artifact needs soul_id and artifact_id."})
        if soul_id not in gs['souls'] : return json.dumps({"error": f"Soul {soul_id} not found."})
        if artifact_id not in gs['artifacts']: return json.dumps({"error": f"Artifact {artifact_id} not found."})

        if 'inventory' not in gs['souls'][soul_id]: gs['souls'][soul_id]['inventory'] = []
        if artifact_id not in gs['souls'][soul_id]['inventory']:
            gs['souls'][soul_id]['inventory'].append(artifact_id)
            log_to_world(f"{gs['souls'][soul_id]['name']} obtained {gs['artifacts'][artifact_id]['name']}.")
            return json.dumps({"message": f"Artifact {gs['artifacts'][artifact_id]['name']} given to {gs['souls'][soul_id]['name']}."})
        return json.dumps({"message": f"Artifact {gs['artifacts'][artifact_id]['name']} already in inventory of {gs['souls'][soul_id]['name']}."})

    def host_apply_effect_on_environment_object(self, params):
        gs = self._get_gs()
        obj_id = params.get('object_id'); effect_to_apply = params.get('effect_details')
        if not obj_id or obj_id not in gs['environment_objects']: return json.dumps({"error": "Invalid object_id."})
        if not effect_to_apply or not isinstance(effect_to_apply, dict): return json.dumps({"error": "Invalid effect_details."})

        env_obj = gs['environment_objects'][obj_id]
        for key, value in effect_to_apply.items():
            env_obj['details'][key] = value

        log_to_world(f"Effect applied to env object '{obj_id}': {json.dumps(effect_to_apply)}")
        broadcast_game_state_to_all_relevant()
        return json.dumps({"message": f"Effect applied to '{obj_id}'."})

    def host_check_puzzle_condition(self, params):
        gs = self._get_gs()
        puzzle_id = params.get('puzzle_id')
        if not puzzle_id or puzzle_id not in gs['puzzle_states']:
            return json.dumps({'condition_met': False, 'message': f'Puzzle {puzzle_id} not found.'})

        puzzle_data = gs['puzzle_states'][puzzle_id]
        checking_func_name = puzzle_data.get('checking_dynamic_function_name')

        if not checking_func_name:
            return json.dumps({'condition_met': False, 'message': f'No checking function registered for puzzle {puzzle_id}.'})

        execution_params = {'puzzle_id': puzzle_id, 'current_puzzle_state': puzzle_data}
        result_str = dynamic_executor.execute_dynamic_function(checking_func_name, execution_params, get_external_apis_for_execution())

        try:
            result_json = json.loads(result_str)
            return json.dumps(result_json)
        except json.JSONDecodeError:
            return json.dumps({'condition_met': False, 'message': f'Error: Puzzle checker {checking_func_name} for {puzzle_id} returned invalid JSON: {result_str}'})
        except Exception as e:
            return json.dumps({'condition_met': False, 'message': f'Error executing puzzle checker {checking_func_name} for {puzzle_id}: {e}'})

    def host_trigger_world_event(self, params):
        gs = self._get_gs()
        event_id = params.get('event_id')
        acting_soul_id = params.get('soul_id')
        event_params = params.get('event_params', {})

        if not event_id or event_id not in gs.get('world_event_handlers', {}):
            log_to_world(f"Warning: World event '{event_id}' triggered but no handler registered.")
            return json.dumps({"message": f"World event '{event_id}' triggered, no specific handler action taken."})

        handler_func_name = gs['world_event_handlers'][event_id]
        execution_params = {
            'event_id': event_id,
            'soul_id': acting_soul_id,
            **event_params
        }

        log_to_world(f"World event '{event_id}' processing via '{handler_func_name}'.")
        result_str = dynamic_executor.execute_dynamic_function(handler_func_name, execution_params, get_external_apis_for_execution())
        broadcast_game_state_to_all_relevant()
        return json.dumps({"message": f"Event '{event_id}' processed. Handler result: {result_str}"})


    def host_create_temporary_object(self, params):
        gs = self._get_gs()
        obj_type = params.get('type')
        duration = params.get('duration', 10)
        from_landmark_id = params.get('from_landmark_id', 'player_current_pos')
        to_landmark_id = params.get('to_landmark_id')
        location_id = params.get('location_id')
        client_visual_config = params.get('client_visual_config', {})

        if not all([obj_type, to_landmark_id, location_id]):
            return json.dumps({"error": "Missing type, to_landmark_id, or location_id for temporary object."})
        if location_id not in gs['locations']: return json.dumps({"error": f"Location '{location_id}' not found for temporary object."})

        temp_obj_id = f"temp_{obj_type}_{str(uuid.uuid4())[:4]}"
        gs['temporary_objects'][temp_obj_id] = {
            'id': temp_obj_id, 'type': obj_type, 'location_id': location_id,
            'from_landmark_id': from_landmark_id, 'to_landmark_id': to_landmark_id,
            'creation_time': time.time(), 'duration': int(duration),
            'creator_soul_id': params.get('soul_id'),
            'client_visual_config': client_visual_config
        }
        log_to_world(f"A {obj_type} appeared from '{from_landmark_id}' to '{to_landmark_id}' in {gs['locations'][location_id]['name']}. It will last {duration}s.")
        broadcast_game_state_to_all_relevant()
        return json.dumps({"message": f"{obj_type} created to '{to_landmark_id}'.", "object_id": temp_obj_id})

    def host_get_entity_data(self, params):
        gs = self._get_gs()
        entity_id = params.get('entity_id')
        if not entity_id or entity_id not in gs['souls']: return json.dumps({"error": "Invalid entity_id."})
        soul_data = gs['souls'][entity_id]
        return json.dumps({"id": soul_data["id"], "name": soul_data["name"], "location_id": soul_data["location_id"]})

    def host_get_location_data(self, params):
        gs = self._get_gs()
        loc_id = params.get('location_id')
        if not loc_id or loc_id not in gs['locations']: return json.dumps({"error": "Invalid location_id."})
        loc_data = gs['locations'][loc_id]
        return json.dumps({"id": loc_data["id"], "name": loc_data["name"], "description": loc_data["description"], "exits": loc_data.get("exits",{})})

    def host_get_environment_object_data(self, params):
        gs = self._get_gs()
        obj_id = params.get('object_id')
        if not obj_id or obj_id not in gs['environment_objects']:
            return json.dumps({"error": f"Environment object '{obj_id}' not found."})
        env_obj = gs['environment_objects'][obj_id]
        return json.dumps({'id': env_obj['id'], 'type': env_obj['type'], 'location_id': env_obj['location_id'], 'details': env_obj['details']})

    def host_set_world_property(self, params):
        gs = self._get_gs()
        property_name = params.get('property_name')
        property_value = params.get('property_value')
        if not property_name: return json.dumps({"error": "property_name is required."})
        gs['world_properties'][property_name] = property_value
        return json.dumps({"message": f"World property '{property_name}' set."})

    def host_register_puzzle_check_function(self, params):
        gs = self._get_gs()
        puzzle_id = params.get('puzzle_id')
        checking_dynamic_function_name = params.get('checking_dynamic_function_name')
        if not puzzle_id or not checking_dynamic_function_name:
            return json.dumps({"error": "puzzle_id and checking_dynamic_function_name are required."})
        if puzzle_id not in gs['puzzle_states']:
             gs['puzzle_states'][puzzle_id] = {'id': puzzle_id}
        gs['puzzle_states'][puzzle_id]['checking_dynamic_function_name'] = checking_dynamic_function_name
        return json.dumps({"message": f"Puzzle checker '{checking_dynamic_function_name}' registered for puzzle '{puzzle_id}'."})

    def host_register_event_handler_function(self, params):
        gs = self._get_gs()
        event_id = params.get('event_id')
        handler_dynamic_function_name = params.get('handler_dynamic_function_name')
        if not event_id or not handler_dynamic_function_name:
            return json.dumps({"error": "event_id and handler_dynamic_function_name are required."})
        if 'world_event_handlers' not in gs: gs['world_event_handlers'] = {}
        gs['world_event_handlers'][event_id] = handler_dynamic_function_name
        return json.dumps({"message": f"Event handler '{handler_dynamic_function_name}' registered for event '{event_id}'."})

    def host_set_location_visual_config(self, params):
        gs = self._get_gs()
        location_id = params.get('location_id')
        config = params.get('config')
        if not location_id or not config: return json.dumps({"error": "location_id and config are required."})
        if location_id not in gs['locations']: return json.dumps({"error": f"Location '{location_id}' not found."})
        gs['locations'][location_id]['client_visual_config'] = config
        return json.dumps({"message": f"Visual config set for location '{location_id}'."})

    def host_set_landmark_visual_config(self, params):
        gs = self._get_gs()
        location_id = params.get('location_id')
        landmark_key = params.get('landmark_key')
        config = params.get('config')

        if not location_id or not landmark_key or not config:
            return json.dumps({"error": "location_id, landmark_key, and config are required."})
        if location_id not in gs['locations']: return json.dumps({"error": f"Location '{location_id}' not found."})

        if 'landmarks' not in gs['locations'][location_id]: gs['locations'][location_id]['landmarks'] = {}

        gs['locations'][location_id]['landmarks'][landmark_key] = {
            'key': landmark_key,
            'name': config.get('display_name', landmark_key),
            'client_visual_config': config
        }
        if config.get('is_exit_to_location_id'):
            if 'exits' not in gs['locations'][location_id]: gs['locations'][location_id]['exits'] = {}
            gs['locations'][location_id]['exits'][landmark_key] = config['is_exit_to_location_id']

        return json.dumps({"message": f"Visual and semantic config set for landmark '{landmark_key}' in location '{location_id}'."})

    def host_set_puzzle_properties(self, params):
        gs = self._get_gs()
        puzzle_id = params.get('puzzle_id')
        properties_to_set = params.get('properties', {})
        if not puzzle_id or not properties_to_set:
            return json.dumps({"error": "puzzle_id and properties are required."})
        if puzzle_id not in gs['puzzle_states']:
            return json.dumps({"error": f"Puzzle '{puzzle_id}' not found."})
        for key, value in properties_to_set.items():
            gs['puzzle_states'][puzzle_id][key] = value
        broadcast_game_state_to_all_relevant()
        return json.dumps({"message": f"Properties updated for puzzle '{puzzle_id}'."})


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
    descriptions = ["Host APIs available in `external_apis` dictionary:", "Note: If an API is documented to return a JSON string, you MUST use `json.loads(result_string)` to parse it into a Python dictionary or list before accessing its elements. The `json` module is available implicitly or via `import json`."]
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
    if broadcast: broadcast_game_state_to_all_relevant()

def send_debug_info(sid, message):
    print(f"[DEBUG {sid if sid else 'GLOBAL'}] {message}")
    if sid: socketio.emit('debugInfo', f"[SERVER DEBUG] {message}", room=sid)
    elif demo_context.get("is_active") and demo_context.get("player_sid"):
        socketio.emit('debugInfo', f"[SERVER DEBUG] {message}", room=demo_context["player_sid"])

def process_initial_prompt_commands():
    gs_meta = game_state['server_metadata']
    if gs_meta['initial_prompt_processing_complete']:
        send_debug_info(None, "Initial prompt already processed for this server session. Skipping.")
        return
    if gs_meta['initial_prompt_processing_started']:
        send_debug_info(None, "Initial prompt processing already started. Waiting for completion.")
        return

    gs_meta['initial_prompt_processing_started'] = True
    send_debug_info(None, f"Processing initial prompt file: {INITIAL_PROMPT_FILE}")
    genesis_engine_was_run = False
    try:
        with open(INITIAL_PROMPT_FILE, 'r') as f:
            prompt_commands = json.load(f)

        for idx, command_entry in enumerate(prompt_commands):
            command_name = command_entry.get("name")
            args = command_entry.get("args")
            if not command_name or args is None: continue

            send_debug_info(None, f"Initial Prompt CMD {idx+1}: {command_name} with args: {json.dumps(args)[:100]}...")
            result = "Error: Unknown command during initial prompt."

            if command_name == 'create_dynamic_function':
                args['host_provided_api_description_for_new_func'] = generate_api_description_for_llm_prompt()
                result = dynamic_executor.execute_dynamic_function(command_name, args, get_external_apis_for_execution())
            elif command_name == 'df_genesis_engine':
                genesis_engine_was_run = True
                log_to_world("Server: Executing df_genesis_engine to build the world...", broadcast=True)
                result = dynamic_executor.execute_dynamic_function(command_name, args, get_external_apis_for_execution())
                log_to_world(f"Server: df_genesis_engine execution finished. Result: {result}", broadcast=True)
            else:
                result = dynamic_executor.execute_dynamic_function(command_name, args, get_external_apis_for_execution())


            send_debug_info(None, f"Initial CMD '{command_name}' Result: {result}")
            if isinstance(result, str) and "Error:" in result:
                 print(f"Error processing initial prompt command: {command_name} - {result}")
                 log_to_world(f"Server Error during initial setup with {command_name}: {result}", broadcast=True)
            eventlet.sleep(0.01)

        send_debug_info(None, "Finished processing initial prompt commands.")
        gs_meta['initial_prompt_processing_complete'] = True

        if genesis_engine_was_run:
            send_debug_info(None, "Genesis engine was run. Checking for players in LIMBO_VOID...")
            initial_start_location = game_state.get('world_properties',{}).get('initial_start_location_id', list(game_state['locations'].keys())[0] if game_state['locations'] else None)
            if initial_start_location and initial_start_location in game_state['locations']:
                limbo_players_updated_sids = []
                for soul_id, soul_data in list(game_state['souls'].items()):
                    if soul_data.get('location_id') == "LIMBO_VOID" or soul_data.get('location_id') is None:
                        game_state['souls'][soul_id]['location_id'] = initial_start_location
                        log_to_world(f"{soul_data['name']} has been brought from the Void into '{game_state['locations'][initial_start_location]['name']}'.", broadcast=False)
                        finalize_player_setup_after_genesis(soul_id, soul_data.get('socket_id'), from_limbo=True)
                        if soul_data.get('socket_id'):
                            limbo_players_updated_sids.append(soul_data.get('socket_id'))

                if limbo_players_updated_sids:
                    broadcast_game_state_to_all_relevant()
            else:
                log_to_world(f"Warning: Initial start location '{initial_start_location}' (from world_properties or fallback) not found after genesis. Players in LIMBO_VOID remain.", broadcast=True)

        if demo_context["is_active"] and demo_context["player_soul_id"] and demo_context["current_step"] == 0 and not demo_context["script_execution_started"]:
            if demo_context["script_data"] and game_state['souls'][demo_context["player_soul_id"]]['location_id'] != "LIMBO_VOID":
                send_debug_info(demo_context["player_sid"], "Genesis complete. Spawning demo script execution now.")
                eventlet.spawn_n(execute_demo_script_async)

    except Exception as e:
        error_msg = f"Critical error during initial prompt processing: {e}\n{traceback.format_exc()}"
        print(error_msg)
        log_to_world(f"Server CRITICAL ERROR during initial setup: {e}", broadcast=True)
        send_debug_info(None, error_msg)
        gs_meta['initial_prompt_processing_started'] = False
        gs_meta['initial_prompt_processing_complete'] = False

def get_filtered_game_state_for_soul(soul_id):
    soul = game_state['souls'].get(soul_id)
    if not soul: return {'error': "Soul not found"}

    current_loc_id = soul.get('location_id')
    current_loc_data_for_client = None

    if current_loc_id == "LIMBO_VOID" or current_loc_id is None or current_loc_id not in game_state['locations']:
        void_visual_config = game_state.get('world_properties', {}).get('void_visual_config', {'center_position_xyz': [0,-50,-100], 'ground_type_key': 'none', 'ground_config': {'color_hex': '#100510'}})
        current_loc_data_for_client = {
            'id': current_loc_id if current_loc_id else "LIMBO_VOID",
            'name': game_state.get('world_properties', {}).get('ui_special_location_names',{}).get('limbo_void', "The Void"),
            'description': game_state.get('world_properties', {}).get('ui_messages',{}).get('limbo_void_description', "Drifting in an unformed expanse..."),
            'exits': {},
            'landmarks': {},
            'client_visual_config': void_visual_config,
            'temporary_notes': "None"
        }
        environment_objects_in_location = []
        active_temp_objects_for_client = []
    else:
        raw_loc_data = game_state['locations'][current_loc_id]
        current_loc_data_for_client = {
            'id': raw_loc_data['id'],
            'name': raw_loc_data['name'],
            'description': raw_loc_data['description'],
            'exits': raw_loc_data.get('exits', {}),
            'landmarks': raw_loc_data.get('landmarks', {}),
            'client_visual_config': raw_loc_data.get('client_visual_config', {})
        }

        active_temp_objects_for_client = []
        environment_objects_in_location = []
        stale_temp_ids = []

        for obj_id, obj_data in game_state['temporary_objects'].items():
            if obj_data['location_id'] == current_loc_id:
                if time.time() > obj_data['creation_time'] + obj_data['duration']:
                    stale_temp_ids.append(obj_id)
                else:
                    active_temp_objects_for_client.append({
                        'id': obj_data['id'], 'type': obj_data['type'],
                        'from_landmark_id': obj_data['from_landmark_id'],
                        'to_landmark_id': obj_data['to_landmark_id'],
                        'location_id': obj_data['location_id'],
                        'duration': obj_data['duration'] - (time.time() - obj_data['creation_time']),
                        'original_duration': obj_data['duration'],
                        'client_visual_config': obj_data.get('client_visual_config', {})
                    })

        if stale_temp_ids:
            needs_broadcast = False
            for temp_id in stale_temp_ids:
                removed_obj = game_state['temporary_objects'].pop(temp_id, None)
                if removed_obj:
                    log_to_world(f"{removed_obj['type']} from {removed_obj.get('from_landmark_id','?')} to {removed_obj.get('to_landmark_id','?')} vanished.", broadcast=False)
                    needs_broadcast = True
            if needs_broadcast: broadcast_game_state_to_all_relevant()

        for obj_id, obj_data in game_state['environment_objects'].items():
            if obj_data.get('location_id') == current_loc_id:
                environment_objects_in_location.append(obj_data)

        current_loc_data_for_client['temporary_notes'] = ", ".join([f"{obj['type']} to {obj['to_landmark_id']}" for obj in active_temp_objects_for_client]) if active_temp_objects_for_client else "None"


    return {
        'playerSoul': {'id': soul['id'], 'name': soul['name'], 'locationId': current_loc_id},
        'currentLocation': current_loc_data_for_client,
        'inventory': [{'id': aid, 'name': game_state['artifacts'][aid]['name'],
                       'description': game_state['artifacts'][aid]['description'],
                       'toolName': game_state['artifacts'][aid].get('linked_dynamic_function_name'),
                       'properties': game_state['artifacts'][aid].get('properties', {})
                      } for aid in soul.get('inventory', []) if aid in game_state['artifacts']],
        'worldLog': game_state['world_log'][-game_state.get('world_properties', {}).get('client_default_settings', {}).get('ui', {}).get('log_max_entries', 20):],
        'activeTemporaryObjects': active_temp_objects_for_client,
        'environmentObjectsInLocation': environment_objects_in_location,
        'allPuzzleStates': game_state['puzzle_states'],
        'worldProperties': game_state.get('world_properties', {})
    }

def send_game_state_update(sid, soul_id):
    state = get_filtered_game_state_for_soul(soul_id)
    socketio.emit('gameStateUpdate', state, room=sid)

def broadcast_game_state_to_all_relevant():
    for sid, soul_id in list(connected_souls_by_sid.items()):
        if soul_id in game_state['souls']: send_game_state_update(sid, soul_id)

@app.route('/')
def index(): return send_from_directory(app.static_folder, 'index.html')

@socketio.on('connect')
def handle_connect():
    global game_primitives_handler
    sid = request.sid
    player_soul_id = str(uuid.uuid4())
    gs_meta = game_state['server_metadata']

    next_player_num = gs_meta.get('next_player_number', 1)
    player_name = f"Player_{next_player_num}"
    gs_meta['next_player_number'] = next_player_num + 1

    initial_location_id = "LIMBO_VOID"

    game_state['souls'][player_soul_id] = {
        'id': player_soul_id, 'name': player_name,
        'location_id': initial_location_id,
        'inventory': [], 'type': 'player', 'socket_id': sid
    }
    connected_souls_by_sid[sid] = player_soul_id
    socketio.emit('assignSoulId', player_soul_id, room=sid)
    send_debug_info(sid, f"Player {player_name} ({player_soul_id}) connected. SID {sid}. Initial location: {initial_location_id}")

    if demo_context["is_active"] and not demo_context["player_soul_id"]:
        game_state['souls'][player_soul_id]['name'] = "DemoPlayer"
        player_name = "DemoPlayer"
        demo_context["player_soul_id"] = player_soul_id
        demo_context["player_sid"] = sid
        send_debug_info(sid, f"DemoPlayer connected ({player_soul_id}). SID {sid}.")
        log_to_world(f"{player_name} has entered. Awaiting world's birth or re-entry...", broadcast=True)

    if not gs_meta['initial_prompt_processing_started']:
        log_to_world("Server: World genesis protocol initiating...", broadcast=True)
        eventlet.spawn_n(process_initial_prompt_commands)
    elif gs_meta['initial_prompt_processing_complete']:
        send_debug_info(sid, "World genesis already complete. Finalizing player setup.")
        finalize_player_setup_after_genesis(player_soul_id, sid)
    else:
        log_to_world(f"{player_name} waits as the world forms...", broadcast=True)
        send_debug_info(sid, "World genesis in progress. Player will be fully set up upon completion.")

    send_game_state_update(sid, player_soul_id)

def finalize_player_setup_after_genesis(p_soul_id, p_sid, from_limbo=False):
    send_debug_info(p_sid, f"Finalizing setup for {p_soul_id} post-genesis (from_limbo={from_limbo}).")
    gs = game_state
    soul = gs['souls'].get(p_soul_id)
    if not soul: return

    if soul['location_id'] == "LIMBO_VOID" or soul['location_id'] is None:
        start_location = gs.get('world_properties', {}).get('initial_start_location_id')
        if start_location and start_location in gs['locations']:
            soul['location_id'] = start_location
            log_to_world(f"{soul['name']} materializes in '{gs['locations'][start_location]['name']}'.", broadcast=False)
        else:
            default_loc_key = list(gs['locations'].keys())[0] if gs['locations'] else None
            if default_loc_key:
                soul['location_id'] = default_loc_key
                log_to_world(f"Warning: Initial start location not properly defined. {soul['name']} placed in '{gs['locations'][default_loc_key]['name']}'.", broadcast=False)
            else:
                log_to_world(f"Warning: No locations available. {soul['name']} remains in The Void.", broadcast=False)


    artifacts_to_give_ids = gs.get('world_properties', {}).get('initial_player_artifacts', [])
    if game_primitives_handler and artifacts_to_give_ids:
        for art_id_key in artifacts_to_give_ids:
            if art_id_key in gs['artifacts'] and art_id_key not in soul.get('inventory', []):
                 give_params = {'soul_id': p_soul_id, 'artifact_id': art_id_key}
                 game_primitives_handler.host_give_artifact_to_soul(give_params)
            elif art_id_key not in gs['artifacts']:
                log_to_world(f"Warning: Initial artifact '{art_id_key}' (from world_properties) not found for {p_soul_id}", broadcast=False)

    if p_sid:
        send_game_state_update(p_sid, p_soul_id)

    if demo_context["is_active"] and p_soul_id == demo_context["player_soul_id"] and \
       demo_context["current_step"] == 0 and demo_context["script_data"] and \
       soul['location_id'] != "LIMBO_VOID" and not demo_context["script_execution_started"]:
        send_debug_info(p_sid, "Spawning demo script execution as player is now finalized.")
        eventlet.spawn_n(execute_demo_script_async)


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid; soul_id = connected_souls_by_sid.pop(sid, None)
    if soul_id and soul_id in game_state['souls']:
        log_to_world(f"{game_state['souls'][soul_id]['name']} disconnected.")
        game_state['souls'][soul_id]['socket_id'] = None
        if demo_context["is_active"] and demo_context["player_soul_id"] == soul_id:
            send_debug_info(None, "DemoPlayer disconnected. Stopping demo.")
            demo_context["is_active"] = False
            demo_context["player_soul_id"] = None; demo_context["player_sid"] = None
            demo_context["current_step"] = 0; demo_context["orb_catalyst_artifact_id_pending_creation"] = None
            demo_context["newly_created_tool_function_name_pending_artifact"] = None
            demo_context["user_description_pending_artifact"] = None
            demo_context["script_execution_started"] = False
            demo_context["current_focused_landmark_key"] = None

@socketio.on('performAction')
def handle_perform_action(data):
    sid = request.sid
    if demo_context["is_active"] and demo_context["player_sid"] == sid and demo_context["script_data"] and demo_context["current_step"] < len(demo_context["script_data"]) and demo_context["script_data"][demo_context["current_step"]]["action"] != "USE_ARTIFACT":
        socketio.emit('actionResult', {'success': False, 'message': "DemoPlayer actions are scripted. Please wait for script completion or next interaction point."}, room=sid)
        return

    player_soul_id = connected_souls_by_sid.get(sid)
    if not player_soul_id or player_soul_id not in game_state['souls']:
        socketio.emit('actionResult', {'success': False, 'message': "Soul not recognized."}, room=sid); return

    artifact_id = data.get('artifactId'); client_args = data.get('args', {})
    result_message, success_flag, event_type, event_data = _internal_perform_action_logic(player_soul_id, artifact_id, client_args)

    response = {'success': success_flag, 'message': str(result_message)}
    if event_type:
        response['event'] = event_type
        response.update(event_data)

    socketio.emit('actionResult', response, room=sid)
    broadcast_game_state_to_all_relevant()

@socketio.on('playerMove')
def handle_player_move(data):
    sid = request.sid
    player_soul_id = connected_souls_by_sid.get(sid)
    if not player_soul_id or player_soul_id not in game_state['souls']:
        socketio.emit('actionResult', {'success': False, 'message': "Soul not recognized for movement."}, room=sid)
        return

    soul = game_state['souls'][player_soul_id]
    current_loc_id = soul.get('location_id')
    exit_landmark_key = data.get('exit_key')

    if not current_loc_id or current_loc_id not in game_state['locations'] or \
       not exit_landmark_key or exit_landmark_key not in game_state['locations'][current_loc_id].get('exits', {}):
        socketio.emit('actionResult', {'success': False, 'message': "Invalid move or exit not found."}, room=sid)
        send_debug_info(sid, f"Failed move: current_loc_id={current_loc_id}, exit_landmark_key={exit_landmark_key}, available_exits={game_state['locations'].get(current_loc_id,{}).get('exits', {})}")
        return

    target_loc_id = game_state['locations'][current_loc_id]['exits'][exit_landmark_key]

    if target_loc_id not in game_state['locations']:
        socketio.emit('actionResult', {'success': False, 'message': f"Target location '{target_loc_id}' for exit '{exit_landmark_key}' does not exist."}, room=sid)
        return

    # Puzzle Check for Player Movement (Real Player)
    can_use_exit = True
    exit_landmark_full_config = game_state['locations'][current_loc_id]['landmarks'].get(exit_landmark_key, {}).get('client_visual_config', {})
    linked_puzzle_id = exit_landmark_full_config.get('linked_puzzle_id_for_open_state')
    puzzle_message = ""

    if linked_puzzle_id:
        puzzle_state = game_state['puzzle_states'].get(linked_puzzle_id)
        if puzzle_state and not puzzle_state.get('is_complete', False) and not puzzle_state.get('is_open', False):
            can_use_exit = False
            puzzle_message = puzzle_state.get('custom_sealed_message', f"{exit_landmark_full_config.get('display_name', exit_landmark_key)} is sealed.")

    if not can_use_exit:
        socketio.emit('actionResult', {'success': False, 'message': puzzle_message}, room=sid)
        log_to_world(f"{soul['name']} tried to use exit '{exit_landmark_full_config.get('display_name', exit_landmark_key)}', but: {puzzle_message}", broadcast=False)
        return

    target_loc_name = game_state['locations'][target_loc_id].get('name', 'an unknown area')
    current_loc_name = game_state['locations'][current_loc_id].get('name', 'somewhere')

    soul['location_id'] = target_loc_id
    log_to_world(f"{soul['name']} moved from {current_loc_name} to {target_loc_name}.")
    demo_context['current_focused_landmark_key'] = None # Reset focus for any player moving

    socketio.emit('actionResult', {'success': True, 'message': f"Moved to {target_loc_name}."}, room=sid)
    broadcast_game_state_to_all_relevant()


def _internal_perform_action_logic(player_soul_id, artifact_id_or_name, action_args, artifact_name_starts_with=False):
    soul = game_state['souls'][player_soul_id]
    if soul['location_id'] == "LIMBO_VOID" or soul['location_id'] is None:
        return "Cannot perform actions while in The Void.", False, None, {}

    artifact = None; actual_artifact_id = None; event_type = None; event_data = {}

    if artifact_name_starts_with:
        for inv_art_id in soul.get('inventory', []):
            art_data = game_state['artifacts'].get(inv_art_id)
            if art_data and art_data['name'].startswith(artifact_id_or_name):
                artifact = art_data; actual_artifact_id = inv_art_id; break
        if not artifact: return f"Artifact like '{artifact_id_or_name}' not found.", False, None, {}
    else:
        actual_artifact_id = artifact_id_or_name
        artifact = game_state['artifacts'].get(actual_artifact_id)
        if not artifact :
            for inv_art_id in soul.get('inventory', []):
                art_data = game_state['artifacts'].get(inv_art_id)
                if art_data and art_data['name'] == artifact_id_or_name:
                    artifact = art_data; actual_artifact_id = inv_art_id; break
            if not artifact: return f"Artifact ID/name '{artifact_id_or_name}' not found.", False, None, {}


    if actual_artifact_id not in soul.get('inventory', []): return "Artifact not possessed.", False, None, {}
    dynamic_function_name = artifact.get('linked_dynamic_function_name')
    if not dynamic_function_name: return "Artifact is inert.", False, None, {}

    params_for_function = {
        'soul_id': player_soul_id, 'location_id': soul['location_id'],
        'artifact_id': actual_artifact_id, 'artifact_properties': artifact.get('properties', {}),
        **action_args
    }
    send_debug_info(soul.get('socket_id'), f"Soul {player_soul_id} using '{artifact['name']}' ({dynamic_function_name}) with args: {json.dumps(params_for_function)[:100]}")
    execution_result = dynamic_executor.execute_dynamic_function(dynamic_function_name, params_for_function, get_external_apis_for_execution())
    send_debug_info(soul.get('socket_id'), f"Dynamic func '{dynamic_function_name}' result: {execution_result}")

    success = not (isinstance(execution_result, str) and execution_result.startswith("Error:"))

    if execution_result == "EVENT:PROMPT_USER_FOR_TOOL_DESCRIPTION":
        if demo_context["is_active"] and demo_context["player_soul_id"] == player_soul_id:
            demo_context["orb_catalyst_artifact_id_pending_creation"] = actual_artifact_id

        ui_messages = game_state.get('world_properties', {}).get('ui_messages', {})
        display_msg = ui_messages.get('orb_tool_prompt_initiate', "The Orb of Ingenuity awaits your command...")

        event_type = "PROMPT_USER_FOR_TOOL_DESCRIPTION"
        event_data = {
            'prompt_for_tool_artifact_id': actual_artifact_id,
            'display_message': display_msg
            }
        return execution_result, True, event_type, event_data
    else:
        try:
            parsed_result = json.loads(execution_result)
            user_message = parsed_result.get('message', execution_result)
            if 'error' in parsed_result: success = False; user_message = parsed_result['error']
            return user_message, success, None, {}
        except (json.JSONDecodeError, TypeError):
            return execution_result, success, None, {}


@socketio.on('submitToolDescription')
def handle_submit_tool_description(data):
    sid = request.sid
    if demo_context["is_active"] and demo_context["player_sid"] == sid and demo_context["script_data"] and demo_context["script_data"][demo_context["current_step"]]["action"] != "DESCRIBE_TOOL":
        log_to_world("[DEMO WARNING] DemoPlayer tool description submission ignored (script handles it or not at that step).", broadcast=True)
        return

    player_soul_id = connected_souls_by_sid.get(sid)
    if not player_soul_id: return
    if game_state['souls'][player_soul_id]['location_id'] == "LIMBO_VOID" or game_state['souls'][player_soul_id]['location_id'] is None:
        socketio.emit('actionResult', {'success': False, 'message': "Cannot describe tools while in The Void."}, room=sid)
        return

    description = data.get('description'); catalyst_artifact_id = data.get('catalyst_artifact_id')
    message, success = _internal_submit_tool_description_logic(player_soul_id, description, catalyst_artifact_id)

    socketio.emit('actionResult', {'success': success, 'message': message}, room=sid)
    broadcast_game_state_to_all_relevant()

def _internal_submit_tool_description_logic(player_soul_id, description, catalyst_artifact_id):
    soul = game_state['souls'][player_soul_id]
    catalyst_artifact = game_state['artifacts'].get(catalyst_artifact_id)
    if not description or not catalyst_artifact_id: return "Missing description or catalyst_artifact_id for tool creation.", False
    if not catalyst_artifact: return f"Catalyst artifact {catalyst_artifact_id} not found.", False

    new_tool_func_name = f"df_user_{player_soul_id[:4]}_{str(uuid.uuid4())[:4]}"

    tool_creation_template = game_state.get('world_properties', {}).get(
        'tool_creation_prompt_template',
        "Player described: '{description}'. This function implements that tool. Use available host APIs: {api_list}."
    )

    tool_func_desc_for_llm = tool_creation_template.format(
        description=description,
        api_list=generate_api_description_for_llm_prompt()
    )

    tool_func_params_schema = {"type": "object", "properties": {}}

    tool_func_creation_args = {
        'new_function_name': new_tool_func_name,
        'new_function_description': tool_func_desc_for_llm,
        'new_function_parameters_schema': tool_func_params_schema,
        'host_provided_api_description_for_new_func': generate_api_description_for_llm_prompt()
    }
    send_debug_info(soul.get('socket_id'), f"Creating tool func '{new_tool_func_name}' for: {description}")
    tool_func_creation_result = dynamic_executor.execute_dynamic_function("create_dynamic_function", tool_func_creation_args, get_external_apis_for_execution())

    if isinstance(tool_func_creation_result, str) and tool_func_creation_result.startswith("Error:"):
        log_to_world(f"{soul['name']}'s Orb tool func creation failed: {tool_func_creation_result}", broadcast=True)
        return f"Orb fizzles (tool func): {tool_func_creation_result}", False
    else:
        send_debug_info(soul.get('socket_id'), f"Tool func '{new_tool_func_name}' created. Now creating charged artifact using 'df_system_finalize_orb_charging'.")

        finalize_params = {
            'soul_id': player_soul_id,
            'catalyst_artifact_id': catalyst_artifact_id,
            'newly_created_tool_function_name': new_tool_func_name,
            'user_provided_description': description
        }
        finalization_result_str = dynamic_executor.execute_dynamic_function("df_system_finalize_orb_charging", finalize_params, get_external_apis_for_execution())

        send_debug_info(soul.get('socket_id'), f"Result of df_system_finalize_orb_charging: {finalization_result_str}")

        if isinstance(finalization_result_str, str) and finalization_result_str.startswith("Error:"):
             log_to_world(f"{soul['name']}'s Orb artifact finalization failed: {finalization_result_str}", broadcast=True)
             return f"Orb created tool func, but artifact finalization failed: {finalization_result_str}", False

        try:
            parsed_final_result = json.loads(finalization_result_str)
            if "error" in parsed_final_result:
                log_to_world(f"{soul['name']}'s Orb artifact finalization failed: {parsed_final_result['error']}", broadcast=True)
                return f"Orb finalization error: {parsed_final_result['error']}", False
            else:
                final_msg = parsed_final_result.get("message", "Orb resonates with new power!")
                log_to_world(f"{soul['name']}: {final_msg}", broadcast=True)

                if demo_context["is_active"] and demo_context["player_soul_id"] == player_soul_id:
                    demo_context["orb_catalyst_artifact_id_pending_creation"] = None
                    demo_context["newly_created_tool_function_name_pending_artifact"] = None
                    demo_context["user_description_pending_artifact"] = None
                return final_msg, True
        except json.JSONDecodeError:
            log_to_world(f"{soul['name']}'s Orb artifact finalization returned invalid JSON: {finalization_result_str}", broadcast=True)
            return "Orb finalization failed to return valid status.", False


@socketio.on('requestState')
def handle_request_state():
    sid = request.sid; player_soul_id = connected_souls_by_sid.get(sid)
    if player_soul_id and player_soul_id in game_state['souls']:
        send_game_state_update(sid, player_soul_id)

def save_game_state():
    print("[SERVER] Saving game state...")
    try:
        for key_to_ensure in ['world_properties', 'world_event_handlers']:
            if key_to_ensure not in game_state:
                game_state[key_to_ensure] = {}
        with open(SAVE_FILE, 'w') as f: json.dump(game_state, f, indent=2)
        print("[SERVER] Game state saved.")
    except Exception as e: print(f"[SERVER] Error saving game state: {e}\n{traceback.format_exc()}")

def load_game_state():
    global game_state
    if os.path.exists(SAVE_FILE):
        print("[SERVER] Loading game state from save...")
        try:
            with open(SAVE_FILE, 'r') as f: loaded_data = json.load(f)
            for key in ['souls', 'locations', 'artifacts', 'environment_objects', 'puzzle_states', 'temporary_objects', 'world_log', 'server_metadata', 'world_properties', 'world_event_handlers']:
                if key not in loaded_data:
                    default_val = {}
                    if key == 'world_log': default_val = []
                    elif key == 'server_metadata': default_val = {'next_player_number': 1, 'initial_prompt_processing_started': False, 'initial_prompt_processing_complete': False}
                    loaded_data[key] = default_val

            if 'initial_prompt_processing_started' not in loaded_data['server_metadata']: loaded_data['server_metadata']['initial_prompt_processing_started'] = False
            if 'initial_prompt_processing_complete' not in loaded_data['server_metadata']: loaded_data['server_metadata']['initial_prompt_processing_complete'] = False
            if 'next_player_number' not in loaded_data['server_metadata']: loaded_data['server_metadata']['next_player_number'] = 1


            game_state.update(loaded_data)
            print("[SERVER] Game state loaded.")
        except Exception as e:
            print(f"[SERVER] Error loading game state: {e}. Starting with default.")
            game_state['server_metadata'] = {'next_player_number': 1, 'initial_prompt_processing_started': False, 'initial_prompt_processing_complete': False}
            game_state['world_properties'] = {}
            game_state['world_event_handlers'] = {}
    else:
        print("[SERVER] No save file found. Starting with default empty state.")
        game_state['server_metadata'] = {'next_player_number': 1, 'initial_prompt_processing_started': False, 'initial_prompt_processing_complete': False}
        game_state['world_properties'] = {}
        game_state['world_event_handlers'] = {}


def signal_handler_save_on_exit(sig, frame):
    print('[SERVER] SIGINT received, saving game state before exit...')
    save_game_state(); sys.exit(0)

def trigger_llm_repair_process(function_name, error, traceback_str, params_for_function, host_api_description_for_repair_context):
    send_debug_info(None, f"LLM REPAIR TRIGGERED: For function '{function_name}' due to error: {error}")

def load_demo_script(filepath):
    try:
        with open(filepath, 'r') as f: script_data = json.load(f)
        send_debug_info(None, f"[DEMO] Successfully loaded demo script: {filepath}")
        return script_data
    except Exception as e:
        send_debug_info(None, f"[DEMO] Error loading demo script {filepath}: {e}")
        return None

def _internal_move_player_for_demo(player_soul_id, target_location_id):
    if player_soul_id in game_state['souls'] and target_location_id in game_state['locations']:
        current_location_name = game_state['locations'].get(game_state['souls'][player_soul_id]['location_id'], {}).get('name', 'Unknown')
        target_location_name = game_state['locations'][target_location_id].get('name', target_location_id)
        game_state['souls'][player_soul_id]['location_id'] = target_location_id
        log_to_world(f"[DEMO] {game_state['souls'][player_soul_id]['name']} moved from '{current_location_name}' to '{target_location_name}'.", broadcast=True)
        demo_context['current_focused_landmark_key'] = None # Reset focus on location change
        return True
    log_to_world(f"[DEMO] Failed to move {player_soul_id} to {target_location_id}.", broadcast=True)
    return False

def execute_demo_script_async():
    if demo_context.get("script_execution_started", False):
        send_debug_info(demo_context.get("player_sid"), "[DEMO] Script execution already initiated by another task. This spawn will exit.")
        return
    demo_context["script_execution_started"] = True
    demo_context["current_focused_landmark_key"] = None # Ensure focus is reset at start of script run

    send_debug_info(demo_context.get("player_sid"), "[DEMO] Starting script execution...")
    log_to_world("[DEMO] Demo sequence initiated.", broadcast=True)

    try:
        while demo_context["is_active"] and demo_context["script_data"] and \
              demo_context["current_step"] < len(demo_context["script_data"]):
            if not demo_context["player_soul_id"] or demo_context["player_soul_id"] not in game_state["souls"]:
                send_debug_info(None, "[DEMO] DemoPlayer not found. Stopping script.")
                demo_context["is_active"] = False; break

            if game_state["souls"][demo_context["player_soul_id"]]['location_id'] == "LIMBO_VOID":
                send_debug_info(demo_context["player_sid"], "[DEMO] Waiting for player to exit LIMBO_VOID before continuing script...")
                eventlet.sleep(1); continue

            step = demo_context["script_data"][demo_context["current_step"]]
            action = step.get("action"); log_msg = step.get("log_message", f"Demo step {demo_context['current_step'] + 1}: {action}")
            send_debug_info(demo_context.get("player_sid"), f"[DEMO] {log_msg}")
            log_to_world(f"[DEMO] {log_msg}", broadcast=True)

            if action == "WAIT": eventlet.sleep(step.get("duration_ms", 1000) / 1000.0)
            elif action == "MOVE_PLAYER_TO": # This is a direct teleport between major locations for demo script
                _internal_move_player_for_demo(demo_context["player_soul_id"], step.get("target_location_id"))
                broadcast_game_state_to_all_relevant()
                eventlet.sleep(0.5)
            elif action == "DEMO_FOCUS_ON_LANDMARK":
                landmark_to_focus_on = step.get("landmark_key") # Can be null to unfocus
                demo_context["current_focused_landmark_key"] = landmark_to_focus_on
                focused_name = "general area"
                if landmark_to_focus_on:
                    current_loc_id = game_state['souls'][demo_context["player_soul_id"]]['location_id']
                    if current_loc_id in game_state['locations'] and \
                       landmark_to_focus_on in game_state['locations'][current_loc_id].get('landmarks', {}):
                        focused_name = game_state['locations'][current_loc_id]['landmarks'][landmark_to_focus_on].get('name', landmark_to_focus_on)
                # Log message is handled by the script step itself.
                send_debug_info(demo_context["player_sid"], f"[DEMO] DemoPlayer logically focused on: {focused_name} ({landmark_to_focus_on})")
            elif action == "DEMO_USE_EXIT":
                exit_key_to_use = step.get("exit_landmark_key")
                player_soul_id = demo_context["player_soul_id"]
                soul = game_state['souls'][player_soul_id]
                current_loc_id = soul.get('location_id')

                if not exit_key_to_use:
                    log_to_world(f"[DEMO ERROR] 'exit_landmark_key' not specified for DEMO_USE_EXIT.", broadcast=True)
                    send_debug_info(demo_context["player_sid"], f"[DEMO ERROR] 'exit_landmark_key' not specified for DEMO_USE_EXIT.")
                    demo_context["is_active"] = False; break

                required_landmark_name_for_log = exit_key_to_use
                if current_loc_id in game_state['locations'] and \
                    exit_key_to_use in game_state['locations'][current_loc_id].get('landmarks', {}):
                    required_landmark_name_for_log = game_state['locations'][current_loc_id]['landmarks'][exit_key_to_use].get('name', exit_key_to_use)


                if demo_context.get("current_focused_landmark_key") != exit_key_to_use:
                    log_to_world(f"[DEMO ACTION FAIL] DemoPlayer must first focus on landmark '{required_landmark_name_for_log}' to use this exit. Currently focused on '{demo_context.get('current_focused_landmark_key')}'.", broadcast=True)
                    send_debug_info(demo_context["player_sid"], f"Demo action DEMO_USE_EXIT failed: incorrect focus for exit {exit_key_to_use}.")
                else:
                    if not current_loc_id or current_loc_id not in game_state['locations'] or \
                       not exit_key_to_use or exit_key_to_use not in game_state['locations'][current_loc_id].get('exits', {}):
                        log_to_world(f"[DEMO ERROR] Invalid exit '{exit_key_to_use}' from location '{current_loc_id}' for DEMO_USE_EXIT.", broadcast=True)
                        send_debug_info(demo_context["player_sid"], f"[DEMO ERROR] Invalid exit '{exit_key_to_use}' for DEMO_USE_EXIT.")
                        demo_context["is_active"] = False; break

                    can_use_exit_server_side = True
                    exit_landmark_full_config = game_state['locations'][current_loc_id]['landmarks'].get(exit_key_to_use, {}).get('client_visual_config', {})
                    linked_puzzle_id = exit_landmark_full_config.get('linked_puzzle_id_for_open_state')

                    if linked_puzzle_id:
                        puzzle_state = game_state['puzzle_states'].get(linked_puzzle_id)
                        if puzzle_state and not puzzle_state.get('is_complete', False) and not puzzle_state.get('is_open', False):
                            can_use_exit_server_side = False
                            puzzle_message = puzzle_state.get('custom_sealed_message', f"{exit_landmark_full_config.get('display_name', exit_key_to_use)} is sealed.")
                            log_to_world(f"[DEMO] DemoPlayer tried to use exit '{exit_key_to_use}', but: {puzzle_message}", broadcast=True)
                            send_debug_info(demo_context["player_sid"], f"[DEMO] Exit '{exit_key_to_use}' is sealed: {puzzle_message}")

                    if can_use_exit_server_side:
                        target_loc_id = game_state['locations'][current_loc_id]['exits'][exit_key_to_use]
                        if target_loc_id not in game_state['locations']:
                            log_to_world(f"[DEMO ERROR] Target location '{target_loc_id}' for exit '{exit_key_to_use}' does not exist.", broadcast=True)
                            send_debug_info(demo_context["player_sid"], f"[DEMO ERROR] Target location '{target_loc_id}' for DEMO_USE_EXIT does not exist.")
                            demo_context["is_active"] = False; break

                        target_loc_name = game_state['locations'][target_loc_id].get('name', 'an unknown area')
                        current_loc_name = game_state['locations'][current_loc_id].get('name', 'somewhere')
                        soul['location_id'] = target_loc_id
                        log_to_world(f"[DEMO] {soul['name']} used exit '{exit_landmark_full_config.get('display_name', exit_key_to_use)}' and moved from {current_loc_name} to {target_loc_name}.", broadcast=True)
                        send_debug_info(demo_context["player_sid"], f"DemoPlayer moved to {target_loc_id} via exit {exit_key_to_use}.")
                        demo_context['current_focused_landmark_key'] = None
                    else:
                         pass # Log message about sealed exit already handled
            elif action == "USE_ARTIFACT":
                name_to_use = step.get("artifact_name", step.get("artifact_name_starts_with"))
                is_prefix = bool(step.get("artifact_name_starts_with"))
                action_args = step.get("args", {})
                player_soul_id = demo_context["player_soul_id"]
                soul = game_state['souls'][player_soul_id]
                current_loc_id = soul['location_id']

                # Demo Player Focus Check for USE_ARTIFACT
                target_env_object_id = action_args.get("target_env_object_id")
                required_landmark_key_for_action = None
                required_landmark_name_for_log = "specific target"

                if target_env_object_id and current_loc_id in game_state['locations']:
                    for lm_key, lm_data in game_state['locations'][current_loc_id].get('landmarks', {}).items():
                        if lm_data.get('client_visual_config',{}).get('targetable_as_env_object_id') == target_env_object_id:
                            required_landmark_key_for_action = lm_key
                            required_landmark_name_for_log = lm_data.get('name', lm_key)
                            break
                
                if required_landmark_key_for_action and demo_context.get("current_focused_landmark_key") != required_landmark_key_for_action:
                    log_to_world(f"[DEMO ACTION FAIL] DemoPlayer must first focus on landmark '{required_landmark_name_for_log}' to use '{name_to_use}' on it. Currently focused on '{demo_context.get('current_focused_landmark_key')}'.", broadcast=True)
                    send_debug_info(demo_context["player_sid"], f"Demo action USE_ARTIFACT failed: incorrect focus for target {target_env_object_id}.")
                else:
                    # Focus matches or no specific landmark target needed for the artifact
                    if target_env_object_id : # Log focusing if it's a targeted action
                         log_to_world(f"[DEMO] DemoPlayer focusing on {required_landmark_name_for_log} to use {name_to_use}.", broadcast=False)

                    res_msg, success, event_type, event_data = _internal_perform_action_logic(player_soul_id, name_to_use, action_args, artifact_name_starts_with=is_prefix)
                    log_to_world(f"[DEMO] Action result: {res_msg}", broadcast=True)
                    if event_type == "PROMPT_USER_FOR_TOOL_DESCRIPTION":
                         demo_context["orb_catalyst_artifact_id_pending_creation"] = event_data['prompt_for_tool_artifact_id']
                    elif not success:
                        send_debug_info(demo_context["player_sid"], f"[DEMO] Error in USE_ARTIFACT: {res_msg}. Stopping demo.")
                        demo_context["is_active"] = False; break
            elif action == "DESCRIBE_TOOL":
                catalyst_id = demo_context.get("orb_catalyst_artifact_id_pending_creation")
                if not catalyst_id:
                    player_inv = game_state['souls'][demo_context["player_soul_id"]]['inventory']
                    orb_artifact_id_from_wp = game_state.get('world_properties',{}).get('orb_of_ingenuity_artifact_id', 'orb_01')
                    for art_id in player_inv:
                        if game_state['artifacts'][art_id]['id'] == orb_artifact_id_from_wp:
                            catalyst_id = art_id; break
                if not catalyst_id:
                    send_debug_info(demo_context["player_sid"], "[DEMO] Error: DESCRIBE_TOOL no catalyst pending or Orb of Ingenuity found. Stopping demo.")
                    demo_context["is_active"] = False; break

                res_msg, success = _internal_submit_tool_description_logic(demo_context["player_soul_id"], step.get("description"), catalyst_id)
                log_to_world(f"[DEMO] Tool description result: {res_msg}", broadcast=True)
                if not success:
                    send_debug_info(demo_context["player_sid"], f"[DEMO] Error in DESCRIBE_TOOL: {res_msg}. Stopping demo.")
                    demo_context["is_active"] = False; break
            elif action == "COMMENT": pass

            demo_context["current_step"] += 1
            broadcast_game_state_to_all_relevant()
            eventlet.sleep(0.2)

        if demo_context["is_active"]:
            send_debug_info(demo_context.get("player_sid"), "[DEMO] Script finished.")
            log_to_world("[DEMO] Demo sequence complete.", broadcast=True)
    finally:
        demo_context["script_execution_started"] = False
        if not demo_context["is_active"]:
            demo_context["current_step"] = 0
            demo_context["orb_catalyst_artifact_id_pending_creation"] = None
            demo_context["newly_created_tool_function_name_pending_artifact"] = None
            demo_context["user_description_pending_artifact"] = None
            demo_context["current_focused_landmark_key"] = None


def main():
    global game_state, game_primitives_handler, demo_context
    parser = argparse.ArgumentParser(description="Orb of Ingenuity Demo Server")
    parser.add_argument('--recreate', action='store_true', help='Recreate world, deleting save and dynamic functions.')
    parser.add_argument('--demo', type=str, metavar='SCRIPT_FILE', help='Run in demo mode with specified JSON script.')
    args = parser.parse_args()

    default_game_state_metadata = {'next_player_number': 1, 'initial_prompt_processing_started': False, 'initial_prompt_processing_complete': False}
    default_game_state_core = {'souls': {}, 'locations': {}, 'artifacts': {}, 'environment_objects': {}, 'puzzle_states': {}, 'temporary_objects': {}, 'world_log': ["Welcome!"], 'world_properties': {}, 'world_event_handlers': {}}

    if args.demo:
        demo_script_content = load_demo_script(args.demo)
        if demo_script_content:
            demo_context["is_active"] = True; demo_context["script_path"] = args.demo
            demo_context["script_data"] = demo_script_content; demo_context["current_step"] = 0
            demo_context["script_execution_started"] = False
            demo_context["current_focused_landmark_key"] = None
            print(f"[SERVER] Demo mode activated with script: {args.demo}")
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
        game_state = {**default_game_state_core, 'server_metadata': default_game_state_metadata.copy()}
        game_state['world_log'] = ["Welcome (World Recreated)!"]
    else:
        load_game_state()

    game_primitives_handler = GamePrimitiveHandler(get_current_game_state, socketio)

    register_host_api_for_llm('host_core_add_location_to_gamestate', "Core API: Creates a new location definition. Landmarks added separately.", {'type':'object', 'properties': {'id':{'type':'string'}, 'name':{'type':'string'}, 'description':{'type':'string'}}, 'required':['id','name','description']}, game_primitives_handler.host_core_add_location_to_gamestate)
    register_host_api_for_llm('host_core_add_artifact_to_gamestate', "Core API: Creates a new artifact definition. 'properties' can include 'client_interaction_rules'.", {'type':'object', 'properties': {'id':{'type':'string'}, 'name':{'type':'string'}, 'description':{'type':'string'}, 'properties':{'type':'object'}, 'linked_dynamic_function_name':{'type':'string'}}, 'required':['id','name','description']}, game_primitives_handler.host_core_add_artifact_to_gamestate)
    register_host_api_for_llm('host_give_artifact_to_soul', "Core API: Gives an existing artifact (by ID) to a soul (by ID). Returns JSON.", {'type':'object', 'properties': {'soul_id':{'type':'string'}, 'artifact_id':{'type':'string'}}, 'required':['soul_id', 'artifact_id']}, game_primitives_handler.host_give_artifact_to_soul)
    register_host_api_for_llm('host_core_add_env_object_to_gamestate', "Core API: Creates an environment object. 'details' can include initial 'client_visual_update'.", {'type':'object', 'properties': {'id':{'type':'string'}, 'location_id':{'type':'string'}, 'type':{'type':'string'}, 'details':{'type':'object'}}, 'required':['id','location_id','type']}, game_primitives_handler.host_core_add_env_object_to_gamestate)
    register_host_api_for_llm('host_core_initialize_puzzle_state', "Core API: Initializes a puzzle's state. 'initial_state' can include 'checking_dynamic_function_name'.", {'type':'object', 'properties': {'id':{'type':'string'}, 'initial_state':{'type':'object'}}, 'required':['id','initial_state']}, game_primitives_handler.host_core_initialize_puzzle_state)

    register_host_api_for_llm('host_set_world_property', "Core API: Sets a global world property (e.g., 'initial_start_location_id', 'ui_messages', 'client_default_settings'). Returns JSON.", {'type':'object', 'properties': {'property_name':{'type':'string'}, 'property_value':{'type':'any'}}, 'required':['property_name', 'property_value']}, game_primitives_handler.host_set_world_property)
    register_host_api_for_llm('host_set_location_visual_config', "Client API: Sets visual configuration for a location. 'config' includes 'center_position_xyz', 'ground_type_key', and 'ground_config' (which has type-specific params like 'size_xz', 'color_hex', 'player_platform_size_xyz', etc.). Returns JSON.", {'type':'object', 'properties': {'location_id':{'type':'string'}, 'config':{'type':'object', 'description': "Full visual config for the location, including ground type and its parameters."}}, 'required':['location_id', 'config']}, game_primitives_handler.host_set_location_visual_config)
    register_host_api_for_llm('host_set_landmark_visual_config', "Client API: Sets visual and semantic config for a landmark. 'config' includes 'display_name', 'relative_position_xyz', 'geometry_config' (type, parameters/dimensions/radius), 'material_config' (base_color_hex), 'targetable_as_env_object_id', 'is_exit_to_location_id', 'landmark_interaction_type_key', 'linked_puzzle_id_for_open_state'. Returns JSON.", {'type':'object', 'properties': {'location_id':{'type':'string'}, 'landmark_key':{'type':'string'}, 'config':{'type':'object', 'description': "Full visual and semantic config for the landmark."}}, 'required':['location_id', 'landmark_key', 'config']}, game_primitives_handler.host_set_landmark_visual_config)

    register_host_api_for_llm('host_log_message_to_world', "Core API: Logs a message to the global game world log.", {'type':'object', 'properties': {'message':{'type':'string'}}, 'required':['message']}, game_primitives_handler.host_log_message_to_world)
    register_host_api_for_llm('host_apply_effect_on_environment_object', "Applies an effect to an environment object's 'details'. 'effect_details' can include 'client_visual_update'. Returns JSON.", {'type':'object', 'properties': {'object_id':{'type':'string'}, 'effect_details':{'type':'object'}}, 'required':['object_id', 'effect_details']}, game_primitives_handler.host_apply_effect_on_environment_object)
    register_host_api_for_llm('host_check_puzzle_condition', "Checks puzzle conditions by calling a registered dynamic function. Returns JSON: {'condition_met': bool, 'message': str}.", {'type':'object', 'properties': {'puzzle_id':{'type':'string'}}, 'required':['puzzle_id']}, game_primitives_handler.host_check_puzzle_condition)
    register_host_api_for_llm('host_trigger_world_event', "Triggers a world event by calling a registered dynamic event handler. Returns JSON.", {'type':'object', 'properties': {'event_id':{'type':'string'}, 'soul_id':{'type':'string', 'description':'ID of soul triggering event, optional.'}, 'event_params': {'type':'object', 'description':'Custom parameters for the event handler.'}}, 'required':['event_id']}, game_primitives_handler.host_trigger_world_event)
    register_host_api_for_llm('host_create_temporary_object', "Creates a temporary object (e.g. light bridge). 'client_visual_config' specifies appearance. Returns JSON.", {'type':'object', 'properties': {'type':{'type':'string'},'duration':{'type':'integer'},'from_landmark_id': {'type':'string'},'to_landmark_id':{'type':'string'},'location_id':{'type':'string'}, 'soul_id':{'type':'string'}, 'client_visual_config':{'type':'object', 'description': "Visual config like geometry, material, dimensions."}}, 'required':['type', 'to_landmark_id', 'location_id', 'soul_id', 'client_visual_config']}, game_primitives_handler.host_create_temporary_object)

    register_host_api_for_llm('host_get_entity_data', "Retrieves entity (soul) data. Returns JSON.", {'type':'object', 'properties': {'entity_id':{'type':'string'}}, 'required':['entity_id']}, game_primitives_handler.host_get_entity_data)
    register_host_api_for_llm('host_get_location_data', "Retrieves basic location data (name, desc, exits). Returns JSON.", {'type':'object', 'properties': {'location_id':{'type':'string'}}, 'required':['location_id']}, game_primitives_handler.host_get_location_data)
    register_host_api_for_llm('host_get_environment_object_data', "Retrieves env object data (type, location, details). Returns JSON.", {'type':'object', 'properties': {'object_id':{'type':'string'}}, 'required':['object_id']}, game_primitives_handler.host_get_environment_object_data)

    register_host_api_for_llm('host_register_puzzle_check_function', "Registers a dynamic function to check a puzzle's condition. Returns JSON.", {'type':'object', 'properties': {'puzzle_id':{'type':'string'}, 'checking_dynamic_function_name':{'type':'string'}}, 'required':['puzzle_id', 'checking_dynamic_function_name']}, game_primitives_handler.host_register_puzzle_check_function)
    register_host_api_for_llm('host_register_event_handler_function', "Registers a dynamic function to handle a world event. Returns JSON.", {'type':'object', 'properties': {'event_id':{'type':'string'}, 'handler_dynamic_function_name':{'type':'string'}}, 'required':['event_id', 'handler_dynamic_function_name']}, game_primitives_handler.host_register_event_handler_function)
    register_host_api_for_llm('host_set_puzzle_properties', "Updates properties of an existing puzzle state. Returns JSON.", {'type':'object', 'properties': {'puzzle_id':{'type':'string'}, 'properties':{'type':'object'}}, 'required':['puzzle_id', 'properties']}, game_primitives_handler.host_set_puzzle_properties)


    if not demo_context["is_active"] and not game_state['server_metadata']['initial_prompt_processing_started']:
        send_debug_info(None, "Non-demo mode: Processing initial prompt at startup if not already done.")
        if not game_state['world_properties']: game_state['world_properties'] = {}
        if not game_state['world_event_handlers']: game_state['world_event_handlers'] = {}
        process_initial_prompt_commands()
    elif demo_context["is_active"] and (args.recreate or not game_state['server_metadata']['initial_prompt_processing_complete']):
         game_state['world_log'] = [f"[DEMO] Server awaiting DemoPlayer. World will be born live..."]


    signal.signal(signal.SIGINT, signal_handler_save_on_exit)
    print(f"Server starting on http://0.0.0.0:{PORT}")
    socketio.run(app, host='0.0.0.0', port=PORT, debug=False, use_reloader=False)

if __name__ == '__main__':
    main()

